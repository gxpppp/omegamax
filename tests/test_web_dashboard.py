"""Tests for the web training dashboard (omigamax.web.dashboard).

Covers:
  * /api/train JSONL parsing (train_step + eval_gate aggregation)
  * /api/games npz metadata listing
  * /api/games/<id> replay move derivation (matches the stored s planes:
    each derived move adds exactly one stone, colours alternate, board grids
    are reconstructed exactly)
  * / serves the dashboard HTML
  * missing-data robustness (empty dir / missing log -> 200 + empty lists)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omigamax.web.dashboard import create_app, parse_train_log, reconstruct_game

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def write_train_log(path: Path, lines: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for ev in lines:
            fh.write(json.dumps(ev, sort_keys=True) + "\n")


TRAIN_FIXTURE = [
    {"event": "cycle_start", "timestamp": "2026-08-01T00:00:00", "cycle": 1,
     "step": 0, "games": 4, "games_generated": 4},
    {"event": "train_step", "timestamp": "2026-08-01T00:00:01", "step": 1,
     "loss": 7.99, "lr": 0.2, "games": 4, "elo": 0.0, "cycle": 1},
    {"event": "train_step", "timestamp": "2026-08-01T00:00:02", "step": 2,
     "loss": 6.5, "lr": 0.2, "games": 4, "elo": 0.0, "cycle": 1},
    {"event": "train_step", "timestamp": "2026-08-01T00:00:03", "step": 3,
     "loss": 5.25, "lr": 0.2, "games": 4, "elo": 0.0, "cycle": 1},
    {"event": "eval_gate", "timestamp": "2026-08-01T00:00:04", "step": 3,
     "cycle": 1, "winrate": 0.6, "replaced": True, "elo": 25.0},
    {"event": "train_step", "timestamp": "2026-08-01T00:00:05", "step": 4,
     "loss": 4.1, "lr": 0.1, "games": 8, "elo": 25.0, "cycle": 1},
]


def make_game_npz(path: Path, size: int = 9, n_moves: int = 5,
                  komi: float = 7.5, winner: str = "B") -> None:
    """Write a tiny synthetic self-play game with a plausible s-plane layout.

    Moves: black then white alternating, each on an empty point (no captures),
    with full 8-position history zero-filled where the game is short.
    """
    moves = [(size - 1, 0), (0, 0), (size - 1, 1), (0, 1), (size - 1, 2)]
    moves = moves[:n_moves]
    boards = []  # absolute 0/1/2 grids, one per recorded position
    grid = np.zeros((size, size), dtype=int)
    boards.append(grid.copy())
    colors = []
    for i, (r, c) in enumerate(moves):
        color = 1 if i % 2 == 0 else 2
        colors.append(color)
        grid[r, c] = color
        boards.append(grid.copy())

    # encode: plane 2t = mover stones, 2t+1 = opponent stones, plane 16 = mover
    s_list = []
    pi_list = []
    for t in range(n_moves):
        mover = 1 if t % 2 == 0 else 2
        opp = 2 if mover == 1 else 1
        cur = boards[t]
        planes = np.zeros((17, size, size), dtype=np.float32)
        planes[0] = (cur == mover)
        planes[1] = (cur == opp)
        planes[16] = 1.0 if mover == 1 else 0.0
        s_list.append(planes)
        pi = np.zeros(size * size + 1, dtype=np.float32)
        r, c = moves[t]
        pi[r * size + c] = 1.0
        pi_list.append(pi)

    s = np.stack(s_list)
    pi = np.stack(pi_list)
    z = np.array([1.0 if colors[i] == (1 if winner == "B" else 2) else -1.0
                  for i in range(n_moves)], dtype=np.float32)
    np.savez_compressed(
        path,
        s=s, pi=pi, z=z,
        board_size=np.int64(size),
        komi=np.float32(komi),
        winner=winner,
        result=f"{winner}+5.5",
        move_count=np.int64(n_moves),
        simulations=np.int64(40),
        temperature_threshold=np.int64(10),
        forced_terminal=np.bool_(False),
        seed=np.int64(1),
    )


@pytest.fixture
def fixtures(tmp_path: Path):
    log = tmp_path / "train.jsonl"
    write_train_log(log, TRAIN_FIXTURE)
    sp = tmp_path / "selfplay"
    sp.mkdir()
    make_game_npz(sp / "game_0000000001.npz", n_moves=5)
    make_game_npz(sp / "game_0000000002.npz", n_moves=3)
    return {"train_log": log, "selfplay_dir": sp}


@pytest.fixture
def client(fixtures):
    app = create_app(train_log=fixtures["train_log"],
                     selfplay_dir=fixtures["selfplay_dir"])
    app.config["TESTING"] = True
    return app.test_client()


# ---------------------------------------------------------------------------
# /api/train
# ---------------------------------------------------------------------------

class TestTrainAPI:
    def test_parses_train_steps_and_evals(self, fixtures):
        data = parse_train_log(fixtures["train_log"])
        assert data["events"] == 6
        assert [p["step"] for p in data["steps"]] == [1, 2, 3, 4]
        assert [p["loss"] for p in data["steps"]] == [7.99, 6.5, 5.25, 4.1]
        assert data["evals"][0] == {
            "step": 3, "cycle": 1, "winrate": 0.6, "replaced": True,
            "elo": 25.0, "timestamp": "2026-08-01T00:00:04",
        }
        assert data["cycles"][0]["cycle"] == 1

    def test_latest_merges_step_and_eval(self, client):
        r = client.get("/api/train")
        assert r.status_code == 200
        latest = r.get_json()["latest"]
        assert latest["step"] == 4
        assert latest["loss"] == 4.1
        assert latest["elo"] == 25.0  # from last train_step
        assert latest["cycle"] == 1

    def test_alive_flag_and_timestamp(self, client):
        data = client.get("/api/train").get_json()
        assert "alive" in data
        assert "now" in data
        assert data["file_mtime"] is not None

    def test_missing_log_yields_empty(self, tmp_path):
        data = parse_train_log(tmp_path / "nope.jsonl")
        assert data["steps"] == []
        assert data["evals"] == []
        assert data["latest"] is None

    def test_corrupt_lines_are_skipped(self, tmp_path):
        log = tmp_path / "train.jsonl"
        log.write_text("not json\n{\"event\": \"train_step\", \"step\": 1, \"loss\": 1.0}\n",
                       encoding="utf-8")
        data = parse_train_log(log)
        assert data["events"] == 1
        assert len(data["steps"]) == 1


# ---------------------------------------------------------------------------
# /api/games
# ---------------------------------------------------------------------------

class TestGamesAPI:
    def test_lists_games_newest_first(self, client, fixtures):
        r = client.get("/api/games")
        assert r.status_code == 200
        games = r.get_json()["games"]
        assert len(games) == 2
        assert games[0]["id"] == "game_0000000002"
        assert games[1]["id"] == "game_0000000001"
        g = games[1]
        assert g["move_count"] == 5
        assert g["winner"] == "B"
        assert g["board_size"] == 9
        assert g["komi"] == 7.5
        assert g["result"] == "B+5.5"
        assert g["size"] > 0
        assert "mtime" in g

    def test_empty_dir_lists_nothing(self, tmp_path):
        app = create_app(selfplay_dir=tmp_path / "empty")
        r = app.test_client().get("/api/games")
        assert r.status_code == 200
        assert r.get_json()["games"] == []


# ---------------------------------------------------------------------------
# /api/games/<id> replay
# ---------------------------------------------------------------------------

class TestGameReplay:
    def test_reconstructs_exact_moves(self, fixtures):
        path = fixtures["selfplay_dir"] / "game_0000000001.npz"
        data = reconstruct_game(path)
        assert data["move_count"] == 5
        assert data["winner"] == "B"
        assert data["result"] == "B+5.5"

        # 5 recorded positions -> 6 positions incl. initial, 5 derived moves
        # (4 from diffs + 1 final pass for a two-pass natural end)
        assert len(data["positions"]) == 6
        assert len(data["moves"]) == 5
        assert data["positions"][0] == [[0] * 9 for _ in range(9)]

        # every *point* move adds exactly one stone on an empty point
        positions = data["positions"]
        point_idx = [i for i, m in enumerate(data["moves"]) if not m.get("pass")]
        for i in point_idx:
            prev, cur = positions[i], positions[i + 1]
            added = [(r, c) for r in range(9) for c in range(9)
                     if prev[r][c] == 0 and cur[r][c] != 0]
            assert added == [(data["moves"][i]["r"], data["moves"][i]["c"])]

        # colours alternate B/W and match the stored s planes
        expected_colors = ["B", "W", "B", "W", "B"]
        actual_colors = [m["color"] for m in data["moves"]]
        assert actual_colors == expected_colors

        # derived point moves match the stored pi argmax / s-plane diffs
        point_moves = [m for m in data["moves"] if not m.get("pass")]
        assert len(point_moves) == 4  # 4 diffs; the 5th move is the final pass
        assert point_moves[0] == {"color": "B", "r": 8, "c": 0, "captured": 0}
        assert point_moves[1] == {"color": "W", "r": 0, "c": 0, "captured": 0}
        assert point_moves[2] == {"color": "B", "r": 8, "c": 1, "captured": 0}

        # final move is a pass (natural two-pass end), board unchanged
        assert data["moves"][-1].get("pass") is True
        assert data["positions"][-1] == data["positions"][-2]

    def test_replay_api(self, client, fixtures):
        r = client.get("/api/games/game_0000000001")
        assert r.status_code == 200
        data = r.get_json()
        assert data["id"] == "game_0000000001"
        assert data["board_size"] == 9
        assert len(data["moves"]) == data["move_count"]
        assert len(data["positions"]) == data["move_count"] + 1

    def test_missing_game_404(self, client):
        r = client.get("/api/games/nope")
        assert r.status_code == 404

    def test_path_traversal_blocked(self, client):
        r = client.get("/api/games/..%2F..%2Fetc")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# / (index.html)
# ---------------------------------------------------------------------------

class TestIndex:
    def test_serves_dashboard_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"replay-board" in r.data
        assert b"<html" in r.data.lower()

    def test_static_assets(self, client):
        for url, needle in (("/static/app.js", b"drawGoban"),
                            ("/static/style.css", b"--wood")):
            r = client.get(url)
            assert r.status_code == 200
            assert needle in r.data


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_missing_all_files_ok(self, tmp_path):
        app = create_app(train_log=tmp_path / "no.jsonl",
                         selfplay_dir=tmp_path / "no-selfplay")
        c = app.test_client()
        assert c.get("/api/train").status_code == 200
        assert c.get("/api/games").status_code == 200
        train = c.get("/api/train").get_json()
        assert train["steps"] == []
        assert train["latest"] is None
        assert train["alive"] is False
