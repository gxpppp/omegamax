"""Lightweight sanity tests for the todo-12 strength-ladder harness.

The full ladder (40/200/800 sims, 60 games per pairing, 19x19) lives in the
CLI harness ``omigamax/cli/mcts_strength.py`` and is far too slow for
pytest. These tests exercise the harness internals on a tiny 9x9 board with
a mini network and a couple of simulations: that a full game is legal
(``Board.play`` raises otherwise), the per-game record is complete, the
higher-sim side win rate is computed correctly, and SGF round-trips.
"""

import numpy as np
import torch

from omigamax.cli.mcts_strength import (
    _RandomAgent,
    _load_weights,
    _mcts_agent,
    _summarize_pairing,
    play_game,
    run_pairing,
)
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board
from omigamax.rules.sgf import parse_sgf

SIZE = 9
KOMI = 7.5


def _tiny_network():
    torch.manual_seed(0)
    net = create_model(blocks=2, channels=16, board_size=SIZE).eval()
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def test_play_game_legal_and_recorded():
    rng = np.random.default_rng(7)
    net = _tiny_network()
    black = _mcts_agent(net, 2, 1.0, KOMI, 3, SIZE, rng)
    white = _RandomAgent(SIZE, np.random.default_rng(8))
    rec = play_game(black, white, SIZE, KOMI, seed=99, max_moves=200)

    assert rec["winner"] in ("B", "W")  # komi 7.5 => no jigo
    assert rec["moves"] == len(rec["move_list"]) > 0
    assert rec["result"].startswith(rec["winner"] + "+")
    assert rec["seed"] == 99
    assert rec["wall_time_s"] >= 0.0
    # every recorded move must be legal when replayed (Board.play raises)
    board = Board(SIZE)
    for move, color in rec["move_list"]:
        board.play(move, color)
    assert board.is_terminal() or rec["forced_terminal"]


def test_pairing_summary_counts_hi_wins():
    records = [
        {"black_sims": 2, "white_sims": 4, "winner": "B"},  # hi=4 as white -> no
        {"black_sims": 4, "white_sims": 2, "winner": "B"},  # hi=4 as black -> yes
        {"black_sims": 2, "white_sims": 4, "winner": "W"},  # hi=4 as white -> yes
    ]
    summary = _summarize_pairing(records, hi_sims=4)
    assert summary["games"] == 3
    assert summary["hi_wins"] == 2
    assert summary["hi_winrate"] == 2 / 3


def test_run_pairing_alternates_colours_and_writes_sgf(tmp_path):
    net = _tiny_network()
    records, summary = run_pairing(
        2, 4, games=4, size=SIZE, komi=KOMI, network=net, tau=1.0,
        virtual_loss=3, base_seed=1000, max_moves=120, sgf_dir=tmp_path,
    )
    assert len(records) == 4
    assert summary["games"] == 4
    # colours alternate: even games lo=black, odd games hi=black
    assert records[0]["black_sims"] == 2
    assert records[1]["black_sims"] == 4
    for rec in records:
        assert rec["winner"] in ("B", "W")
        assert "sgf" in rec
    # every SGF parses and has the recorded move count
    sgf_files = sorted(tmp_path.glob("*.sgf"))
    assert len(sgf_files) == 4
    for i, sgf_path in enumerate(sgf_files):
        parsed = parse_sgf(sgf_path.read_text(encoding="utf-8"))
        assert parsed["size"] == SIZE
        assert len(parsed["moves"]) == records[i]["moves"]


def test_load_weights_accepts_full_checkpoint_and_raw_state_dict(tmp_path):
    """Regression (todo 21): --weights must load a trained checkpoint -- the
    full checkpoint format wraps the state dict under ``model_state_dict`` --
    and still accept the raw state dicts this harness itself emits."""
    import torch

    from omigamax.cli import mcts_strength
    from omigamax.network.model import create_model

    net = create_model(blocks=2, channels=16, board_size=SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # full training checkpoint (the models/best.pt shape)
    ckpt_path = tmp_path / "best.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "global_step": 1234,
        "arch": {"blocks": 2, "channels": 16, "board_size": SIZE},
        "config": {"board_size": SIZE},
    }, ckpt_path)
    loaded = create_model(blocks=2, channels=16, board_size=SIZE).to(device)
    src = _load_weights(loaded, ckpt_path, device)
    assert "loaded from" in src
    # weights actually matched the checkpoint
    ref = net.state_dict()
    for k, v in loaded.state_dict().items():
        assert torch.equal(v.cpu(), ref[k].cpu()), k

    # raw state dict (this harness's _smoke_train output) still works
    raw_path = tmp_path / "raw.pt"
    torch.save(net.state_dict(), raw_path)
    loaded2 = create_model(blocks=2, channels=16, board_size=SIZE).to(device)
    _load_weights(loaded2, raw_path, device)
    for k, v in loaded2.state_dict().items():
        assert torch.equal(v.cpu(), ref[k].cpu()), k
