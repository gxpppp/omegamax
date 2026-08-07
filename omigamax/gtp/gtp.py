"""Standard GTP (Go Text Protocol) engine (todo 18).

Implements the full GTP version-2 command set (GNU Go's spec -- plan
References: https://www.gnu.org/software/gnugo/gnugo_19.html) so omigamax can
be driven by external GUIs, match controllers and (later) online-platform
bots::

    protocol_version / name / version / known_command / list_commands
    boardsize / clear_board / komi / play / genmove
    fixed_handicap / place_free_handicap / set_free_handicap
    loadsgf / kgs-time_settings / time_left / final_score / printsgf
    undo / kgs-chat (silent) / quit

Response format (GTP v2): a response frame is ``=id <text>`` on success or
``?id <text>`` on error (``id`` echoed only when the client supplied one;
``=``/``?`` is followed by a space -- ``= `` for an empty success), terminated
by a blank line. Multi-line responses (``list_commands``) continue on lines
without the status prefix. Empty input lines are ignored.

Coordinates are GTP letters A-T skipping I (columns) with row numbers 1..N
(row 1 = the bottom edge, opposite to the rules engine's 0-based ``(row,
col)`` where row 0 is the top). ``pass`` is a legal move.

Move generation reuses the existing search stack
(``omigamax.mcts.run_search`` + the todo-11 :class:`BatchedNetworkEvaluator`)
with the todo-15 *evaluation discipline*: no Dirichlet root noise and
``tau = 0`` (argmax) -- GTP games are evaluation games, not self-play, so
play is deterministic (per-move; a seeded RNG resolves ties). ``kgs-time_settings``
is parsed and mapped to a search budget by a simplified stub (plan: 简化预算
映射 stub); the full byo-yomi clock is a deferred extension. ``time_left`` is
accepted and stored but does not change the budget (stub). ``kgs-chat`` is
handled by responding empty (plan Must-NOT: no chat semantics) and appended to
:attr:`GTPEngine.chat_log` so a future platform layer can inspect it.
``undo [n]`` replays the game to before the last ``n`` moves (handicap-aware:
re-undoing past the handicap stones clears it).

Robustness layer (todo 19): every command handler is wrapped by
:meth:`GTPEngine.handle_line`, which never raises -- malformed input (garbage,
wrong arity, bad coords, absurd ids, binary junk, over-long lines) produces a
well-formed ``?`` frame and the engine stays alive for the next line. The
echoed command id and response lines are capped so pathological input cannot
produce unbounded frames.

Handicap (``fixed_handicap`` etc.): the standard star-point placement, black
stones; after handicap it is *white* to move (the engine tracks the handicap
count so the mover stays correct). The MCTS tree's side-to-move is derived
from move-count parity, which disagrees with the true mover when the handicap
has an EVEN number of stones (the board then holds an even count of BLACK
stones, so parity says black to play). :meth:`GTPEngine._genmove` therefore
threads the requested colour into the search root (:func:`make_root`'s
``color`` parameter), so the legal-move mask, the value perspective and the
colour plane fed to the network all reflect the true mover; the per-move
legality validation in :meth:`GTPEngine._legalize_action` is kept as a
defensive check.

SGF export: ``printsgf <file>`` writes the current game (plan: SGF 导出 对局后).

Model loading: ``GTPEngine(model_path=...)`` accepts either a todo-14
checkpoint (``models/best.pt`` -- ``arch`` + ``model_state_dict``) or a plain
``state_dict`` (architecture inferred from tensor shapes). Without a model the
engine still answers every protocol command and ``genmove`` falls back to a
uniform-random legal move. The requested board size is authoritative:
``GTPEngine(board_size=...)``, ``boardsize`` and ``loadsgf`` to a size that
does not match the loaded checkpoint rebuild a random-init network for that
size (strength degraded, warning logged) instead of forcing the checkpoint's
own size.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch

from omigamax.config import load_config
from omigamax.mcts import BatchedNetworkEvaluator, make_root, run_search, sample_action
from omigamax.network.features import index_to_point, pass_index
from omigamax.network.model import create_model
from omigamax.rules import BLACK, WHITE, Board
from omigamax.rules.sgf import export_sgf, parse_sgf

# GTP column letters: A..T skipping I (19 columns, uppercase).
GTP_COLUMNS = "ABCDEFGHJKLMNOPQRST"

ENGINE_NAME = "omigamax"
VERSION = "0.1.0"

logger = logging.getLogger(__name__)

# Acceptable board sizes for the GTP engine (rules engine is parameterized).
MIN_SIZE = 2
MAX_SIZE = 19

# --- kgs-time_settings -> search-budget mapping stub -----------------------
# The plan (todo 18) locks this as a *simplified budget-mapping stub*; the full
# byo-yomi clock is a documented deferred extension for platform integration.
EXPECTED_MOVES = 250      # typical GTP game length used to split main time
SIMS_PER_SECOND = 100     # conservative estimate (RTX 3060: ~90-160 sims/s)
MIN_SIMS = 8
MAX_SIMS = 800

# Standard handicap star-point offsets for 19x19 / 13x13 / 9x9 (GNU Go's
# convention: 2 stones = bottom-left + top-right stars, then top-left,
# bottom-right, centre, edge middles).
_HANDICAP_STARS = {19: (3, 9, 15), 13: (3, 6, 9), 9: (2, 4, 6)}


# Max length of the echoed command id and of each response line. Pathological
# input (a 10k-digit id or a 10k-char token) must still yield a bounded frame.
_MAX_ID_LEN = 64
_MAX_LINE_LEN = 2000


def _clip(text: str, limit: int = _MAX_LINE_LEN) -> str:
    """Truncate ``text`` to ``limit`` chars with an ellipsis marker."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


class GTPCommandError(Exception):
    """Raised to produce a ``?id <message>`` GTP error response."""


def parse_color(token: str) -> int:
    """Parse a GTP colour token (``B``/``W``, case-insensitive) to BLACK/WHITE."""
    c = token.strip().upper()
    if c == "B":
        return BLACK
    if c == "W":
        return WHITE
    raise GTPCommandError(f"invalid color: {token!r}")


def parse_vertex(token: str, size: int) -> "tuple[int, int] | None":
    """Parse a GTP vertex to a ``(row, col)`` board move; ``None`` for a pass.

    GTP coordinates: column letter A-T (skipping I), row number 1..size with
    row 1 at the bottom edge. ``pass``/``PASS``/``0``/``""`` mean pass.
    """
    token = token.strip()
    if token.lower() == "pass" or token == "0" or token == "":
        return None
    if len(token) < 2:
        raise GTPCommandError(f"invalid coordinate: {token!r}")
    col_char, num = token[0].upper(), token[1:]
    if col_char not in GTP_COLUMNS:
        raise GTPCommandError(f"invalid coordinate: {token!r}")
    if not num.isdigit():
        raise GTPCommandError(f"invalid coordinate: {token!r}")
    col = GTP_COLUMNS.index(col_char)
    row_num = int(num)
    if not (1 <= row_num <= size):
        raise GTPCommandError(f"coordinate out of bounds: {token!r}")
    return (size - row_num, col)


def to_gtp(move: "tuple[int, int] | None", size: int) -> str:
    """Encode a board move (or pass) as a GTP coordinate string."""
    if move is None:
        return "pass"
    r, c = move
    return GTP_COLUMNS[c] + str(size - r)


def _handicap_points(size: int, n: int) -> "list[tuple[int, int]]":
    """Standard star-point placement for ``n`` handicap stones (board coords)."""
    if size not in _HANDICAP_STARS:
        raise GTPCommandError(f"fixed handicap requires board size 9, 13 or 19")
    a, c, b = _HANDICAP_STARS[size]
    pattern = [
        (b, a),  # bottom-left star
        (a, b),  # top-right star
        (b, b),  # bottom-right star
        (a, a),  # top-left star
        (c, c),  # centre
        (a, c),  # top-middle
        (b, c),  # bottom-middle
        (c, a),  # left-middle
        (c, b),  # right-middle
    ]
    n = int(n)
    if not (2 <= n <= len(pattern)):
        raise GTPCommandError(f"invalid handicap: {n}")
    return pattern[:n]


class GTPEngine:
    """A GTP engine over the omigamax rules + MCTS search stack."""

    _COMMANDS = [
        "protocol_version",
        "name",
        "version",
        "known_command",
        "list_commands",
        "quit",
        "boardsize",
        "clear_board",
        "komi",
        "play",
        "genmove",
        "fixed_handicap",
        "place_free_handicap",
        "set_free_handicap",
        "loadsgf",
        "kgs-time_settings",
        "time_left",
        "final_score",
        "printsgf",
        "undo",
    ]
    # command -> bound handler method name ("kgs-chat" handled but not listed).
    _HANDLERS = {
        "protocol_version": "_cmd_protocol_version",
        "name": "_cmd_name",
        "version": "_cmd_version",
        "known_command": "_cmd_known_command",
        "list_commands": "_cmd_list_commands",
        "quit": "_cmd_quit",
        "boardsize": "_cmd_boardsize",
        "clear_board": "_cmd_clear_board",
        "komi": "_cmd_komi",
        "play": "_cmd_play",
        "genmove": "_cmd_genmove",
        "fixed_handicap": "_cmd_fixed_handicap",
        "place_free_handicap": "_cmd_place_free_handicap",
        "set_free_handicap": "_cmd_set_free_handicap",
        "loadsgf": "_cmd_loadsgf",
        "kgs-time_settings": "_cmd_kgs_time_settings",
        "time_left": "_cmd_time_left",
        "final_score": "_cmd_final_score",
        "printsgf": "_cmd_printsgf",
        "undo": "_cmd_undo",
        "kgs-chat": "_cmd_kgs_chat",
    }

    def __init__(
        self,
        model_path: "str | Path | None" = None,
        network: "torch.nn.Module | None" = None,
        *,
        board_size: "int | None" = None,
        komi: "float | None" = None,
        simulations: "int | None" = None,
        device: "str | torch.device | None" = None,
        config_path: "str | Path | None" = None,
        seed: int = 0,
    ) -> None:
        self._config_path = config_path
        cfg = load_config(config_path)
        self.size = int(
            board_size if board_size is not None else cfg.get("board_size", 19)
        )
        self.komi = float(komi if komi is not None else cfg.get("komi", 7.5))
        self.simulations = int(
            simulations if simulations is not None else cfg.get("simulations", 200)
        )
        self.c_puct = float(cfg.get("c_puct", 2.5))
        self.virtual_loss = int(cfg.get("virtual_loss", 3))
        self.leaf_batch = int(cfg.get("leaf_batch", 16))
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self._network: "torch.nn.Module | None" = None
        self._evaluator = None
        self._handicap = 0
        self._last_root = None  # the most recent search root (testable probe)
        self._time_settings: "dict | None" = None
        self._time_left: dict = {}
        self.chat_log: "list[tuple[str, str]]" = []
        self.should_quit = False

        if network is not None:
            self.size = int(network.board_size)
            self._set_network(network)
        elif model_path is not None:
            self._load_model(Path(model_path))
        self.board = Board(self.size)

    # -- network plumbing ----------------------------------------------------

    def _set_network(self, network: "torch.nn.Module") -> None:
        network.eval()
        network.to(self.device)
        self._network = network
        self._evaluator = BatchedNetworkEvaluator(network, batch_size=self.leaf_batch)

    def _build_default_network(self) -> None:
        """Random-init network of the config architecture for the current size."""
        cfg = load_config(self._config_path)
        net = create_model(
            int(cfg.get("blocks", 10)), int(cfg.get("channels", 128)), self.size
        )
        self._set_network(net)

    def _load_model(self, path: Path) -> int:
        """Load a checkpoint or raw state_dict; returns its board size.

        The requested board size (:attr:`size` -- from the constructor's
        ``board_size`` or a ``boardsize``/``loadsgf`` command) is
        authoritative. A checkpoint whose native board size differs from it
        is **not** force-loaded at the wrong size (that would yield
        out-of-bounds coordinates, e.g. a 19x19 ``T14`` against a 9x9
        board). Instead a random-init network of the requested size is built
        with the checkpoint's blocks/channels and a warning is logged --
        the same rebuild-for-a-new-size behaviour as
        :meth:`_ensure_network_for_size`, so the session stays consistent
        (strength degraded). A same-size checkpoint loads normally.
        """
        if not path.exists():
            raise FileNotFoundError(f"model file not found: {path}")
        state = torch.load(path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "arch" in state and "model_state_dict" in state:
            a = state["arch"]
            native = int(a["board_size"])
            blocks, channels = int(a["blocks"]), int(a["channels"])
            if native == self.size:
                net = create_model(blocks, channels, native).to(self.device)
                net.load_state_dict(state["model_state_dict"])
            else:
                logger.warning(
                    "checkpoint trained on %dx%d; building random-init %dx%d "
                    "network -- strength degraded",
                    native, native, self.size, self.size,
                )
                net = create_model(blocks, channels, self.size).to(self.device)
        else:
            native, net = _build_from_state_dict(state, self.device)
            if native != self.size:
                logger.warning(
                    "state_dict trained on %dx%d; building random-init %dx%d "
                    "network -- strength degraded",
                    native, native, self.size, self.size,
                )
                net = create_model(net.blocks, net.channels, self.size).to(self.device)
        self._set_network(net)
        return native

    def _ensure_network_for_size(self) -> None:
        if self._network is not None and self._network.board_size != self.size:
            self._build_default_network()

    def _set_board_size(self, size: int) -> None:
        self.size = size
        self.board = Board(size)
        self._handicap = 0
        self._time_left.clear()
        self._ensure_network_for_size()

    # -- game-state helpers --------------------------------------------------

    def _color_to_move(self) -> int:
        """Side to move: black opens; after handicap stones white moves first."""
        count = len(self.board.moves)
        if self._handicap > 0:
            return WHITE if (count - self._handicap) % 2 == 0 else BLACK
        return BLACK if count % 2 == 0 else WHITE

    # -- move generation -----------------------------------------------------

    def _genmove_random(self, color: int) -> "tuple[int, int] | None":
        points = [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.board.is_legal((r, c), color)
        ]
        idx = int(self.rng.integers(0, len(points) + 1))  # last slot = pass
        return None if idx == len(points) else points[idx]

    def _legalize_action(self, action: int, color: int, root) -> "tuple[int, int] | None":
        """Return the move for ``color`` that ``action`` maps to.

        The search root is built with the *requested* colour as its side to
        move (:meth:`_genmove` threads ``color`` into :func:`make_root`), so
        the sampled action is already from ``color``'s legal mask. This
        validation remains as a defensive check: if the action is ever not
        legal for ``color`` (protocol misuse, a stale root) it falls back to
        the most-visited child that is legal.
        """
        size = self.size

        def legal_for(a: int) -> bool:
            if a == pass_index(size):
                return True
            rr, cc = index_to_point(a, size)
            return self.board.is_legal((rr, cc), color)

        if legal_for(action):
            if action == pass_index(size):
                return None
            r, c = index_to_point(action, size)
            return (r, c)
        candidates = [a for a in root.children if legal_for(a)]
        if not candidates:
            return None  # pass is always legal -- defensive
        action = max(candidates, key=lambda a: (root.children[a].visit_count, -a))
        if action == pass_index(size):
            return None
        r, c = index_to_point(action, size)
        return (r, c)

    def _genmove(self, color: int) -> "tuple[int, int] | None":
        if self.board.is_terminal():
            move = None  # two passes already ended the game
        elif self._network is None:
            move = self._genmove_random(color)
        else:
            # Thread the requested colour as the search root's side to move:
            # after an EVEN fixed handicap (2/4/6/8) the board holds an even
            # number of BLACK stones, so move-count parity would (wrongly)
            # make the tree play as BLACK -- wrong legal mask, wrong value
            # perspective and a wrong colour plane 16. The explicit colour
            # keeps the search on the true mover (odd handicaps coincide with
            # parity and are unaffected).
            root = make_root(self.board, color=color)
            run_search(
                root, None, self.simulations,
                evaluator=self._evaluator, komi=self.komi, c_puct=self.c_puct,
                virtual_loss=self.virtual_loss, batch_size=self.leaf_batch,
            )  # dirichlet_alpha stays None -> no root noise (evaluation discipline)
            self._last_root = root
            action = sample_action(root, 0.0, rng=self.rng)  # tau=0 -> argmax
            move = self._legalize_action(action, color, root)
        self.board.play(move, color)
        return move

    # -- parsing / dispatch --------------------------------------------------

    def _parse(self, line: str):
        """Split ``[id] command [args...]``; command lowercased (case-tolerant)."""
        tokens = line.split()
        if tokens[0].isdigit():
            cmd_id, tokens = tokens[0], tokens[1:]
        else:
            cmd_id = None
        if not tokens:
            raise GTPCommandError("missing command")
        return cmd_id, tokens[0].lower(), tokens[1:]

    def _dispatch(self, command: str, args: list):
        handler_name = self._HANDLERS.get(command)
        if handler_name is None:
            raise GTPCommandError(f"unknown command: {command}")
        return getattr(self, handler_name)(args)

    def _format_response(self, ok: bool, cmd_id, text: str) -> list:
        status = "=" if ok else "?"
        head = f"{status}{cmd_id} {text}" if cmd_id is not None else f"{status} {text}"
        if cmd_id is not None and len(cmd_id) > _MAX_ID_LEN:
            head = f"{status}{cmd_id[:_MAX_ID_LEN]} {text}"
        lines = [_clip(ln) for ln in head.split("\n")]
        lines.append("")  # blank-line frame terminator (GTP v2)
        return lines

    def handle_line(self, line: str):
        """Process one GTP input line; return the response frame lines.

        The returned list ends with the blank terminator line (``""``) and is
        what :mod:`omigamax.cli.gtp_main` writes verbatim. Whitespace-only
        input is ignored (returns ``None``). Never raises: malformed input
        produces a ``?`` error frame and internal failures are caught and
        reported (traceback to stderr) so the engine always stays alive.
        """
        line = line.rstrip("\r\n")
        if not line.strip():
            return None
        cmd_id = None
        try:
            cmd_id, command, args = self._parse(line)
            ok, text = self._dispatch(command, args)
        except GTPCommandError as exc:
            ok, text = False, str(exc)
        except Exception as exc:  # noqa: BLE001 - never crash a GTP peer
            import traceback

            traceback.print_exc(file=sys.stderr)
            ok, text = False, f"internal error: {exc}"
        return self._format_response(ok, cmd_id, text)

    # -- command handlers ----------------------------------------------------

    def _cmd_protocol_version(self, args):
        return True, "2"

    def _cmd_name(self, args):
        return True, ENGINE_NAME

    def _cmd_version(self, args):
        return True, VERSION

    def _cmd_known_command(self, args):
        if len(args) != 1:
            raise GTPCommandError("known_command requires one argument")
        return True, "true" if args[0].lower() in self._HANDLERS else "false"

    def _cmd_list_commands(self, args):
        return True, "\n".join(self._COMMANDS)

    def _cmd_quit(self, args):
        self.should_quit = True
        return True, ""

    def _cmd_boardsize(self, args):
        if len(args) != 1:
            raise GTPCommandError("boardsize requires one argument")
        try:
            size = int(args[0])
        except ValueError:
            raise GTPCommandError(f"invalid board size: {args[0]!r}")
        if not (MIN_SIZE <= size <= MAX_SIZE):
            raise GTPCommandError(f"unacceptable board size: {size}")
        self._set_board_size(size)
        return True, ""

    def _cmd_clear_board(self, args):
        self.board = Board(self.size)
        self._handicap = 0
        self._time_left.clear()
        return True, ""

    def _cmd_komi(self, args):
        if len(args) != 1:
            raise GTPCommandError("komi requires one argument")
        try:
            komi = float(args[0])
        except ValueError:
            raise GTPCommandError(f"invalid komi: {args[0]!r}")
        if not math.isfinite(komi):
            raise GTPCommandError(f"invalid komi: {args[0]!r}")
        self.komi = komi
        return True, ""

    def _cmd_play(self, args):
        if len(args) != 2:
            raise GTPCommandError("play requires a color and a vertex")
        color = parse_color(args[0])
        move = parse_vertex(args[1], self.size)
        if not self.board.is_legal(move, color):
            raise GTPCommandError(f"illegal move: {args[0]} {args[1]}")
        self.board.play(move, color)
        return True, ""

    def _cmd_genmove(self, args):
        if len(args) != 1:
            raise GTPCommandError("genmove requires a color")
        color = parse_color(args[0])
        move = self._genmove(color)
        return True, to_gtp(move, self.size)

    def _place_handicap(self, n) -> "list[tuple[int, int]]":
        points = _handicap_points(self.size, n)
        self.board = Board(self.size)
        self._handicap = 0
        for point in points:
            self.board.play(point, BLACK)
        self._handicap = n
        return points

    def _cmd_fixed_handicap(self, args):
        if len(args) != 1:
            raise GTPCommandError("fixed_handicap requires one argument")
        try:
            n = int(args[0])
        except ValueError:
            raise GTPCommandError(f"invalid handicap: {args[0]!r}")
        points = self._place_handicap(n)
        return True, ",".join(to_gtp(p, self.size) for p in points)

    def _cmd_place_free_handicap(self, args):
        # Free placement uses the standard star points (documented stub) and
        # reports where black placed them -- the GTP contract.
        if len(args) != 1:
            raise GTPCommandError("place_free_handicap requires one argument")
        try:
            n = int(args[0])
        except ValueError:
            raise GTPCommandError(f"invalid handicap: {args[0]!r}")
        points = self._place_handicap(n)
        return True, ",".join(to_gtp(p, self.size) for p in points)

    def _cmd_set_free_handicap(self, args):
        if len(args) < 1:
            raise GTPCommandError("set_free_handicap requires n and coordinates")
        try:
            n = int(args[0])
        except ValueError:
            raise GTPCommandError(f"invalid handicap: {args[0]!r}")
        coords = args[1:]
        if len(coords) != n:
            raise GTPCommandError(f"expected {n} coordinates, got {len(coords)}")
        if not (2 <= n <= 9):
            raise GTPCommandError(f"invalid handicap: {n}")
        if self.size not in _HANDICAP_STARS:
            raise GTPCommandError("set_free_handicap requires board size 9, 13 or 19")
        self.board = Board(self.size)
        self._handicap = 0
        for coord in coords:
            move = parse_vertex(coord, self.size)
            if not self.board.is_legal(move, BLACK):
                raise GTPCommandError(f"illegal handicap stone: {coord}")
            self.board.play(move, BLACK)
        self._handicap = n
        return True, ""

    def _cmd_loadsgf(self, args):
        if len(args) != 1:
            raise GTPCommandError("loadsgf requires a filename")
        path = Path(args[0])
        if not path.exists():
            raise GTPCommandError(f"file not found: {path}")
        try:
            parsed = parse_sgf(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError, OSError) as exc:
            raise GTPCommandError(f"invalid SGF file: {exc}")
        if not (MIN_SIZE <= parsed["size"] <= MAX_SIZE):
            raise GTPCommandError(f"unacceptable board size in SGF: {parsed['size']}")
        self.size = parsed["size"]
        self.komi = parsed["komi"]
        self.board = Board(self.size)
        self._handicap = 0
        self._time_left.clear()
        for color, move in parsed["moves"]:
            if not self.board.is_legal(move, color):
                raise GTPCommandError(
                    f"illegal move in SGF: {to_gtp(move, self.size)}"
                )
            self.board.play(move, color)
        self._ensure_network_for_size()
        return True, ""

    def _budget_from_time(self, main_s, byo_s, byo_stones):
        """Simplified kgs-time_settings -> search-budget mapping stub.

        Main time is split over an expected game length; when only byo-yomi
        time is given its period is split per stone. Returns ``None`` (keep
        the current budget) when no positive time is available.
        """
        main_s = max(0, int(main_s))
        byo_s = max(0, int(byo_s))
        byo_stones = max(0, int(byo_stones))
        if main_s > 0:
            per_move = main_s / EXPECTED_MOVES
        elif byo_s > 0:
            per_move = byo_s / max(byo_stones, 1)
        else:
            return None
        return min(MAX_SIMS, max(MIN_SIMS, int(per_move * SIMS_PER_SECOND)))

    def _cmd_kgs_time_settings(self, args):
        # Standard KGS / GNU Go form: ``kgs-time_settings <main> <byo> <stones>``.
        # KataGo's variant prefixes a clock type (``none``/``absolute``/
        # ``byoyomi``/``canadian``) -- tolerate it for cross-engine interop.
        if args and args[0].lower() in ("none", "absolute", "byoyomi", "canadian"):
            args = args[1:]
        if len(args) != 3:
            raise GTPCommandError("kgs-time_settings requires 3 arguments")
        try:
            main_s, byo_s, byo_stones = (int(v) for v in args)
        except ValueError:
            raise GTPCommandError(f"invalid time settings: {' '.join(args)}")
        budget = self._budget_from_time(main_s, byo_s, byo_stones)
        if budget is not None:
            self.simulations = budget
        self._time_settings = {
            "main_time_s": int(main_s),
            "byo_time_s": int(byo_s),
            "byo_stones": int(byo_stones),
        }
        return True, ""

    def _cmd_time_left(self, args):
        # Stub: accepted + stored; the full byo-yomi clock is a deferred
        # extension (plan) and does not change the search budget here.
        if len(args) != 3:
            raise GTPCommandError("time_left requires 3 arguments")
        color = parse_color(args[0])
        try:
            time_s, stones = int(args[1]), int(args[2])
        except ValueError:
            raise GTPCommandError(f"invalid time_left args: {' '.join(args)}")
        self._time_left[color] = (int(time_s), int(stones))
        return True, ""

    def _cmd_final_score(self, args):
        if not self.board.is_terminal():
            raise GTPCommandError("game not finished")
        return True, self.board.result_string(self.komi)

    def _cmd_printsgf(self, args):
        if len(args) != 1:
            raise GTPCommandError("printsgf requires a filename")
        path = Path(args[0])
        path.parent.mkdir(parents=True, exist_ok=True)
        sgf = export_sgf(
            self.board, komi=self.komi,
            player_black=ENGINE_NAME, player_white=ENGINE_NAME,
        )
        path.write_text(sgf, encoding="utf-8")
        return True, str(path)

    def _undo_moves(self, n: int) -> None:
        """Replay the game without its last ``n`` moves (handicap-aware).

        Handicap stones are recorded as ordinary BLACK moves, so re-undoing
        past them removes the handicap along with the stones.
        """
        kept = self.board.moves[:-n]
        board = Board(self.size)
        for move, color in kept:
            board.play(move, color)
        self.board = board
        self._handicap = min(self._handicap, len(kept))

    def _cmd_undo(self, args):
        # Standard GTP undo (GNU Go convention): replay the position from
        # before the last move; ``undo <n>`` removes n moves at once.
        if len(args) > 1:
            raise GTPCommandError("undo accepts at most one argument")
        n = 1
        if args:
            try:
                n = int(args[0])
            except ValueError:
                raise GTPCommandError(f"invalid undo count: {args[0]!r}")
        if n <= 0:
            raise GTPCommandError("invalid undo count: must be positive")
        if n > len(self.board.moves):
            raise GTPCommandError("cannot undo: not enough moves")
        self._undo_moves(n)
        return True, ""

    def _cmd_kgs_chat(self, args):
        # kgs-chat semantics deliberately not implemented (plan Must-NOT):
        # respond with the empty string and record the message so a future
        # platform layer can inspect it. args = [channel, message...]; a bare
        # call is tolerated without error.
        channel = args[0] if args else ""
        self.chat_log.append((channel, " ".join(args[1:])))
        return True, ""


def _build_from_state_dict(state: dict, device: "torch.device"):
    """Reconstruct the architecture from a raw ``state_dict`` tensor shapes."""
    if not isinstance(state, dict) or "policy_head.fc.weight" not in state:
        raise ValueError("model file is neither a checkpoint nor a state_dict")
    fc = state["policy_head.fc.weight"]
    n_logits = int(fc.shape[0])  # N+1
    size = int(round(math.sqrt(n_logits - 1)))
    if size * size + 1 != n_logits:
        raise ValueError(f"policy logit count {n_logits} is not a square + 1")
    channels = int(state["input_conv.weight"].shape[0])
    blocks = max(int(k.split(".")[1]) for k in state if k.startswith("res_blocks.")) + 1
    net = create_model(blocks, channels, size).to(device)
    net.load_state_dict(state)
    return size, net
