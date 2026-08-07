"""GTP engine command-line entry point (todo 18).

Reads GTP commands from stdin line-by-line and writes the standard response
frame (``=id text`` / ``?id text`` + a blank terminator line) to stdout,
flushing after every response.

Windows pipe specifics (plan todo 18, Oracle review):

* every response is explicitly flushed -- a line-buffered pipe would stall a
  GTP peer that is waiting for a complete frame;
* output is LF-only: ``sys.stdout`` is reconfigured with ``newline="\\n"`` so
  the text layer never emits CRLF on Windows;
* input is read in universal-newlines mode and every line is ``rstrip``ped of
  ``\\r`` -- a peer sending CRLF line endings is tolerated;
* for fully unbuffered IO launch as ``python -u -m omigamax.cli.gtp_main``.

Usage::

    echo -e "protocol_version\\nlist_commands\\nquit" | uv run python -m omigamax.cli.gtp_main --model models/best.pt
    uv run python -m omigamax.cli.gtp_main --model models/best.pt --simulations 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omigamax.gtp.gtp import GTPEngine


def _reconfigure_stdio() -> None:
    """LF-only stdout (no CRLF translation) + universal-newline stdin."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="\n", line_buffering=True)
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(newline=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="omigamax GTP engine (todo 18): standard Go Text Protocol "
                    "on stdin/stdout, MCTS genmove, deterministic play."
    )
    parser.add_argument(
        "--model", type=str, default="models/best.pt",
        help="checkpoint or state_dict to load (default models/best.pt; "
             "models/ is relative to the project root)",
    )
    parser.add_argument(
        "--simulations", type=int, default=None,
        help="MCTS simulations per genmove (default config simulations=200)",
    )
    parser.add_argument(
        "--komi", type=float, default=None,
        help="komi on white (default config komi=7.5)",
    )
    parser.add_argument(
        "--board-size", type=int, default=None,
        help="initial board size (default config board_size=19)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="torch device (default: cuda if available)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--config", type=str, default=None,
        help="config YAML path (default: config/default.yaml)",
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    if args.model and not Path(args.model).exists():
        print(
            f"error: model file not found: {args.model}",
            file=sys.stderr, flush=True,
        )
        return 1
    engine = GTPEngine(
        model_path=args.model,
        board_size=args.board_size,
        komi=args.komi,
        simulations=args.simulations,
        device=args.device,
        seed=args.seed,
        config_path=args.config,
    )
    _reconfigure_stdio()
    for line in sys.stdin:
        response = engine.handle_line(line)
        if response is None:
            continue
        for out in response:
            sys.stdout.write(out + "\n")
        sys.stdout.flush()
        if engine.should_quit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
