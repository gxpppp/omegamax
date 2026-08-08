"""P6b: ``--save-every`` periodic-checkpoint tests for the pretrain CLI.

Covers:
(a) ``--save-every``: a 25-step run with save-every 10 writes intermediate
    checkpoints at steps 10 and 20 (verified via the printed ``[save]``
    lines -- the on-disk checkpoint is overwritten each save so ``--resume``
    picks up the latest one, ending at step 25) and logs exactly one line per
    logged step (no duplicate step-0 across chunks);
(b) ``--resume`` from the periodic checkpoint continues step 20 -> 25 without
    re-logging step 0 (the jsonl keeps a single line per step value);
(c) default behavior unchanged: without ``--save-every`` (i.e. 5000 > steps),
    only the final checkpoint is saved -- no ``[save]`` line, and the on-disk
    checkpoint carries the final ``global_step``.
"""

import json

import pytest

from omigamax.cli.pretrain import _build_parser, main
from omigamax.train.train import load_checkpoint

from test_pretrain import make_synthetic_chunks


def _argv(data_dir, ckpt, log, **kw):
    args_list = [
        "--data-dir", str(data_dir),
        "--steps", "25",
        "--batch-size", "8",
        "--lr", "0.02",
        "--blocks", "1",
        "--channels", "8",
        "--board-size", "9",
        "--seed", "0",
        "--device", "cpu",
        "--checkpoint", str(ckpt),
        "--log-path", str(log),
        "--log-every", "10",
    ]
    for k, v in kw.items():
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                args_list.append(flag)
        else:
            args_list += [flag, str(v)]
    return args_list


def _read_steps(log_path):
    steps = []
    if not log_path.exists():
        return steps
    for line in log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            steps.append(int(json.loads(line)["step"]))
    return steps


# ---------------------------------------------------------------------------
# (a) periodic checkpointing
# ---------------------------------------------------------------------------

def test_cli_save_every_writes_periodic_checkpoints(tmp_path, capsys):
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[128])
    ckpt, log = tmp_path / "pretrain.pt", tmp_path / "pretrain.jsonl"

    rc = main(_argv(data_dir, ckpt, log, save_every=10, steps=25))
    assert rc == 0

    out = capsys.readouterr().out
    # intermediate saves at each completed block boundary...
    assert "[save] step=10 checkpoint=" in out
    assert "[save] step=20 checkpoint=" in out
    # ...but the final save stays silent (unchanged behavior)
    assert "[save] step=25" not in out

    # the on-disk checkpoint (overwritten by each save) ends at the final step
    ck = load_checkpoint(ckpt)
    assert ck["global_step"] == 25
    assert ck["arch"] == {"blocks": 1, "channels": 8, "board_size": board}

    # one line per logged step across chunks -- no duplicate step-0
    assert _read_steps(log) == [0, 10, 20]


# ---------------------------------------------------------------------------
# (b) resume from a periodic checkpoint
# ---------------------------------------------------------------------------

def test_cli_resume_from_periodic_checkpoint_no_duplicate_step0(tmp_path, capsys):
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[128])
    ckpt, log = tmp_path / "pretrain.pt", tmp_path / "pretrain.jsonl"

    # run 1: 20 steps in 10-step blocks -> periodic checkpoint at step 10,
    # final checkpoint at step 20; jsonl logs steps 0 and 10
    rc1 = main(_argv(data_dir, ckpt, log, save_every=10, steps=20))
    assert rc1 == 0
    assert load_checkpoint(ckpt)["global_step"] == 20
    assert _read_steps(log) == [0, 10]
    capsys.readouterr()  # discard run-1 stdout

    # run 2: --resume continues 20 -> 25 and appends the step-20 line once
    rc2 = main(_argv(data_dir, ckpt, log, save_every=10, steps=5, resume=True))
    assert rc2 == 0
    assert load_checkpoint(ckpt)["global_step"] == 25

    steps = _read_steps(log)
    assert steps == [0, 10, 20]
    assert len(steps) == len(set(steps)), f"duplicate step lines: {steps}"
    assert steps.count(0) == 1, "step 0 was re-logged on resume"


# ---------------------------------------------------------------------------
# (c) default behavior unchanged
# ---------------------------------------------------------------------------

def test_cli_default_save_every_only_final_checkpoint(tmp_path, capsys):
    board = 9
    data_dir = make_synthetic_chunks(tmp_path / "data", board, sizes=[128])
    ckpt, log = tmp_path / "pretrain.pt", tmp_path / "pretrain.jsonl"

    rc = main(_argv(data_dir, ckpt, log, steps=25))  # default save_every=5000
    assert rc == 0

    out = capsys.readouterr().out
    assert "[save]" not in out, "default run must not print [save] lines"
    assert load_checkpoint(ckpt)["global_step"] == 25
    assert _read_steps(log) == [0, 10, 20]


def test_cli_save_every_default_and_validation():
    ap = _build_parser()
    assert ap.parse_args([]).save_every == 5000
    with pytest.raises(SystemExit):
        main(["--save-every", "0", "--steps", "1", "--device", "cpu"])
