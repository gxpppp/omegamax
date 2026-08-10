"""omigamax.web: KataGui-style web training monitor.

Read-only Flask dashboard over the RL loop's artifacts (``logs/train.jsonl``
and ``data/selfplay/*.npz``). Never imports the training loop; the server can
run in a separate process while training runs.
"""
