"""AlphaGo Zero training loss and single train step (todo 8; used by todo 14).

The AGZ training objective (Nature 550:354-359, 2017, Methods) has two loss
components plus L2 weight regularization:

    L = policy_cross_entropy(logits, pi) + value_mse(value, z)   [+ L2(1e-4)]

* ``policy_cross_entropy``: soft-target cross entropy over the ``N**2 + 1``
  logits (``pi`` = MCTS search policy distribution, todo 13), batch-averaged.
* ``value_mse``: mean squared error of the scalar value head against the game
  outcome ``z in [-1, 1]``.
* L2 regularization (config ``l2``, default ``1e-4``): applied as SGD
  ``weight_decay`` in the optimizer (see :func:`make_sgd_optimizer` /
  :func:`train_step`). This is the canonical AGZ implementation used by
  KataGo, Leela Zero and every major AGZ port. An alternative form that adds
  the squared weight norm explicitly into the *loss graph* was empirically
  found to DESTABILIZE training at the locked ``lr=0.2`` + momentum 0.9 on
  the b10c128 @ batch-128 smoke: the momentum accumulator carries the L2
  gradient into a weight-blowup regime (weight norm ~11x in 25 steps, loss
  diverges 3-15x over 200 steps, reproducible across seeds and target
  recipes; documented in ``.omo/evidence/omigamax-go/task-8-smoke.json``).
  The weight-decay form converges to ``loss_last/loss_first ~ 0.15-0.20``.

:func:`agz_loss` returns the training objective value (CE + MSE). The L2
regularization magnitude for a network is available via :func:`weight_l2`
(analytic, used for monitoring/evidence -- it is deliberately NOT part of
the loss graph). :func:`train_step` runs one optimizer step and returns the
scalar loss. FP16/autocast is supported through ``use_fp16`` (locked off by
default; ``--fp16`` exercises the toggle).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def policy_cross_entropy(logits: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """Soft-target cross entropy: ``-mean_c sum_c pi_c * log_softmax_c(logits)``.

    ``logits`` is ``(B, N**2 + 1)``; ``pi`` is a probability distribution over
    the same classes (soft targets, e.g. the MCTS search policy). Batch-mean.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return torch.mean(-torch.sum(pi * log_probs, dim=-1))


def value_mse(value: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the value head against the game outcome ``z``."""
    return F.mse_loss(value, z)


def agz_loss(
    logits: torch.Tensor,
    value: torch.Tensor,
    pi: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """AGZ training objective value: ``policy_cross_entropy + value_mse``.

    L2 regularization is applied separately via SGD ``weight_decay`` (see
    :func:`make_sgd_optimizer`); it is not part of this scalar so the
    reported loss is exactly the value the optimizer minimizes.
    """
    return policy_cross_entropy(logits, pi) + value_mse(value, z)


def weight_l2(model: torch.nn.Module) -> torch.Tensor:
    """Squared Frobenius norm of all conv/linear weights (``ndim >= 2``).

    Analytic monitor of the plan's L2 term (``l2 * ||W||^2``). Excludes 1-D
    parameters (biases, batch-norm affine scales), matching the AGZ "L2 on
    weights, not biases" rule. Deliberately detached from the loss graph --
    regularization is applied via SGD weight decay.
    """
    total = torch.zeros(
        (), device=next(model.parameters()).device, dtype=torch.float32
    )
    for p in model.parameters():
        if p.ndim >= 2:
            total = total + p.detach().pow(2).sum()
    return total


def make_sgd_optimizer(
    model: torch.nn.Module,
    lr: float,
    momentum: float,
    l2: float,
) -> torch.optim.Optimizer:
    """Build the AGZ SGD optimizer (momentum) with L2 weight regularization.

    ``weight_decay=l2`` is the plan's ``l2=1e-4`` regularization. See module
    docstring for why it is applied here rather than as an explicit loss term.
    """
    return torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=l2
    )


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    pi: torch.Tensor,
    z: torch.Tensor,
    use_fp16: bool = False,
) -> float:
    """Run one AGZ training step on a batch; return the scalar loss (CE + MSE).

    Puts the model in ``train()`` mode (batch-norm statistics update, as in
    the real training loop), zeroes gradients, computes the AGZ loss
    (optionally under ``torch.autocast`` for the FP16 toggle smoke),
    back-propagates and steps the optimizer (which carries the L2
    ``weight_decay``).

    Args:
        model: the policy-value network.
        optimizer: SGD (momentum, ``weight_decay=l2``) optimizer from
            :func:`make_sgd_optimizer`.
        inputs: ``(B, 17, N, N)`` feature planes on the model's device.
        pi: ``(B, N**2 + 1)`` target policy distribution.
        z: ``(B, 1)`` target game outcome.
        use_fp16: if True wrap the forward pass in ``torch.autocast("cuda")``
            (FP16 toggle smoke; the locked config default is ``fp16=false``).

    Returns:
        The detached scalar loss (python float) of this step.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    autocast_ctx = torch.autocast(
        "cuda", enabled=bool(use_fp16 and inputs.is_cuda), dtype=torch.float16
    )
    with autocast_ctx:
        logits, value = model(inputs)
        loss = agz_loss(logits, value, pi, z)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())
