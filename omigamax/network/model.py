"""ResNet policy-value network (todo 6).

AlphaGo Zero style convolutional neural network: a residual tower with a
policy head outputting ``board_size**2 + 1`` logits (361 intersections + 1
pass) and a value head outputting a single scalar in ``[-1, 1]``.

Architecture (AGZ Nature 2017 Fig. 2 / Methods, b10c128 default):

* input block:  conv 3x3 (channels) -> batch-norm -> ReLU
* residual tower: ``blocks`` x [ conv 3x3 -> BN -> ReLU -> conv 3x3 -> BN ],
  skip-connection add, then ReLU
* policy head:  conv 1x1 (2 ch) -> BN -> flatten (2*N) -> linear -> N+1 logits
* value head:   conv 1x1 (1 ch) -> BN -> flatten (N) -> linear 128 -> ReLU
  -> linear 1 -> tanh

The input is ``(N, 17, board_size, board_size)`` float32 (the 17 AGZ feature
planes, encoded by todo 7; the network itself only needs the plane count).

``blocks``/``channels``/``board_size`` are fully configurable so small boards
(9x9) can be used to speed up tests. The factory :func:`create_model` builds
and AGZ-initializes the network; :meth:`PolicyValueNetwork.save` /
:meth:`PolicyValueNetwork.load` persist and restore the ``state_dict``.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from omigamax.config import load_config

# Number of input feature planes: 8 history steps x 2 colors + current player.
# Fixed by the AGZ 17-plane encoding (todo 7); the network must accept
# (N, INPUT_PLANES, board_size, board_size).
INPUT_PLANES = 17


def policy_logit_count(board_size: int) -> int:
    """Number of policy logits: one per intersection plus the pass move."""
    return board_size * board_size + 1


class ResidualBlock(nn.Module):
    """AGZ residual block: conv3x3->BN->ReLU->conv3x3->BN with skip add."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class PolicyHead(nn.Module):
    """Policy head: conv1x1 (2ch) -> BN -> flatten -> linear to N+1 logits.

    The pass move is the last logit (index ``board_size**2``).
    """

    def __init__(self, channels: int, board_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * board_size * board_size, policy_logit_count(board_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        x = x.flatten(start_dim=1)  # (B, 2*N)
        return self.fc(x)  # (B, N+1)


class ValueHead(nn.Module):
    """Value head: conv1x1 (1ch) -> BN -> flatten -> 128 FC -> ReLU -> 1 FC -> tanh."""

    def __init__(self, channels: int, board_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(board_size * board_size, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU(inplace=False)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        x = x.flatten(start_dim=1)  # (B, N)
        x = self.relu(self.fc1(x))
        return self.tanh(self.fc2(x))


class PolicyValueNetwork(nn.Module):
    """AlphaGo Zero style ResNet with separate policy and value heads."""

    def __init__(self, blocks: int, channels: int, board_size: int) -> None:
        super().__init__()
        self.blocks = int(blocks)
        self.channels = int(channels)
        self.board_size = int(board_size)
        self.input_conv = nn.Conv2d(INPUT_PLANES, channels, kernel_size=3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.input_relu = nn.ReLU(inplace=False)
        self.res_blocks = nn.ModuleList(ResidualBlock(channels) for _ in range(self.blocks))
        self.policy_head = PolicyHead(channels, board_size)
        self.value_head = ValueHead(channels, board_size)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(policy_logits (B, N+1), value (B, 1))`` for input ``(B, 17, N, N)``."""
        x = self.input_relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            x = block(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

    def _init_weights(self) -> None:
        """AGZ-style initialization.

        Conv/linear weights ~ N(0, 0.05), biases zero, BN affine set to
        identity. The second conv of the last residual block is zeroed so the
        residual tower starts as the identity mapping.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if self.res_blocks:
            nn.init.zeros_(self.res_blocks[-1].conv2.weight)

    def save(self, path: "str | Path") -> None:
        """Persist the model ``state_dict`` to ``path`` (creates parent dirs)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: "str | Path", blocks: int, channels: int, board_size: int) -> "PolicyValueNetwork":
        """Build a fresh model and load a ``state_dict`` written by :meth:`save`."""
        model = create_model(blocks, channels, board_size)
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model


def create_model(blocks: int, channels: int, board_size: int) -> nn.Module:
    """Factory: build an AGZ-style policy-value ResNet (weights initialized).

    Parameters are fully configurable so small boards (e.g. 9x9) can be used
    in tests; the default b10c128 topology comes from ``config/default.yaml``
    (blocks=10, channels=128, board_size=19, ~3.3M parameters).
    """
    if not (isinstance(blocks, int) and blocks >= 1):
        raise ValueError(f"blocks must be a positive int, got {blocks!r}")
    if not (isinstance(channels, int) and channels >= 1):
        raise ValueError(f"channels must be a positive int, got {channels!r}")
    if not (isinstance(board_size, int) and board_size >= 2):
        raise ValueError(f"board_size must be an int >= 2, got {board_size!r}")
    return PolicyValueNetwork(blocks=blocks, channels=channels, board_size=board_size)


def create_model_from_config(config: "dict | None" = None) -> nn.Module:
    """Build the network from a config dict (defaults to config/default.yaml)."""
    cfg = load_config() if config is None else config
    return create_model(blocks=cfg["blocks"], channels=cfg["channels"], board_size=cfg["board_size"])
