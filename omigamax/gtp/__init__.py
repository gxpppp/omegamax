"""Standard GTP engine interface and platform-facing command set (todos 18-19)."""

from omigamax.gtp.gtp import (
    GTPCommandError,
    GTPEngine,
    VERSION,
    parse_color,
    parse_vertex,
    to_gtp,
)

__all__ = [
    "GTPCommandError",
    "GTPEngine",
    "VERSION",
    "parse_color",
    "parse_vertex",
    "to_gtp",
]
