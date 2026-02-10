"""Swarm: MAKER-informed multi-agent collective intelligence."""

from .maker import run
from .memory import recall, learn, format_learnings
from .hooks import register_hook, run_hooks
from .agent import claude, red_flag
from .opinion import run_opinion

__all__ = [
    "run", "recall", "learn", "format_learnings",
    "register_hook", "run_hooks", "claude", "red_flag", "run_opinion",
]
