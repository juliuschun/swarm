"""Hook system for extending the MAKER loop."""

import asyncio
import sys

# Hook points: pre_decompose, post_decompose, pre_vote, post_vote,
#              post_compose, pre_verify, post_verify, post_learn, post_loop

_hooks: dict[str, list] = {}


def register_hook(point: str, fn) -> None:
    """Register a hook function for a given point in the MAKER loop.

    Hook functions are async callables that receive a context dict and return
    a dict of overrides. Returning an empty dict means no changes.

    Example:
        async def my_hook(ctx):
            print(f"Step {ctx.get('step_num')} done")
            return {}
        register_hook("post_vote", my_hook)
    """
    if point not in _hooks:
        _hooks[point] = []
    _hooks[point].append(fn)


async def run_hooks(point: str, ctx: dict) -> dict:
    """Run all hooks for a given point, merging returned overrides."""
    result = {}
    for fn in _hooks.get(point, []):
        try:
            override = await fn(ctx) if asyncio.iscoroutinefunction(fn) else fn(ctx)
            if isinstance(override, dict):
                result.update(override)
        except Exception as e:
            print(f"[hook:{point}] {fn.__name__} failed: {e}", file=sys.stderr)
    return result
