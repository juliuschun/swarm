"""CLI entry point for swarm."""

import asyncio
import json
import os
import sys

from .agent import claude
from .config import MAX_LOOPS, TIMEOUT_SECONDS
from .maker import run
from .memory import format_learnings, get_best_session, load_sessions, recall


def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Swarm: MAKER-informed multi-agent collective intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  uv run swarm "Design an auth system" -v
  uv run swarm --tags coding "Write a rate limiter" -v
  uv run swarm --mode opinion "Is Rust better than Go?"
  uv run swarm --resume
  uv run swarm --sessions""",
    )
    p.add_argument("prompt", nargs="?", help="The task or question")
    p.add_argument("--mode", choices=["maker", "opinion"], default="maker",
                   help="maker=decompose+vote+verify loop (default), opinion=v1 parallel opinions")
    p.add_argument("--workers", type=int, default=3,
                   help="Number of workers for opinion mode (default: 3)")
    p.add_argument("--worker-model", type=str, default="sonnet",
                   help="Model for workers: sonnet (default) or haiku")
    p.add_argument("--tags", type=str, default="", help="Comma-separated memory tags")
    p.add_argument("-t", "--timeout", type=int, default=TIMEOUT_SECONDS,
                   help=f"Timeout per agent call (default: {TIMEOUT_SECONDS}s)")
    p.add_argument("--max-loops", type=int, default=MAX_LOOPS,
                   help=f"Max verify→re-plan loops (default: {MAX_LOOPS})")
    p.add_argument("--tools", action="store_true",
                   help="Give workers tool access (Read, Glob, Grep, Bash - read-only)")
    p.add_argument("--tools-rw", action="store_true",
                   help="Give workers read-write tool access (Read, Glob, Grep, Bash, Edit, Write)")
    p.add_argument("--cwd", type=str, default=None,
                   help="Working directory for workers (default: current dir)")
    p.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    p.add_argument("-v", "--verbose", action="store_true", help="Print detailed progress")
    p.add_argument("--quiet", action="store_true", help="Suppress progress output (default: show progress in TTY)")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--resume", nargs="?", const="BEST", metavar="SESSION_ID",
                   help="Resume a worker session (default: best from last run)")
    p.add_argument("--sessions", action="store_true", help="List resumable sessions")
    p.add_argument("--recall", action="store_true",
                   help="Output past learnings (for injection into other systems)")
    p.add_argument("--recall-limit", type=int, default=10,
                   help="Max learnings to recall (default: 10)")
    args = p.parse_args()

    # Warn if using read-write tools
    if args.tools_rw:
        print("WARNING: --tools-rw enables parallel workers to EDIT and WRITE files. "
              "Risk of data corruption if multiple workers write to same files simultaneously.",
              file=sys.stderr)

    # Recall mode
    if args.recall:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
        learnings = recall(tags, limit=args.recall_limit)
        if args.json:
            print(json.dumps([l for l in learnings]))
        else:
            text = format_learnings(learnings)
            print(text if text else "No learnings found.")
        return

    # Sessions mode
    if args.sessions:
        sessions = load_sessions(20)
        if not sessions:
            print("No sessions saved yet.")
        else:
            print(f"{'ID':<40} {'Role':<12} {'Conf':>5}  {'Time':<20}  Prompt")
            print("-" * 100)
            for s in sessions:
                print(f"{s['session_id']:<40} {s['role']:<12} {s['confidence']:>4.1f}  "
                      f"{s['ts'][:19]:<20}  {s['prompt'][:40]}...")
        return

    # Resume mode
    if args.resume is not None:
        if args.resume == "BEST":
            session = get_best_session()
            if not session:
                print("No sessions to resume.", file=sys.stderr)
                sys.exit(1)
            session_id = session["session_id"]
            print(f"[resume] {session['role']} (confidence={session['confidence']}/10)", file=sys.stderr)
        else:
            session_id = args.resume

        follow_up = args.prompt or "Continue and refine your previous answer. Go deeper."
        result = asyncio.run(claude(follow_up, session_id=session_id))
        if result["error"]:
            print(f"[error] {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(result["content"])
        return

    # Run swarm
    prompt = sys.stdin.read().strip() if args.stdin else args.prompt
    if not prompt:
        p.print_help()
        sys.exit(1)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None

    cwd = args.cwd or os.getcwd()
    show_progress = (not args.quiet and not args.json and sys.stderr.isatty()) or args.verbose
    try:
        result = asyncio.run(run(prompt, tags=tags, verbose=args.verbose, mode=args.mode,
                                 tools=args.tools, tools_rw=args.tools_rw, cwd=cwd, workers=args.workers,
                                 worker_model=args.worker_model,
                                 timeout=args.timeout, max_loops=args.max_loops,
                                 progress=show_progress))
    except KeyboardInterrupt:
        print("\n[interrupted] Shutting down...", file=sys.stderr)
        sys.exit(130)

    if isinstance(result, dict):
        answer = result.get("answer", "")
        cost = result.get("cost", 0)
    else:
        answer = result
        cost = 0

    if args.json:
        output = {"answer": answer, "prompt": prompt, "tags": tags, "mode": args.mode, "cost": round(cost, 3)}
        print(json.dumps(output))
    else:
        print(answer)
