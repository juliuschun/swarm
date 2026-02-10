# Swarm: Multi-Agent Collective Intelligence

## What This Is

A Python package (`src/swarm/`) that uses Claude Code agents in a MAKER-inspired loop:

```
RECALL → DECOMPOSE → VOTE each step (escalating K + judge) → COMPOSE → VERIFY → LEARN
  ↑                                                                                  |
  └──────────────────────── re-plan with learnings if verify fails ─────────────────┘
```

Based on: "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025, arXiv:2511.09030)

## Architecture

### Package Structure

```
src/swarm/
├── __init__.py   # Public API exports
├── __main__.py   # python -m swarm support
├── config.py     # Constants, models, roles
├── hooks.py      # register_hook, run_hooks
├── memory.py     # recall, learn, format_learnings, sessions
├── agent.py      # claude() wrapper, red_flag()
├── maker.py      # MAKER loop: decompose, vote, judge, compose, verify, run()
├── opinion.py    # Parallel diverse opinions + consensus
└── cli.py        # Argparse CLI entry point
```

### Models (configurable in config.py)
| Role | Model | Why |
|------|-------|-----|
| Planner (decompose) | Opus | Best judgment to break tasks well |
| Worker (vote steps) | Sonnet | Smart enough for quality, voting adds reliability |
| Judge (pick best) | Opus | Expert evaluation > democratic consensus |
| Composer (merge) | Opus | Best judgment to synthesize coherently |
| Verifier (check) | Opus | Best judgment to catch issues |

### Key Parameters
- `K_START = 2` — initial K, escalates if judge wants more options
- `K_MAX = 5` — max K before forced judge pick
- `MAX_SAMPLES = 12` — max vote samples per step
- `BATCH_SIZE = 3` — parallel workers per vote round
- `MAX_LOOPS = 3` — max verify→re-plan iterations
- `CHECKPOINT_INTERVAL = 10` — checkpoint every N steps for crash recovery

### Two Modes
- `--mode maker` (default): Full decompose → vote → judge → compose → verify → learn loop
- `--mode opinion`: Parallel opinions (3 agents + consensus check). Good for quick questions.

### Key Mechanisms
- **Judge-based selection**: Opus evaluates all worker responses and picks the best. Majority alignment is a signal, not a requirement.
- **Escalating K**: Start with K=2 workers. If judge wants more options, escalate to K=3, then K=5. Easy steps converge fast.
- **Per-step verification**: Each step verified immediately. Failures retried with feedback before moving to next step.
- **Checkpointing**: Progress saved every 10 steps. Crash recovery loses at most 10 steps.
- **State compression**: Older steps compressed into summaries. Last 5 steps kept in full detail.

## Usage

```bash
# MAKER mode (default) — for tasks that need reliability
uv run swarm "Design an auth system with JWT" -v

# Opinion mode — for quick questions / brainstorming
uv run swarm --mode opinion "Best approach for caching?"

# With memory tags
uv run swarm --tags coding,python "Write a rate limiter" -v

# Resume best worker from last run
uv run swarm --resume

# Resume specific session with follow-up
uv run swarm --resume <session_id> "Now add refresh tokens"

# List all resumable sessions
uv run swarm --sessions

# Pipe input
cat spec.md | uv run swarm --stdin -v
```

## Memory

- Learnings stored in `~/.swarm/learnings.jsonl` (append-only JSONL)
- Sessions stored in `~/.swarm/sessions.jsonl` (for resume)
- Checkpoints stored in `~/.swarm/checkpoints/` (for crash recovery)
- Learnings are injected into all agent prompts automatically
- System extracts `LEARNING: [category] content` from agent outputs

## Key Principles

1. **An expert judge beats a democratic vote** — Opus judge picks quality over popularity
2. **Catch errors early** — per-step verification beats post-hoc checking
3. **Decompose before you swarm** — break into steps, vote each step
4. **Red-flag before judging** — discard structurally suspicious responses
5. **Separate planning from execution** — Opus plans, Sonnet executes
6. **Memory compounds** — a single agent with learnings beats 10 amnesiac agents
7. **Diversity via roles, not temperature** — pragmatist/skeptic/innovator roles give 15-25% quality improvement

## MAKER Paper Adaptation

This is a MAKER-inspired approach adapted for open-ended reasoning tasks.

**Preserved from paper**: Decomposition, diverse generation, red-flagging, model specialization, learning loop.

**Evolved beyond paper**: Judge-based selection (Opus picks best) instead of exact string consensus. Per-step verification instead of post-hoc only. Escalating K instead of fixed K. State compression for longer step chains.

**Honest assessment**: The paper achieved zero errors at 1M steps because Towers of Hanoi has deterministic ground truth. Our system handles ~100-1000 steps reliably for open-ended tasks. The mathematical guarantee becomes approximate — reliability compounds through judge quality + role diversity + learning, but without formal proof.

## Integration with swarm-team Go CLI

The swarm CLI can be called from the Go CLI for two purposes:

### 1. Inject learnings into lead prompts
```bash
LEARNINGS=$(uv run swarm --recall --tags coding)
swarm lead spawn my-lead $SESSION_DIR -a "Do X. $LEARNINGS"
```

### 2. Decision consensus before committing
```bash
uv run swarm --mode opinion --json "JWT vs sessions for this project?"
```

### 3. Hook into lead workflow
```python
from swarm import register_hook

async def notify_lead(ctx):
    import json
    with open(f"{ctx['cwd']}/result.json", "w") as f:
        json.dump({"answer": ctx["answer"], "cost": ctx["cost"]}, f)
    return {}

register_hook("post_loop", notify_lead)
```

## Dependencies

- Python 3.10+
- Claude Code CLI (`claude` command must be available)
- No pip dependencies required
