# Swarm: Multi-Agent Collective Intelligence

## What This Is

A single-file Python system (`swarm.py`) that uses Claude Code agents as workers in a MAKER-inspired loop:

```
RECALL → DECOMPOSE → VOTE each step (adaptive, ahead-by-K) → COMPOSE → VERIFY → LEARN
  ↑                                                                               |
  └────────────────────── re-plan with learnings if verify fails ─────────────────┘
```

Based on: "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025, arXiv:2511.09030)

## Architecture

### Models (configurable at top of swarm.py)
| Role | Model | Why |
|------|-------|-----|
| Planner (decompose) | Sonnet | Needs intelligence to break tasks well |
| Worker (vote steps) | Haiku | Cheap — voting makes it reliable (MAKER insight) |
| Composer (merge) | Sonnet | Needs to synthesize coherently |
| Verifier (check) | Sonnet | Needs judgment to catch issues |

### Key Parameters
- `K_AHEAD = 3` — votes ahead needed to win (from MAKER paper: sufficient for 1M+ steps)
- `MAX_SAMPLES = 10` — max vote samples per step before fallback
- `BATCH_SIZE = 3` — parallel workers per vote round
- `MAX_LOOPS = 3` — max verify→re-plan iterations

### Two Modes
- `--mode maker` (default): Full decompose → vote → compose → verify → learn loop
- `--mode opinion`: v1 parallel opinions (3 agents + consensus check). Good for quick questions.

## Usage

```bash
# MAKER mode (default) — for tasks that need reliability
python3 swarm.py "Design an auth system with JWT" -v

# Opinion mode — for quick questions / brainstorming
python3 swarm.py --mode opinion "Best approach for caching?"

# With memory tags
python3 swarm.py --tags coding,python "Write a rate limiter" -v

# Resume best worker from last run
python3 swarm.py --resume

# Resume specific session
python3 swarm.py --resume <session_id> "Now add refresh tokens"

# List all resumable sessions
python3 swarm.py --sessions

# Pipe input
cat spec.md | python3 swarm.py --stdin -v
```

## Memory

- Learnings stored in `~/.swarm/learnings.jsonl` (append-only JSONL)
- Sessions stored in `~/.swarm/sessions.jsonl` (for resume)
- Learnings are injected into all agent prompts automatically
- System extracts `LEARNING: [category] content` from agent outputs

## Key Principles (from research + testing)

1. **Cheap models + voting > expensive models solo** — Haiku with K=3 voting is both cheaper AND more reliable than single Sonnet
2. **Decompose before you swarm** — Don't send big tasks to parallel agents. Break into steps, vote each step.
3. **Red-flag before counting** — Discard structurally suspicious responses (too long, empty, low confidence) before they pollute votes
4. **Separate planning from execution** — Sonnet plans, Haiku executes. Different models for different jobs.
5. **Memory compounds** — A single agent with learnings beats 10 amnesiac agents
6. **Diversity via roles, not temperature** — Pragmatist/skeptic/innovator roles give 15-25% quality improvement vs temperature variation

## Dependencies

- Python 3.10+
- Claude Code CLI (`claude` command must be available)
- No pip dependencies required
