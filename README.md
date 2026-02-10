# Swarm

Second opinion machine for Claude Code. Get verification where mistakes are expensive.

No dependencies beyond the `claude` CLI.

## What It Does

```
RECALL → DECOMPOSE → VOTE each step (escalating K + judge) → COMPOSE → VERIFY → LEARN
  ↑                                                                                  |
  └──────────────────────── re-plan with learnings if verify fails ─────────────────┘
```

Based on [MAKER](https://arxiv.org/abs/2511.09030) — "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025).

## Quick Start

```bash
# Design question (opinion mode — fast, 3 parallel agents)
uv run swarm --mode opinion "JWT vs sessions for auth?" -v

# Coding task (MAKER mode — decompose, vote, verify)
uv run swarm --mode maker "Add input validation" -v

# With tool access (workers can read files, search code)
uv run swarm --mode maker --tools "Review the auth flow for vulnerabilities" -v

# Read-write tools (workers can edit files — use with caution)
uv run swarm --mode maker --tools-rw "Refactor the error handling" -v
```

## Two Modes

### Opinion Mode (`--mode opinion`)
3 diverse agents answer in parallel → consensus check → judge if needed. Fast. Good for design questions and trade-offs.

### MAKER Mode (`--mode maker`, default)
Opus decomposes → Sonnet workers generate diverse responses → Opus judge picks best → Opus verifies → learn → retry if failed. Reliable. Good for tasks where correctness matters.

## Key Flags

| Flag | Does |
|------|------|
| `--mode maker\|opinion` | MAKER loop (default) or parallel opinions |
| `--tools` | Workers get read-only access (Read, Glob, Grep, Bash) |
| `--tools-rw` | Workers get read-write access (Edit, Write too) |
| `--workers N` | Number of opinion-mode workers (default: 3) |
| `--cwd PATH` | Working directory for workers |
| `--tags x,y` | Memory tags for recall/learn |
| `-v` | Verbose progress output |
| `-t N` | Timeout per agent call in seconds (default: 120) |
| `--worker-model MODEL` | Model for workers: sonnet (default) or haiku |
| `--max-loops N` | Max verify-replan loops (default: 3) |
| `--json` | Output result as JSON |
| `--resume [ID]` | Resume a worker session (default: best from last run) |
| `--sessions` | List resumable sessions |
| `--recall` | Output past learnings for injection into other systems |

## How It Works

### Judge-Based Selection (not consensus voting)

At each step, multiple workers (Sonnet) attempt the same task with different role prompts. An Opus judge evaluates all responses and picks the best one — using majority alignment as a quality signal, not a requirement. A brilliant minority answer can win over a mediocre majority.

If the judge wants more options, K escalates automatically (2 → 3 → 5). Easy steps converge fast. Hard steps get more samples.

### Per-Step Verification

Each step is verified immediately after the judge picks. If the verifier catches an issue, the step is retried with the feedback. Errors are caught before they cascade — not after all steps complete.

### Role Diversity

Workers get different perspectives via role prompts:
- Pragmatist, Skeptic, Innovator, Systems Thinker, Historian, Contrarian

This gives 15-25% quality improvement vs temperature variation (~2%). The prompt is what prevents mode collapse, not randomness.

### Model Separation

| Role | Model | Why |
|------|-------|-----|
| Planner | Opus | Best judgment to decompose well |
| Worker | Sonnet | Smart enough for quality, voting adds reliability |
| Judge | Opus | Expert evaluation > democratic consensus |
| Composer | Opus | Coherent synthesis |
| Verifier | Opus | Best judgment to catch issues |

### Crash Recovery

Progress is checkpointed every 10 steps. If a run crashes at step 50, resume loses at most 10 steps of work — not all 50.

## Memory

Learnings stored in `~/.swarm/learnings.jsonl`. Append-only JSONL.

- Before each run: `recall()` injects relevant past learnings into agent prompts
- After verify: `learn()` extracts `LEARNING: [category] content` from output
- Categories: mistake, strategy, pattern, constraint
- Confidence-sorted, tag-filtered retrieval

## Principles

1. **An expert judge beats a democratic vote** — majority alignment is a signal, not a mechanism
2. **Catch errors early** — per-step verification beats post-hoc checking
3. **Decompose before you swarm** — vote per step, not per task
4. **Red-flag before judging** — discard bad responses, don't try to fix them
5. **Separate planning from execution** — different models for different jobs
6. **Memory compounds** — one agent with learnings beats 10 without
7. **Diversity via roles, not temperature**

## Installation

### As a Claude Code plugin (recommended)

```bash
/plugin marketplace add yourname/swarm
/plugin install swarm
```

Then use `/swarm "your question"` or just ask Claude to "ask the swarm".

### As a standalone CLI

```bash
git clone https://github.com/yourname/swarm.git
cd swarm
uv run swarm --help
# That's it. uv run handles the rest.
```

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (`uv run` auto-creates venv + installs on first run)
- [Claude Code CLI](https://claude.ai/claude-code) (`claude` command must be in PATH)
- No pip dependencies

## Data Storage

All data is stored in `~/.swarm/`:
- `learnings.jsonl` — Accumulated learnings (append-only JSONL)
- `sessions.jsonl` — Resumable worker sessions
- `checkpoints/` — Mid-run progress for crash recovery

This directory is created automatically on first run.

## MAKER Paper Adaptation

This is a **MAKER-inspired** system adapted for open-ended reasoning tasks. Key differences from the original paper:

- **Selection**: Opus judge picks best response instead of exact string consensus (necessary because open-ended tasks have no deterministic ground truth)
- **Verification**: Per-step Opus verification + retry instead of rule-based correctness checking
- **State**: Compressed history + recent context window instead of formal state tracking
- **Decomposition**: Tasks are broken into reasoning steps, not mechanical moves

The core principles (decompose, diverse generation, red-flag, learn) are preserved. The mathematical guarantee (`P(correct) → 1` as K increases) becomes approximate rather than exact — reliability compounds through judge quality + role diversity + learning, but without formal proof.

See [the paper](https://arxiv.org/abs/2511.09030) for the original formulation.

## License

MIT
