# Swarm

Multi-agent collective intelligence for Claude Code.

One Python file. No dependencies beyond the `claude` CLI.

## What It Does

```
RECALL → DECOMPOSE → VOTE each step (adaptive K-ahead) → COMPOSE → VERIFY → LEARN
  ↑                                                                              |
  └──────────────────── re-plan with learnings if verify fails ──────────────────┘
```

Based on [MAKER](https://arxiv.org/abs/2511.09030) — "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025).

## Quick Start

```bash
# Design question (opinion mode — fast, 3 parallel agents)
python3 swarm.py --mode opinion "JWT vs sessions for auth?" -v

# Coding task (MAKER mode — decompose, vote, verify)
python3 swarm.py --mode maker "Add input validation to swarm.py" -v

# With tool access (workers can read files, search code)
python3 swarm.py --mode maker --tools "Read swarm.py and add error logging" -v

# Read-write tools (workers can edit files — use with caution)
python3 swarm.py --mode maker --tools-rw "Refactor the red_flag function" -v
```

## Two Modes

### Opinion Mode (`--mode opinion`)
3 diverse agents answer in parallel → consensus check → judge if needed. Fast. Good for design questions and trade-offs.

### MAKER Mode (`--mode maker`, default)
Sonnet decomposes → Haiku workers vote each step (K-ahead) → Sonnet composes → Sonnet verifies → learn → retry if failed. Reliable. Good for tasks where correctness matters.

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
| `-k N` | Votes ahead to win (default: 3) |
| `-t N` | Timeout per agent call in seconds (default: 120) |
| `--worker-model MODEL` | Model for workers: haiku (default) or sonnet |
| `--max-cost N` | Max total cost in USD before stopping (default: 1.00) |
| `--max-loops N` | Max verify-replan loops (default: 3) |
| `--json` | Output result as JSON |
| `--resume [ID]` | Resume a worker session (default: best from last run) |
| `--sessions` | List resumable sessions |

## Memory

Learnings stored in `~/.swarm/learnings.jsonl`. Append-only JSONL.

- Before each run: `recall()` injects relevant past learnings into agent prompts
- After verify: `learn()` extracts `LEARNING: [category] content` from output
- Categories: mistake, strategy, pattern, constraint
- Confidence-sorted, tag-filtered retrieval

## How It Works

### MAKER Voting (the core insight)

At each step, multiple cheap workers (Haiku) attempt the same task with different role prompts. Sampling continues until one answer leads by K votes. This is **not** majority voting — it's an adaptive stopping rule.

- K=3 was sufficient for 1,048,575 sequential steps with zero errors (MAKER paper)
- Red-flagging discards structurally suspicious responses before they pollute votes
- Cost scales log-linearly: `O(s ln s)`, not exponentially

### Role Diversity

Workers get different perspectives via role prompts:
- Pragmatist, Skeptic, Innovator, Systems Thinker, Historian, Contrarian

This gives 15-25% quality improvement vs temperature variation (~2%). The prompt is what prevents mode collapse, not randomness.

### Model Separation

| Role | Model | Why |
|------|-------|-----|
| Planner | Sonnet | Intelligence to decompose well |
| Worker | Haiku | Cheap — voting makes it reliable |
| Composer | Sonnet | Coherent synthesis |
| Verifier | Sonnet | Judgment to catch issues |

## Principles

1. **Cheap models + voting > expensive models solo**
2. **Decompose before you swarm** — vote per step, not per task
3. **Red-flag before counting** — discard bad responses, don't try to fix them
4. **Separate planning from execution** — different models for different jobs
5. **Memory compounds** — one agent with learnings beats 10 without
6. **Diversity via roles, not temperature**

## Installation

```bash
git clone https://github.com/yourname/swarm.git
cd swarm
```

Requirements:
- Python 3.10+
- [Claude Code CLI](https://claude.ai/claude-code) (`claude` command must be in PATH)
- No pip dependencies

## Data Storage

All data is stored in `~/.swarm/`:
- `learnings.jsonl` — Accumulated learnings (append-only JSONL)
- `sessions.jsonl` — Resumable worker sessions

This directory is created automatically on first run.

## MAKER Paper Adaptation

This is a **MAKER-inspired** system adapted for open-ended reasoning tasks. Key differences from the original paper:

- **Agreement checking**: Uses LLM-based semantic consensus instead of exact string matching (necessary because open-ended tasks have no deterministic ground truth)
- **Verification**: Sonnet verifier + retry loop instead of rule-based correctness checking
- **Decomposition**: Tasks are broken into reasoning steps, not mechanical moves

The core principles (decompose, vote with K-ahead, red-flag, learn) are preserved. See [the paper](https://arxiv.org/abs/2511.09030) for the original formulation.

## License

MIT
