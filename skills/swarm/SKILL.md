---
name: swarm
description: Second opinion machine for high-stakes decisions. Use when mistakes are expensive (auth, payments, migrations, security, architecture) or when the user wants trade-off analysis. Do NOT use for routine implementation, styling, naming, or tasks where compiler/tests give instant feedback.
user-invocable: true
allowed-tools: Bash
argument-hint: "your question or task"
---

# Swarm — Verification Where Mistakes Are Expensive

A multi-agent voting system. NOT for everything — only when the cost of being wrong exceeds the cost of asking.

## When to use this skill

**USE swarm when:**
- Auth, payment, crypto, security decisions (blind spots = breaches)
- Architecture choices that are hard to reverse (DB schema, API contracts, framework picks)
- Trade-off analysis where diverse perspectives matter (JWT vs sessions, REST vs GraphQL)
- Code review for security-critical paths
- Debugging non-deterministic production issues
- The user says "ask the swarm", "get consensus", "what do multiple agents think"

**DO NOT use swarm when:**
- Routine implementation (just write the code)
- CSS, naming, formatting, boilerplate (instant feedback via compiler/UI)
- Tasks with clear docs to follow (just follow the docs)
- Creative work needing coherent vision (voting averages taste into mediocrity)
- Simple factual questions (just answer them)

**Rule of thumb:** If the cost of being wrong > 10x the cost of voting, swarm it. Otherwise, solo.

## Pick the right mode

**Opinion mode** — fast, 3 parallel agents + consensus. For trade-off questions:
```bash
uv run swarm --mode opinion "$ARGUMENTS" -v
```

**MAKER mode** — decompose, vote each step, compose, verify. For tasks where correctness matters:
```bash
uv run swarm --mode maker "$ARGUMENTS" -v
```

## Mode selection guide

| Task type | Mode | Why |
|---|---|---|
| "X vs Y?" trade-off | opinion | Fast, surfaces perspectives |
| "Is this secure?" review | maker + --tools | Decompose into checks, vote each |
| "Design the schema for..." | opinion | Architecture needs perspectives, not steps |
| "Implement X per spec" | maker + --tools | Step-by-step with verification |
| "What's wrong with this code?" | maker + --tools | Multiple investigative angles |

## Flags

- `--tools` — workers can read code (Read, Glob, Grep, Bash)
- `--tools-rw` — workers can edit files (use with caution)
- `--json` — JSON output for piping
- `--tags tag1,tag2` — memory tags for recall/learn
- `--worker-model sonnet` — upgrade workers for harder tasks (default: haiku)
- `--max-cost 0.50` — cost ceiling in USD

## Examples

```bash
# Security review (high stakes, use swarm)
uv run swarm --mode maker --tools "Review auth flow in src/auth/ for vulnerabilities" -v

# Architecture trade-off (need perspectives)
uv run swarm --mode opinion "Microservices vs monolith for a 3-person team?" -v

# Hard debugging (multiple investigative angles)
uv run swarm --mode maker --tools "Why does the webhook handler fail silently in production?" -v

# Get past learnings
uv run swarm --recall --tags security
```

## Important

- Always run from the project root (where pyproject.toml is)
- The `uv run` command auto-installs on first use — no setup needed
- Show the user the final answer, not the raw command output
- Present swarm results as "here's what multiple agents concluded" — not as absolute truth
