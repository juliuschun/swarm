---
name: swarm
description: Multi-agent MAKER voting for tasks needing verified consensus, design trade-offs, or collective intelligence. Use when the user wants multiple agents to collaborate on an answer.
user-invocable: true
allowed-tools: Bash
argument-hint: "your question or task"
---

# Swarm — MAKER Multi-Agent Voting

Run the swarm CLI to get collective intelligence from multiple Claude agents.

## Pick the right mode

**Opinion mode** — fast, 3 parallel agents + consensus. Good for design questions and trade-offs:
```bash
uv run swarm --mode opinion "$ARGUMENTS" -v
```

**MAKER mode** — decompose, vote each step, compose, verify. Good for tasks where correctness matters:
```bash
uv run swarm --mode maker "$ARGUMENTS" -v
```

## When to add flags

- User wants agents to read code: add `--tools`
- User wants agents to edit files: add `--tools-rw`
- User wants JSON output: add `--json`
- User specifies tags: add `--tags tag1,tag2`
- User wants cheaper/faster: mode is already opinion by default for questions
- User wants more reliable: use `--mode maker`

## Examples

```bash
# Quick opinion on a design question
uv run swarm --mode opinion "JWT vs sessions for auth?" -v

# Reliable answer for a coding task
uv run swarm --mode maker --tools "Review the error handling in src/" -v

# Get past learnings
uv run swarm --recall --tags coding

# JSON output for piping
uv run swarm --mode opinion --json "Best caching strategy?"
```

## Important

- Always run from the project root (where pyproject.toml is)
- The `uv run` command auto-installs on first use — no setup needed
- Show the user the final answer, not the raw command output
- If the user says "ask the swarm" or "get consensus", use this skill
