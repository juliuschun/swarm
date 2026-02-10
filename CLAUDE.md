# Swarm — Developer Reference

Claude Code skill that gets diverse expert opinions by running parallel agents with different perspectives. One skill file, no runtime dependencies.

## Whats in the repo

```
skills/swarm/SKILL.md    The skill. All logic lives here.
CLAUDE.md                This file.
README.md                User docs.
RESEARCH.md              Why this approach works (and what we tried before).
ROADMAP.md               Whats next.
legacy/                  Old Python CLI (gitignored, kept for reference).
```

## How it works

```
/swarm "your question"
  → Check ~/.swarm/learnings.jsonl for relevant past insights
  → Spawn 3 agents in parallel (pragmatist, skeptic, systems thinker)
  → Compare their responses
  → If they agree: merge, keep each one's best insight
  → If they disagree: synthesize, highlight the trade-offs
  → Save anything worth remembering
  → Show the user one structured answer
```

All of this runs through Claude Code's Task tool. No Python, no subprocess, nothing external.

## The skill file

`skills/swarm/SKILL.md` is the whole product. Markdown with YAML frontmatter. Claude Code loads it when someone types `/swarm`. Inside:

- Role prompts for each agent
- Memory instructions (read/write learnings.jsonl)
- Consensus logic (merge vs synthesize)
- Output structure (answer, agreements, disagreements, confidence)

### Agents

| Role | Looks for | Catches |
|------|-----------|---------|
| Pragmatist | Simplest working path | Over-engineering, unnecessary complexity |
| Skeptic | Failure modes, edge cases | Optimism bias, missed error handling |
| Systems Thinker | Second-order effects | Downstream consequences, scaling traps |

Add Innovator and Contrarian for harder questions (5 agents total).

### Models

Workers inherit whatever model the Task tool uses (usually Sonnet). The host Claude Code instance handles synthesis — same context window, no information lost in handoff.

## Memory

Append-only JSONL at `~/.swarm/learnings.jsonl`:

```json
{"id":"abc123","ts":"2026-02-10T14:30:00Z","category":"strategy","tags":["auth"],"content":"Always separate token generation from validation","confidence":0.7,"times_confirmed":3,"active":true}
```

**Recall**: sorted by `times_confirmed` desc, then `confidence` desc. Top 5 active entries go into agent prompts. Useful stuff rises. Stale stuff sinks.

**Categories**: `mistake` (dont repeat this), `strategy` (this worked), `pattern` (keep seeing this), `constraint` (hard limit to respect).

## Principles

1. **Roles beat temperature.** Pragmatist/skeptic/systems thinker produce genuinely different analyses. Temperature adds noise to the same viewpoint.

2. **Disagreement is the feature.** If all three agree easily the question didnt need swarm. The interesting output is where they differ.

3. **Memory compounds.** An agent with 30 confirmed learnings beats ten fresh agents. Every time.

4. **Opinions not truth.** Always present as "heres what multiple experts concluded." Never as the answer.

## When to trigger

- Architecture decisions (hard to undo)
- Security reviews (blind spots = breaches)
- Trade-offs (JWT vs sessions, REST vs GraphQL)
- Weird debugging (multiple investigative angles)
- User says "ask the swarm" or "get opinions"

## When not to trigger

- Clear spec, just implement it
- Simple factual question
- Creative work needing one coherent vision
- Tests or compiler would give faster feedback

## Legacy code

`legacy/` has the original Python CLI with the full MAKER pipeline. Decompose, vote, judge, compose, verify, learn. We benchmarked it against single Opus and got equivalent quality at 28x latency. Diverse perspectives were the valuable part. The pipeline was overhead.

Run it: `cd legacy && uv run swarm --help`
