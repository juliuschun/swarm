---
name: swarm
description: Three expert perspectives on your hardest questions. Use when mistakes are expensive.
user-invocable: true
allowed-tools: Task, Read, Write, Glob, Grep
argument-hint: "your question or task"
---

# Swarm — Three Experts, One Answer

Spawn three agents with different perspectives. Check if they agree. Synthesize the best answer when they dont.

## When to use

- Architecture decisions that are hard to reverse
- Security reviews where blind spots mean breaches
- Trade-off analysis (JWT vs sessions, REST vs GraphQL, monolith vs microservices)
- Debugging non-deterministic issues from multiple angles
- User says "ask the swarm", "get opinions", "what do multiple agents think"

## When NOT to use

- Routine implementation. Just write the code.
- Tasks with clear specs. Just follow them.
- Simple factual questions. Just answer.
- Creative work that needs a single coherent vision.

## How it works

### Step 1: Recall learnings

Read `~/.swarm/learnings.jsonl` if it exists. Each line is JSON with `{id, category, content, confidence, times_confirmed, active}`. Sort by `times_confirmed` descending, take top 5 active entries. Format them as context for the agents.

### Step 2: Spawn 3 agents in parallel

Use the Task tool to spawn 3 agents simultaneously. Each gets the same question but a different perspective. All three run at the same time.

**Agent 1 — Pragmatist**
```
You are a pragmatist. Find the simplest working solution. Cut through complexity. What's the most practical path that actually ships?

QUESTION: $ARGUMENTS

[Include any recalled learnings here]

Give a thorough, actionable answer. At the end, rate your confidence: CONFIDENCE: X/10
```

**Agent 2 — Skeptic**
```
You are a skeptic. Find what could go wrong. Identify edge cases, failure modes, and assumptions that might not hold. What is everyone else missing?

QUESTION: $ARGUMENTS

[Include any recalled learnings here]

Give a thorough, actionable answer. At the end, rate your confidence: CONFIDENCE: X/10
```

**Agent 3 — Systems Thinker**
```
You are a systems thinker. Consider second-order effects, feedback loops, and long-term consequences. What happens six months from now if we go this route?

QUESTION: $ARGUMENTS

[Include any recalled learnings here]

Give a thorough, actionable answer. At the end, rate your confidence: CONFIDENCE: X/10
```

Use `subagent_type: "general-purpose"` for all three. Launch all 3 in a single message (parallel).

### Step 3: Check consensus

Read all 3 responses. Do they substantially agree on the key recommendations?

- **If they agree**: Merge into one answer, keeping each perspective's best insight. Agreement means high confidence.
- **If they disagree**: Synthesize. Note where they differ and why. A minority insight that others missed can be the most valuable part of the whole response.

### Step 4: Present the result

Show the user a single synthesized answer. Structure it as:

1. **Answer** — the synthesized recommendation
2. **Where they agreed** — high-confidence points all three landed on
3. **Where they differed** — trade-offs, tensions, minority insights worth paying attention to
4. **Confidence** — overall confidence based on how much they aligned

### Step 5: Save learnings

If the answer contains a reusable insight, append to `~/.swarm/learnings.jsonl`:

```json
{"id":"<random-12-hex>","ts":"<ISO-8601>","category":"strategy","tags":[],"content":"<the insight>","confidence":0.7,"times_confirmed":0,"active":true}
```

Categories: `mistake`, `strategy`, `pattern`, `constraint`.

Only save genuine insights, not the full answer. One learning per run is plenty. If nothing is worth saving, dont save anything.

## Scaling up

- For harder questions, spawn 5 agents instead of 3. Add an **Innovator** (unconventional approaches nobody else would try) and a **Contrarian** (argues against the obvious choice to stress-test it).
- For code questions, use `subagent_type: "general-purpose"` so agents can read files and explore the codebase.

## Important

- Present results as "heres what multiple experts concluded." Not absolute truth.
- The value is in the diversity. Not in any single agent's answer.
- If all agents agree easily, the question probably didnt need swarm.
- Disagreements are features. Highlight them.
