---
name: swarm
description: Three expert perspectives on your hardest questions. Use when mistakes are expensive.
user-invocable: true
allowed-tools: Task, Read, Write, Glob, Grep
argument-hint: "your question or task"
---

# Swarm — Three Experts, One Answer

Spawn three agents with different perspectives. Check if they agree. Synthesize the best answer when they dont. Break it into objectives. Execute. Verify.

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

Read `~/.swarm/learnings.jsonl` if it exists. Each line is JSON with `{id, category, content, confidence, times_confirmed, active}`. Sort by `times_confirmed` descending, take top 5 active entries. Keep track of which learning IDs were recalled — you'll need this later.

Format them as context for the agents.

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

### Step 3: Sanity check responses

Before using any agent response, quick scan:
- Empty or trivially short? Drop it.
- Refusal pattern ("I cannot", "I'm unable to")? Drop it.
- Self-reported confidence below 3/10? Drop it.

Two good perspectives beat three where one is junk. Dont retry — work with what you have.

### Step 4: Check consensus

Read the usable responses. Do they substantially agree on the key recommendations?

- **If they agree**: Merge into one answer, keeping each perspective's best insight. Agreement means high confidence.
- **If they disagree**: Synthesize. Note where they differ and why. A minority insight that others missed can be the most valuable part of the whole response.

### Step 5: Slice into objectives

Take the synthesized answer and break it into concrete, ordered next steps. These should be specific enough to execute immediately.

Good objectives:
- "Add rate limiting middleware to the auth endpoints in src/auth/routes.ts"
- "Write integration tests covering the token refresh edge case"
- "Refactor the cache layer to use write-through instead of write-behind"

Bad objectives:
- "Consider security implications" (too vague)
- "Improve the architecture" (not actionable)
- "Think about edge cases" (thats analysis, not a mission)

Each objective should be one clear thing to do. If its too big, split it. If the agents disagreed on approach, pick the strongest reasoning and note the trade-off.

Order by dependency — things that need to happen first go first.

### Step 6: Present the result

Show the user:

1. **Answer** — the synthesized recommendation
2. **Where they agreed** — high-confidence points all three landed on
3. **Where they differed** — trade-offs, tensions, minority insights worth paying attention to
4. **Confidence** — overall confidence based on how much they aligned
5. **Objectives** — numbered list of concrete next steps, ready to execute

### Step 7: Execute objectives

Proceed to work through the objectives in order. Dont stop at the opinion. Start doing.

- Work through objectives sequentially (earlier ones may inform later ones)
- If an objective requires user input or a decision, ask before proceeding
- If an objective turns out to be unnecessary based on what you find, skip it and explain why
- Mark each objective as you complete it so the user can track progress

### Step 8: Verify

After completing the objectives, spawn a Task agent to review the work:

```
You are a reviewer. Look at what was just done and check:

1. Do the changes actually address the original question?
2. Are there obvious issues, bugs, or gaps?
3. Did anything get missed from the original plan?
4. Is there anything that should be rolled back or revised?

ORIGINAL QUESTION: $ARGUMENTS
OBJECTIVES COMPLETED: [list what was done]
FILES CHANGED: [list files modified]

Be specific. If something needs fixing, say exactly what and where.
```

Use `subagent_type: "general-purpose"` so the reviewer can read the actual files.

- **If the reviewer finds issues**: Fix them. Work through the feedback, then move on.
- **If the reviewer finds nothing**: Done. Move to saving learnings.

Dont loop more than once. If the fix creates new issues, flag them for the user rather than spiraling.

### Step 9: Save learnings

Two things happen here.

**Confirm recalled learnings**: If learnings from Step 1 were recalled and the result was good, increment `times_confirmed` for each one. Read `~/.swarm/learnings.jsonl`, find the entries by ID, update the count, write the file back. This is how useful learnings rise over time.

**Save new insights**: If the answer or execution revealed something reusable, append to `~/.swarm/learnings.jsonl`:

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
