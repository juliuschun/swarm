# Swarm

Three experts are better than one. Thats basically the whole idea.

You know that moment when you're about to pick JWT over sessions, or go microservices when a monolith would've been fine? You ask Claude and get a solid answer. One perspective though. One set of assumptions. One blind spot you wont discover until something breaks in production three weeks later.

Swarm fixes that.

```
/swarm "JWT vs sessions for auth in a distributed system?"
```

Three agents think about your question at the same time. A pragmatist looking for the simplest path. A skeptic hunting for what breaks. A systems thinker tracing what happens six months down the line. They run in parallel and come back with their own take.

If they agree, you know you're on solid ground. If they disagree, thats actually where the good stuff is. The trade-offs nobody mentioned. The edge case the pragmatist glossed over that the skeptic caught.

## Install

```bash
claude plugin marketplace add juliuschun/swarm
claude plugin install swarm@swarm-marketplace
```

Then restart Claude Code. Thats it.

## Why bother

Every senior dev has the story. Made a decision, shipped it, found the edge case that a five-minute conversation would've surfaced on day one.

Code reviews exist because two sets of eyes beat one. Architecture boards exist because irreversible decisions deserve scrutiny. Swarm brings that same idea to your AI workflow.

One agent gives you an answer. Three give you a decision.

## What comes back

```
Answer:       The synthesized recommendation
Agreed on:    Points all three landed on (high confidence)
Differed on:  Trade-offs and the minority insight you'd have missed
Confidence:   How much they aligned
Objectives:   Concrete next steps, ready to execute
```

The disagreements are usually the best part. When the skeptic flags something the pragmatist skipped, thats the blind spot you were about to ship with.

But swarm doesnt stop at opinions. It slices the recommendation into concrete objectives — specific, ordered, actionable — and then your main agent starts working through them. After execution, a reviewer agent checks the work against the original question. If something's off, it gets fixed before you move on.

Opinions without action are just talk. Swarm gives you the plan, carries it out, and checks its own work.

## Real examples

Security review — blind spots become breaches. Three angles catch what one misses.
```
/swarm "Review the auth flow in src/auth/ for vulnerabilities"
```

Architecture trade-off — hard to undo, expensive to get wrong.
```
/swarm "Microservices vs monolith for a 3-person team building a marketplace?"
```

Weird bugs — sometimes you need multiple investigative angles. One agent checks the cache, another looks at race conditions, third one digs into the data layer.
```
/swarm "API returns 200 but data is stale intermittently. Redis cache + PostgreSQL. What's wrong?"
```

Design decisions — when theres no single right answer.
```
/swarm "Best approach for real-time updates — WebSockets, SSE, or polling?"
```

### When not to bother

- Writing code to a clear spec. Just write it.
- Simple factual questions. Just answer.
- Creative work. Voting averages taste into mediocrity.

If the cost of being wrong is 10x the cost of asking, swarm it. Otherwise dont.

## It learns

Swarm saves insights between runs to `~/.swarm/learnings.jsonl`. Next time you ask something related, those learnings show up in the agent prompts automatically.

Stuff that keeps proving useful floats to the top. When a recalled learning contributes to a good result, its score goes up. Stale insights naturally sink. An agent with 30 confirmed learnings does better work than ten agents starting blank.

## The three roles

| Agent | What they look for | Why it helps |
|-------|-------------------|--------------|
| **Pragmatist** | Simplest working path | Cuts through over-engineering |
| **Skeptic** | Failure modes and edge cases | Catches what optimists miss |
| **Systems Thinker** | Second-order effects | Sees downstream consequences |

For harder questions add two more: an **Innovator** for unconventional approaches and a **Contrarian** who argues against the obvious choice.

## How its built

One skill file. Thats it. `skills/swarm/SKILL.md` tells Claude Code to:

1. Recall past learnings that might be relevant
2. Spawn 3 agents in parallel with different role prompts
3. Drop any bad responses (empty, refusals, low confidence)
4. Merge or synthesize depending on consensus
5. Slice the answer into concrete, ordered objectives
6. Execute those objectives
7. Verify the work with a reviewer agent
8. Confirm useful learnings, save new ones

Claude Code's Task tool handles the parallelism natively. No subprocess management, no orchestration framework, no extra moving parts.

## The backstory

We started with a way more complex approach. Built the full [MAKER pipeline](https://arxiv.org/abs/2511.09030) from a paper that achieved zero errors across a million sequential Towers of Hanoi steps. Decompose the task, vote on each step with multiple workers, judge the responses, compose everything, verify, learn, loop.

It was thorough. Also 28x slower than just asking Opus once. For the same quality on real-world questions.

Turns out that paper worked because Towers of Hanoi has one right answer per move. Architecture decisions dont. You cant vote on the right database the way you vote on the right chess move.

The piece that actually mattered was the diversity. Three agents with different prompts genuinely catch different things. So we kept that and dropped everything else.

## License

MIT
