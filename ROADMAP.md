# Roadmap

## Works right now

- `/swarm "question"` spawns 3 parallel agents with different role prompts
- Pragmatist, skeptic, and systems thinker on every question
- Consensus check. If they agree, merge. If they disagree, synthesize the best parts.
- Memory across runs via `~/.swarm/learnings.jsonl`
- Confirmed learnings rise, stale ones sink
- Scale to 5 agents for harder questions
- Zero dependencies. Pure Claude Code skill.

## Whats next

**Smarter role matching.** Right now every question gets the same three roles. But a security review should probably get a red-teamer and a compliance person. A database question should get a DBA perspective. Match roles to the domain automatically.

**Confidence tracking.** Do high-confidence answers actually turn out better? Each agent self-reports confidence right now. With enough runs we could check that against which learnings get confirmed. Havent done that yet.

**Learning quality at scale.** At 200+ learnings, does recall actually help? The system assumes it does because confirmed learnings get priority. Needs real measurement.

**Better cross-session context.** When you ask about auth today and auth again next week, agents only see stored learnings. They could see a summary of the full previous discussion instead. Deeper continuity.

**Custom roles.** Let people define their own role prompts. An embedded systems team might want "safety engineer" and "real-time specialist" instead of "systems thinker."

## Not building

**Full MAKER pipeline.** Built it. Benchmarked it. Same quality as one Opus call at 28x the latency for open-ended questions. The perspectives matter. The pipeline doesnt. Story in RESEARCH.md.

**Code execution.** Swarm gives you opinions and analysis. Actually writing the code is your job or your main Claude agent's job. Mixing execution into the opinion process muddies both.

**Adversarial agents.** A dedicated "break the answer" agent sounds useful. In practice it makes everything defensive and hedged. Answers survive attack by being bland. The skeptic role already catches failure modes without that dynamic.

**More agents by default.** Spawning 7 agents for every question sounds thorough. Diminishing returns hit fast. Three solid perspectives cover most questions. Five covers hard ones. Beyond that you're paying more for less.
