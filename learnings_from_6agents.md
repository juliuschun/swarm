# Learnings from the 6-Agent Experiment

> Feb 10, 2026 — What we actually learned, and why we need to rethink what we're building.

---

## The Two Things That Exist

### 1. `teammate/` — Swarm-Team Skill (Go CLI)
PM → Lead → Junior hierarchy. File-based IPC (`/tmp/swarm-team/`). Go binary for fast spawning. Designed for Claude Code's native team features. Has a Tauri companion app for visualization.

**Model**: Command & control. PM decomposes, Leads execute, Juniors code.

### 2. `teammate_revived3/swarm.py` — MAKER Loop (Python)
Single 700-line file implementing the research-backed MAKER pattern:
```
RECALL → DECOMPOSE → VOTE each step (adaptive K-ahead) → COMPOSE → VERIFY → LEARN
  ↑                                                                              |
  └──────────────────── re-plan with learnings if verify fails ──────────────────┘
```

Based on "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025).

**Model**: Flat swarm. Sonnet plans, cheap Haiku workers vote per step, Sonnet verifies, loop until pass.

---

## The Experiment (What the 6 Agents Proved)

**Question**: Does role-diverse parallel thinking beat single-agent thinking?

| Condition | Score | Notes |
|-----------|:-----:|-------|
| 1 solo agent | 26/60 | Over-documented, missed failure modes entirely |
| 3 diverse agents | 43/60 | +65%. Pragmatist, Skeptic, DX Designer |
| 5 diverse agents | 53/60 | +23% more. Added Ops Engineer, Knowledge Manager |

**Verdict**: Yes, measurably. But the mechanism is **role diversity**, not agent count.

### What Each Perspective Uniquely Saw

| Agent | Unique Contribution | No One Else Caught This |
|-------|--------------------|-----------------------|
| **Pragmatist** | 35-line implementation, ship code not docs | Set the "is it this simple?" bar |
| **Skeptic** | SHA256 dedup, TSV alternative, knowledge decay | Only agent to challenge format choice itself |
| **DX Designer** | "Will anyone use this?" filter, demo-first | Usability as design constraint |
| **Ops Engineer** | Self-describing headers, severity-based rotation, failure recovery matrix | "Can we see it working?" (observability) |
| **Knowledge Manager** | "Retrieve BEFORE you start working" | Retrieval-first > storage-first design |

---

## The 8 Validated Principles

These survived scrutiny from all perspectives and are already implemented in `swarm.py`:

### 1. Cheap models + voting > expensive models solo
MAKER proved this at 1M steps. `swarm.py` implements it: Haiku workers with K=3 ahead-by voting. Cost is log-linear, not exponential.

### 2. Decompose before you swarm
Don't send big tasks to parallel agents. Break into atomic steps, vote each step. `swarm.py` does this: Sonnet decomposes → Haiku executes per step.

### 3. Red-flag before counting
Discard structurally suspicious responses (too long, empty, low confidence) before they pollute votes. Already in `swarm.py`: `red_flag()` function.

### 4. Separate planning from execution
Sonnet plans, Haiku executes. Different cognitive loads need different models. Already in `swarm.py`: `MODELS` config.

### 5. Diversity via roles, not temperature
Role prompts give 15-25% improvement. Temperature variation gives ~2%. `swarm.py` has 6 roles (pragmatist, skeptic, innovator, systems thinker, historian, contrarian).

### 6. Memory compounds everything
A single agent with learnings beats 10 amnesiac agents. `swarm.py` implements: `recall()` → inject into prompts → `learn()` from verification output.

### 7. One judge is sufficient
No ensemble-of-judges. Judging the judge is a trap. `swarm.py` uses one Sonnet verifier.

### 8. Design for graceful degradation
Timeouts are data, not bugs. 2 of 5 agents timed out in the experiment. `swarm.py` handles this: red-flag → discard → resample.

---

## The Uncomfortable Truth (Tech-Lead Lens)

### We built two things that solve the same problem differently

| | Swarm-Team (teammate/) | swarm.py (revived3/) |
|---|---|---|
| **Architecture** | Hierarchical (PM → Lead → Junior) | Flat (Orchestrator → Parallel Workers → Verifier) |
| **Communication** | File-based IPC, `/tmp/` | Subprocess stdout, in-memory |
| **Language** | Go + Claude Code native teams | Python + Claude CLI subprocess |
| **Agent model** | Workers doing different jobs | Workers doing the same job differently (roles) |
| **Scaling model** | More agents = more throughput | More agents = more perspectives |
| **Reliability** | Hope each agent does its job | Vote until consensus, verify, retry |
| **Memory** | None (stateless) | JSONL learnings, session resume |
| **Research basis** | Intuition + experience | MAKER paper (proven at 1M steps) |

### Which one is right?

**swarm.py is more right.** Here's why:

1. **It has mathematical backing.** K-ahead voting with red-flagging provably reaches zero errors at million-step scale. The PM → Lead → Junior hierarchy has no such guarantee.

2. **It treats reliability as a process, not a property.** Swarm-team trusts each agent to do its job correctly. `swarm.py` assumes they'll be wrong and votes until consensus.

3. **It compounds.** Every run makes the next run better via learnings. Swarm-team starts from zero every time.

4. **It's simpler.** 700 lines of Python vs. a Go CLI + file IPC + status files + companion app.

### But swarm.py has gaps

1. **It can't do real work.** It runs `claude -p` (non-interactive). Workers can't read files, edit code, run tests. They can only think.

2. **No tool use.** The MAKER paper solved Towers of Hanoi — a pure reasoning task. Real software engineering needs file access, git, shell commands.

3. **No hierarchy for complex projects.** "Design an auth system" works. "Build and deploy an auth system" needs decomposition into tasks that require different tools and scopes.

4. **Workers are isolated.** No worker-to-worker communication. Good for diversity, bad for tasks where context needs to flow between steps.

---

## The Fundamental Rethink

### The real question isn't "which architecture?"

It's: **what layer of the problem does each approach solve?**

```
Layer 3: PROJECT ORCHESTRATION (swarm-team territory)
         "Build auth system" → decompose into Lead-level tasks
         Needs: file access, tool use, long-running sessions, task tracking

Layer 2: DECISION QUALITY (swarm.py territory)
         "How should auth work?" → diverse perspectives → vote → verify
         Needs: parallel workers, role diversity, K-ahead voting, memory

Layer 1: EXECUTION RELIABILITY (MAKER territory)
         "Implement this specific function" → atomic steps → vote each → compose
         Needs: decomposition, red-flagging, adaptive voting
```

**The insight**: These aren't competing. They're different layers.

### What "relying more on agents" actually means

Not more agents. **More agent-backed decisions at every layer.**

| Today | Rethought |
|-------|-----------|
| PM decides task decomposition alone | PM uses swarm.py to vote on decomposition strategy |
| Lead picks an approach and runs with it | Lead uses swarm.py `--mode opinion` to evaluate 3 approaches before committing |
| Junior writes code solo | Junior uses MAKER mode for each function (decompose → vote → verify) |
| No memory across sessions | Every decision's learnings feed the next session |

### The architecture that combines both

```
swarm.py becomes the DECISION ENGINE (Layer 2)
swarm-team becomes the ORCHESTRATION LAYER (Layer 3)
MAKER voting becomes the EXECUTION LAYER (Layer 1)

Flow:
1. User: "Build auth system"
2. swarm-team PM spawns the project structure
3. PM calls swarm.py --mode opinion to VOTE on decomposition
4. Each Lead gets a task + relevant learnings (swarm recall)
5. Lead calls swarm.py --mode maker for each implementation step
6. Results verified, learnings extracted, fed back
7. PM composes final result
```

### What to build next

**Don't build more infrastructure.** Both tools work. Wire them together:

1. **Give swarm-team memory.** The Go CLI should call `swarm.py recall` to inject learnings before spawning leads. One function call.

2. **Give swarm.py tool access.** The workers currently run `claude -p` (think-only). Add `--allow-tools` so workers can read files, run tests, check git. This turns MAKER mode from "think reliably" to "work reliably."

3. **Make swarm.py callable from swarm-team leads.** A Lead should be able to say "I need to decide between JWT and sessions" and run `python swarm.py --mode opinion "JWT vs sessions for this codebase"` to get a diverse-perspective answer before committing.

4. **Stop building the companion app.** The timeline visualization is nice but it's not where the value is. The value is in the decision quality and memory compounding. Ship the CLI, not the UI.

---

## The Scores (Detailed)

| Dimension | Solo | 3 Agents | 5 Agents |
|-----------|:----:|:--------:|:--------:|
| Completeness | 7 | 8 | **10** |
| Simplicity | 5 | **7** | 6 |
| Novel Ideas | 3 | 7 | **9** |
| Failure Handling | 2 | 6 | **10** |
| Practical Usability | 5 | 8 | **9** |
| Retrieval Quality | 4 | 7 | **9** |
| **TOTAL** | **26/60** | **43/60** | **53/60** |

### Cost reality

| Approach | Cost | Quality | When to Use |
|----------|------|---------|-------------|
| 1 Opus, well-prompted | ~$0.035 | Good | 80% of tasks (simple, well-defined) |
| 3 Haiku + Sonnet judge | ~$0.06 | Great | Design decisions, trade-offs |
| 5 Haiku + Sonnet judge | ~$0.10 | Best | Architecture, failure mode analysis |
| MAKER mode (per step) | ~$0.02/step | Reliable | Implementation tasks needing correctness |

**N=3 is the sweet spot** for opinions. MAKER mode (adaptive voting) is the sweet spot for execution.

---

## Open Questions

1. **Can swarm.py workers use tools?** Adding `--allow-tools` to the `claude -p` call would let workers read files, run code. This changes everything — MAKER mode goes from "think correctly" to "work correctly." But tool-using agents are slower and more expensive. Trade-off.

2. **How do learnings transfer between projects?** `~/.swarm/learnings.jsonl` is global. Should it be? Project-specific learnings vs universal learnings.

3. **When does the verify step need tools?** "Is this code correct?" can't be answered by reading — it needs `pytest`. The verifier may need shell access.

4. **Can the system decide its own mode?** Instead of the user picking `--mode maker` vs `--mode opinion`, can a meta-agent look at the task and decide: "this needs MAKER (correctness matters)" vs "this needs opinion (trade-off question)" vs "this is simple, one agent is fine"?

5. **What's the minimal integration?** What's the smallest change to swarm-team that gives it swarm.py's decision quality? Probably: before each Lead starts, call `python swarm.py --mode opinion "How should we approach: {task}"` and feed the result into the Lead's prompt.
