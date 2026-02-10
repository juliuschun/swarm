# Swarm: Where We Are and Where We're Going

## What We Built

A second opinion machine. Not a general-purpose AI tool — a specialized system for decisions where being wrong costs more than asking twice.

The core loop: Opus breaks a task into steps. Sonnet workers attempt each step independently with different perspectives (pragmatist, skeptic, innovator, contrarian, etc.). Opus judge picks the best response — using majority alignment as a quality signal, not as law. Opus verifies the composed result. If it fails, the system learns from the failure and tries again.

This is adapted from the MAKER paper (Meyerson et al., 2025), which achieved zero errors across 1,048,575 sequential steps of Towers of Hanoi. We took their core ideas — decompose, vote, red-flag, learn — and adapted them for open-ended reasoning where there's no deterministic ground truth.

## What Actually Works Today

- **3-30 step tasks** with high reliability. Architecture decisions, security reviews, trade-off analysis, debugging.
- **Escalating K** (start with 2 workers, scale to 5 if the judge wants more options). No wasted API calls on easy steps.
- **Opus judge** instead of fuzzy consensus. Eliminates the biggest reliability killer from v1 — asking Haiku "do these agree?"
- **Per-step verification** catches errors immediately instead of after all steps complete.
- **Checkpointing** every 10 steps. A crash at step 50 loses 10 steps, not 50.
- **State compression** keeps context manageable beyond the first 5 steps.
- **Memory** that compounds across runs. Run 10 knows what run 1 learned.
- **Plugin-ready** for Claude Code. `/swarm "your question"` just works.

## The Honest Gap

The MAKER paper's million-step guarantee relied on four things we don't have:

1. **Deterministic correctness** — each Hanoi move is right or wrong. Our tasks are subjective.
2. **Exact string matching** for consensus — we use an Opus judge instead, which is smarter but not mathematically provable.
3. **Immediate state validation** — Hanoi could verify after every move. We verify per-step now (improvement from v1's post-hoc-only check), but it's still LLM-judged, not formally verified.
4. **Formal state tracking** — Hanoi knew the exact position of every disk. We have compressed summaries.

**Realistic scaling ceiling today: ~100-1000 steps** depending on task difficulty. Not a million. The million-step dream requires formal verification and deterministic state — which may be possible for structured tasks (code generation with test suites, data pipelines with schema validation) but not for open-ended reasoning.

## What This Is Actually For

**Use swarm when:**
- You're about to make a decision and think "I should ask a few people about this"
- The cost of being wrong > 10x the cost of asking (auth, payments, migrations, security, architecture)
- You want diverse perspectives on a trade-off
- You're debugging something non-deterministic and need multiple investigative angles

**Don't use swarm when:**
- You'd know it's wrong within 30 seconds (compiler, tests, visual inspection)
- The task needs a single coherent creative vision
- You're doing routine implementation with clear specs

**Swarm for decisions, solo for execution.**

## What's Next (If We Keep Going)

### Quick wins (hours)
- [ ] Upgrade opinion mode to use judge pattern (currently still uses old Haiku fuzzy consensus)
- [ ] Add `--resume-task` CLI flag to trigger checkpoint resume
- [ ] Update docs to reflect current architecture

### Medium effort (days)
- [ ] Formal StepState class with variable tracking and dependency graph (enables 500+ steps)
- [ ] Deterministic verification for structured outputs (JSON schema validation, code that compiles, tests that pass)
- [ ] SQLite-backed memory with semantic search (enables 10K+ learnings)

### The big bet (weeks)
- [ ] Code generation pipeline: decompose → generate per-file → run tests as verification → iterate. This is where MAKER's guarantees actually apply — tests are deterministic ground truth.
- [ ] Integration with the swarm-team Go CLI as a decision oracle: before a lead commits to an approach, swarm evaluates it.

## The Principles We Believe In

1. **An expert judge beats a democratic vote.** Majority alignment is a signal for the judge, not a decision mechanism.
2. **Catch errors early.** Per-step verification beats post-hoc verification every time.
3. **Memory compounds.** One agent with learnings beats ten amnesiac agents.
4. **Diversity via roles, not temperature.** Pragmatist/skeptic/innovator prompts prevent mode collapse. Temperature just adds noise.
5. **Match the tool to the task.** Swarm for high-stakes decisions. Solo Opus for creative vision. Solo Sonnet for routine execution.
6. **Ship the 10-step tool, not the million-step dream.** A tool that reliably handles 10 hard steps is more useful today than a theoretical system that might handle a million.
