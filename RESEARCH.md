# Learnings

## Paper: "Solving a Million-Step LLM Task with Zero Errors"
**Meyerson, E. et al. (2025), arXiv:2511.09030**

---

### The Problem

An LLM with 99% per-step accuracy across 1,000,000 steps: `0.99^1,000,000 ≈ 0`. Guaranteed failure. Errors compound — every long-horizon LLM task eventually collapses.

### The Solution: MAKER

**M**aximal **A**gentic decomposition, First-to-ahead-by-**K** **E**rror correction, **R**ed-flagging.

Three pillars. That's it.

---

### Core Insight #1: Maximal Decomposition

Break the task into the **smallest possible atomic subtasks**. Each micro-agent handles ONE decision. Not "solve the problem" — "make this ONE move."

They solved 20-disk Towers of Hanoi: **1,048,575 sequential dependent moves, zero errors.**

The key: they gave the algorithm/strategy upfront in the prompt. The agent's job isn't to figure out the strategy — it's to execute one step of a known strategy correctly. **Separate planning from execution.**

### Core Insight #2: First-to-ahead-by-K Voting

At each step, multiple agents attempt the same micro-task. Sampling continues until one answer gets **K more votes than any other answer**.

- **K = 3** was sufficient for 1M+ steps
- Most steps converge in ~3 rounds
- Worst case observed: 18 rounds (handled gracefully)

The math:
```
P(correct selection) = 1 / (1 + ((1-p)/p)^k)
```

Cost scales **log-linearly**: `Θ(s ln s)` — not exponentially. You can spend proportionally more to get exponentially more reliability.

**This is NOT simple majority voting.** It's an adaptive stopping rule — you keep sampling until consensus is decisive, not until you hit a fixed N.

### Core Insight #3: Red-Flagging

Certain structural signals predict errors BEFORE you check correctness:
- **Overly long responses**: Error rates jump from ~0.1% to ~10% beyond 700 tokens
- **Format violations**: Responses not matching expected output patterns

When red-flagged: **discard and resample**. Don't try to fix. Don't include in voting. Throw it away.

This reduced correlated errors (where multiple voters are wrong in the same way) from dozens to near-theoretical minimum.

---

### The Numbers

| Metric | Value |
|--------|-------|
| Task | 20-disk Towers of Hanoi |
| Total steps | 1,048,575 (2^20 - 1) |
| Errors | **0** |
| K (voting threshold) | 3 |
| Best model | GPT-4.1-mini (0.22% per-step error, ~$3.5K) |
| Required K growth | Θ(ln s) — logarithmic in step count |
| Cost growth | Θ(s ln s) — log-linear |

**Counter-intuitive finding**: Smaller, non-reasoning models gave the best reliability-per-dollar. Cheap models + voting > expensive models solo.

### The Formula

Minimum K for target success probability `t` over `s` steps:
```
k_min = ⌈ln(t^(-1/s) - 1) / ln((1-p)/p)⌉
```

For `p = 0.99`, `s = 100`, `t = 0.95`: **k = 2** is enough.
For `p = 0.99`, `s = 1,000,000`, `t = 0.95`: **k = 3** is enough.

K grows with ln(s). This is why it's practical.

---

## Applied to Our Swarm (v3 — judge-based architecture)

### What we implemented from MAKER:

1. **Decompose → Vote → Judge → Compose → Verify → Learn loop**
   - Opus decomposes task into atomic steps
   - Sonnet workers generate diverse responses per step
   - Opus judge picks the best response (majority alignment = signal, not law)
   - Opus composes results and verifies
   - Learnings feed back if verification fails

2. **Red-flagging before judging**
   - Discard: empty, too long (>3000 chars), low confidence (<3/10), errors
   - Reduces correlated failures without extra cost

3. **Escalating K (not fixed)**
   - Start with K=2 workers, escalate to K=3, then K=5 if judge wants more
   - Easy steps converge at K=2 (fast). Hard steps get more samples (reliable)
   - No classifier needed — judge naturally drives escalation

4. **Separate planning from execution**
   - Opus plans (decompose) + judges + verifies
   - Sonnet executes (vote steps)

### What we evolved beyond the paper:

5. **Judge replaces consensus**
   - Paper: exact string match for Hanoi moves
   - Us: Opus judge evaluates quality and picks best response
   - Stronger than fuzzy consensus — judge can pick a brilliant minority answer

6. **Per-step verification**
   - Paper: verified after every move (deterministic)
   - Us: Opus verifies each step immediately, retries with feedback if failed
   - Errors caught before cascading — not after all steps complete

7. **State compression + checkpointing**
   - Older steps compressed into summaries, last 5 kept in full
   - Checkpoint every 10 steps for crash recovery
   - Enables 100+ step runs without context overflow

---

## Empirical Findings from Our Testing

### N=3 vs N=6 workers (opinion mode, pre-MAKER)

| Question type | N=3 | N=6 | Winner |
|---------------|-----|-----|--------|
| Convergent ("what makes software reliable?") | Consensus, 21s | Consensus, 20s | Tie |
| Divisive ("rewrite vs refactor?") | Consensus, 46s | Consensus, 49s | Tie |

**Finding**: For well-trodden topics, Sonnet has strong priors — even 6 diverse roles converge on the same answer. Extra workers don't help when the model already "knows" the answer.

**When more workers help**: Novel/ambiguous problems, code-level decisions requiring tool use, creative tasks needing surface area.

### Roles > Temperature

Using distinct roles (pragmatist, skeptic, innovator, systems thinker, historian, contrarian) produces 15-25% quality improvement vs temperature variation alone. Mode collapse is the #1 killer of ensemble methods.

---

## One-Line Takeaway

> Reliability is not a property of the model — it's a property of the process. Small models + decomposition + adaptive voting + red-flagging = zero errors at million-step scale.
