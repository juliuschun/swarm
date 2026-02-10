# Research

## The problem with asking one person

You ask someone a question and you get their answer. Shaped by what they know, what they've seen, what they havent seen. Might be great. You wont know what it missed until something breaks.

This isnt a new observation. Code reviews, architecture boards, medical tumor boards, scientific peer review. They all exist for the same reason: multiple independent perspectives catch stuff any single expert misses. The question was whether AI agents work the same way.

They do.

## The paper that started this

### MAKER: Zero errors at one million steps

**"Solving a Million-Step LLM Task with Zero Errors"** — Meyerson et al., 2025 ([arXiv:2511.09030](https://arxiv.org/abs/2511.09030))

Heres the core math problem. An LLM with 99% per-step accuracy across 1,000,000 steps: `0.99^1,000,000 ≈ 0`. Effectively zero chance of getting everything right. Errors compound and every long-horizon task eventually collapses.

MAKER solved it with three things:

**Decomposition.** Break the task into the smallest possible atomic steps. Each agent handles one decision. Not "solve the problem." One move. Thats it.

**Diverse voting.** Multiple agents try each step independently. Keep sampling until one answer leads by K votes. Not simple majority. An adaptive stopping rule that keeps going until consensus is decisive.

**Red-flagging.** Some structural signals predict errors before you even check correctness. Responses that are too long, too short, or dont match expected formats get thrown out. Dont fix them. Discard and resample.

Result: 20-disk Towers of Hanoi. 1,048,575 sequential dependent moves. Zero errors. Cost scales log-linearly, not exponentially.

### The surprising finding about model size

Smaller cheaper models with voting beat expensive models going solo. GPT-4.1-mini at $3.5K total outperformed larger models on reliability-per-dollar. The reliability came from the process, not from the model being smarter.

That stuck with us. Reliability isnt a property of the model. Its a property of the process around it.

## What we learned when we actually built it

We implemented the full pipeline in Python. Decompose, vote per step with escalating K, Opus judge picking the best response, compose everything together, verify, learn, loop. It worked for deterministic stuff.

But for the questions we actually cared about — architecture calls, security reviews, trade-off analysis — the pipeline produced the same quality as asking Opus once. At 28x the latency.

Why? MAKER was designed for problems where correctness is binary. Did you move the right disk? Yes or no. Open-ended reasoning doesnt have that. You cant vote on "the right architecture" like you vote on "the right chess move."

### What actually worked: the perspectives

The thing that consistently added value was having different agents look at the same problem from different angles. A pragmatist, a skeptic, and a systems thinker dont agree because they're optimizing for different things. Thats the point.

This maps to stuff thats worked for decades in other fields:

**Delphi Method** — RAND Corporation, 1950s. Get structured independent responses from diverse experts across multiple rounds. No groupthink. Everyone sees the same question but answers from their own expertise.

**Tumor boards in oncology** — A surgeon, radiologist, oncologist, and pathologist review the same case. The chair weighs all opinions and makes the call. One brilliant insight from a single specialist can override the majority. Thats exactly how synthesis works in swarm.

**Toyota A3 problem solving** — You cant solve a problem you havent broken down. The discipline of decomposition forces clarity. We kept this principle even after dropping the voting pipeline.

**Kaizen** — Small improvements over time compound into something big. An agent with 30 confirmed learnings from past runs does genuinely better work than ten fresh agents. We see this every time.

### What we dropped and why

- **The voting pipeline.** Overhead without measurable quality gain for open-ended reasoning.
- **Deterministic verification.** Cant verify "good architecture" the way you verify "correct disk move."
- **Escalating K.** Built for million-step reliability. We're doing 3-agent parallel opinions, not million-step chains.
- **Checkpointing and crash recovery.** A 3-agent parallel spawn takes seconds. It doesnt crash mid-run.

## The principle underneath all of it

For deterministic tasks with a clear right answer, use voting and verification. MAKER nailed that.

For open-ended reasoning where judgment matters, use diverse perspectives and synthesis. Thats what swarm does.

Both work because they break the same failure mode: one perspective, one set of blind spots, one way to be wrong. Whether you fix it through voting or through synthesis, the mechanism is different but the principle is identical.

## One takeaway

Three experts disagreeing teaches you more than one expert being thorough.
