"""MAKER loop: decompose, vote, compose, verify, learn."""

import asyncio
import hashlib
import json
import re
import sys
import time

from .agent import claude, red_flag
from .config import (
    BATCH_SIZE, CHECKPOINT_INTERVAL, CHECKPOINTS_DIR, K_MAX, K_START,
    MAX_LOOPS, MAX_SAMPLES, MODELS, ROLES, TIMEOUT_SECONDS,
)
from .hooks import run_hooks
from .memory import format_learnings, learn, recall, save_session
from .opinion import run_opinion


# ── Checkpointing ──────────────────────────────────────────────────────

def _task_hash(task: str) -> str:
    return hashlib.sha256(task.encode()).hexdigest()[:12]


def _save_checkpoint(task: str, step_num: int, step_results: list[dict],
                     total_cost: float, loop: int):
    """Save progress every N steps for crash recovery."""
    d = CHECKPOINTS_DIR / _task_hash(task)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"step_{step_num:06d}.json").write_text(json.dumps({
        "step_num": step_num,
        "step_results": step_results,
        "total_cost": total_cost,
        "loop": loop,
    }))


def _load_checkpoint(task: str) -> dict | None:
    """Resume from most recent checkpoint."""
    d = CHECKPOINTS_DIR / _task_hash(task)
    if not d.exists():
        return None
    checkpoints = sorted(d.glob("step_*.json"), reverse=True)
    if not checkpoints:
        return None
    return json.loads(checkpoints[0].read_text())


def _clear_checkpoints(task: str):
    """Remove checkpoints after successful completion."""
    d = CHECKPOINTS_DIR / _task_hash(task)
    if d.exists():
        for f in d.glob("step_*.json"):
            f.unlink()
        try:
            d.rmdir()
        except OSError:
            pass


# ── State Compression ──────────────────────────────────────────────────

def _build_context(base_context: str, step_results: list[dict],
                   compressed_history: list[str]) -> str:
    """Build context with compressed history + recent full results."""
    context = base_context

    # Compressed history of older steps
    if compressed_history:
        context += "\n\nPrior work (summarized):\n" + "\n".join(compressed_history)

    # Last 5 steps in full detail
    if step_results:
        recent = step_results[-5:]
        prior = "\n".join(f"Step {r['step_num']}: {r['answer'][:500]}"
                          for r in recent)
        context += f"\n\nRecent steps (full detail):\n{prior}"

    return context


async def _compress_steps(step_results: list[dict], start: int, end: int,
                          timeout: float) -> tuple[str, float]:
    """Compress a chunk of step results into a short summary."""
    chunk = "\n".join(
        f"Step {r['step_num']}: {r['answer'][:300]}"
        for r in step_results[start:end]
    )
    result = await claude(
        f"Summarize these step results in 2-3 sentences. Keep key decisions and outputs:\n\n{chunk}",
        model="haiku", timeout=timeout
    )
    return result.get("content", chunk[:200]), result.get("cost", 0)


# ── Phase 1: DECOMPOSE ────────────────────────────────────────────────

async def decompose(task: str, learnings_text: str, verbose: bool = False,
                    timeout: float = TIMEOUT_SECONDS) -> tuple[list[str], float]:
    """One smart agent breaks the task into atomic steps. Returns (steps, cost)."""
    system = f"""You are a task decomposer. Break the given task into a numbered list of small, atomic steps.

Rules:
- Each step should be independently completable
- Each step should have ONE clear deliverable
- Order steps by dependency (earlier steps feed later ones)
- Aim for 3-8 steps (fewer for simple tasks, more for complex)
- Each step description should be 1-2 sentences, specific and actionable

{learnings_text}

Respond with ONLY a JSON array of step strings. Example:
["Step 1: Define the data model with fields X, Y, Z", "Step 2: Implement the API endpoint", ...]"""

    result = await claude(task, model=MODELS["planner"], system=system, timeout=timeout)
    cost = result.get("cost", 0)
    if result["error"]:
        if verbose:
            print(f"  [decompose] ERROR: {result['error']}", file=sys.stderr)
        print("[decompose] WARNING: Fallback to single-step mode (no decomposition benefit)", file=sys.stderr)
        return [task], cost

    content = result.get("content", "")
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if json_match:
        try:
            steps = json.loads(json_match.group())
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return steps, cost
        except json.JSONDecodeError:
            pass

    lines = [re.sub(r'^\d+[\.\)]\s*', '', l.strip())
             for l in content.splitlines() if re.match(r'^\d+[\.\)]', l.strip())]
    if not lines:
        print("[decompose] WARNING: Fallback to single-step mode (no decomposition benefit)", file=sys.stderr)
    return (lines if lines else [task]), cost


# ── Phase 2: VOTE per step (escalating K + judge) ─────────────────────

async def vote_step(
    step: str,
    step_num: int,
    total_steps: int,
    context: str,
    verbose: bool = False,
    tools: bool = False,
    tools_rw: bool = False,
    cwd: str | None = None,
    worker_model: str = "sonnet",
    timeout: float = TIMEOUT_SECONDS,
) -> dict:
    """Escalating K voting with Opus judge selection.

    Instead of fuzzy consensus checking:
    1. Generate diverse responses with escalating batches
    2. Opus judge picks the best one, using majority alignment as a signal
    """
    batch_size = BATCH_SIZE
    if tools_rw:
        batch_size = 1
        print("[vote] Sequential mode: --tools-rw forces batch_size=1 to prevent concurrent writes", file=sys.stderr)

    all_responses = []
    flagged_count = 0
    flagged_reasons = {}
    total_sampled = 0
    round_num = 0
    total_cost = 0

    step_prompt = f"""You are executing step {step_num}/{total_steps} of a larger task.

STEP: {step}

{context}

Complete ONLY this step. Be specific and concrete.
At the end, rate your confidence: CONFIDENCE: X/10"""

    # Escalating rounds: K_START responses, then K_START+BATCH_SIZE, up to K_MAX
    current_k = K_START

    while total_sampled < MAX_SAMPLES:
        round_num += 1
        batch_count = min(batch_size, MAX_SAMPLES - total_sampled)
        tasks = []
        for i in range(batch_count):
            role = ROLES[(total_sampled + i) % len(ROLES)]
            tasks.append(claude(step_prompt, model=worker_model, system=role,
                               tools=tools, tools_rw=tools_rw, cwd=cwd, timeout=timeout))

        results = await asyncio.gather(*tasks)
        total_sampled += batch_count

        for r in results:
            total_cost += r.get("cost", 0)
            flag = red_flag(r)
            if flag:
                flagged_count += 1
                flagged_reasons[flag] = flagged_reasons.get(flag, 0) + 1
                if verbose:
                    print(f"  [vote step {step_num}] RED-FLAG: {flag}", file=sys.stderr)
            else:
                all_responses.append(r)

        if verbose:
            print(f"  [vote step {step_num}] round {round_num}: "
                  f"{len(all_responses)} valid, {flagged_count} flagged, "
                  f"{total_sampled} total sampled", file=sys.stderr)

        # Once we have enough responses, send to judge
        if len(all_responses) >= current_k:
            best, judge_cost = await _judge_responses(all_responses, step, step_num, verbose, timeout)
            total_cost += judge_cost

            if best is not None:
                if verbose:
                    print(f"  [vote step {step_num}] JUDGE PICKED after {total_sampled} samples", file=sys.stderr)
                return {
                    "step": step,
                    "step_num": step_num,
                    "answer": best.get("content", ""),
                    "agree_count": len(all_responses),
                    "total_sampled": total_sampled,
                    "flagged": flagged_count,
                    "flagged_reasons": flagged_reasons,
                    "rounds": round_num,
                    "session_ids": [r.get("session_id") for r in all_responses if r.get("session_id")],
                    "cost": total_cost,
                }

            # Judge said none are good enough — escalate K
            current_k = min(current_k + BATCH_SIZE, K_MAX)
            if verbose:
                print(f"  [vote step {step_num}] Judge wants more options, escalating to K={current_k}", file=sys.stderr)

    # Exhausted MAX_SAMPLES — judge picks from what we have
    if verbose:
        print(f"  [vote step {step_num}] MAX SAMPLES reached. Judge picks from {len(all_responses)} responses.", file=sys.stderr)

    if all_responses:
        best, judge_cost = await _judge_responses(all_responses, step, step_num, verbose, timeout, force=True)
        total_cost += judge_cost
        answer = best.get("content", "") if best else all_responses[0].get("content", "")
    else:
        answer = ""

    return {
        "step": step,
        "step_num": step_num,
        "answer": answer,
        "agree_count": 1,
        "total_sampled": total_sampled,
        "flagged": flagged_count,
        "flagged_reasons": flagged_reasons,
        "rounds": round_num,
        "session_ids": [r.get("session_id") for r in all_responses if r.get("session_id")],
        "cost": total_cost,
    }


async def _judge_responses(responses: list[dict], step: str, step_num: int,
                           verbose: bool, timeout: float,
                           force: bool = False) -> tuple[dict | None, float]:
    """Opus judge evaluates all responses and picks the best one.

    Majority alignment is a signal, not a requirement.
    Returns (best_response, cost) or (None, cost) if judge wants more options.
    """
    responses_text = "\n\n---\n\n".join(
        f"RESPONSE {i+1}:\n{r.get('content', '')[:2000]}"
        for i, r in enumerate(responses)
    )

    force_clause = "You MUST pick one." if force else (
        "If ALL responses are low quality or fundamentally flawed, respond PICK: NONE to request more samples."
    )

    judge_prompt = f"""You are judging {len(responses)} responses for this step:

STEP: {step}

{responses_text}

Evaluate each response on: correctness, completeness, actionability.
Note which responses agree with each other (majority alignment = good signal, but a brilliant minority answer can win).

{force_clause}

Reply in this format:
REASONING: <1-2 sentences on why you picked this one>
PICK: <number 1-{len(responses)}, or NONE>"""

    result = await claude(judge_prompt, model=MODELS["judge"],
                          system="You are an expert judge. Pick the single best response. Quality over popularity.",
                          timeout=timeout)
    cost = result.get("cost", 0)
    text = result.get("content", "")

    pick_match = re.search(r"PICK:\s*(\d+|NONE)", text, re.IGNORECASE)
    if not pick_match or pick_match.group(1).upper() == "NONE":
        return None, cost

    try:
        idx = int(pick_match.group(1)) - 1
        return responses[max(0, min(idx, len(responses) - 1))], cost
    except (ValueError, IndexError):
        return responses[0] if responses else None, cost


# ── Phase 2.5: PER-STEP VERIFICATION ──────────────────────────────────

async def _verify_step(step: str, answer: str, step_num: int,
                       verbose: bool, timeout: float) -> tuple[bool, str, float]:
    """Quick per-step verification. Catches errors before they cascade."""
    verify_prompt = f"""Step: {step}

Answer: {answer[:2000]}

Does this answer correctly and completely address the step?
Reply: STEP_OK or STEP_ISSUE: <brief description of the problem>"""

    result = await claude(verify_prompt, model=MODELS["verifier"],
                          system="Strict step validator. Only STEP_OK if genuinely correct.",
                          timeout=timeout)
    cost = result.get("cost", 0)
    text = result.get("content", "")

    if "STEP_OK" in text:
        return True, "", cost

    issue_match = re.search(r"STEP_ISSUE:\s*(.+)", text, re.DOTALL)
    issue = issue_match.group(1).strip()[:200] if issue_match else "Unknown issue"
    if verbose:
        print(f"  [verify step {step_num}] ISSUE: {issue}", file=sys.stderr)
    return False, issue, cost


# ── Phase 3: COMPOSE ──────────────────────────────────────────────────

async def compose(task: str, step_results: list[dict], verbose: bool = False,
                  timeout: float = TIMEOUT_SECONDS) -> tuple[str, float]:
    """Merge step results into a single coherent answer. Returns (answer, cost)."""
    if len(step_results) == 1:
        return step_results[0]["answer"], 0

    steps_block = "\n\n".join(
        f"### Step {r['step_num']}: {r['step']}\n"
        f"(agreement: {r['agree_count']} votes, {r['total_sampled']} sampled)\n\n"
        f"{r['answer']}"
        for r in step_results
    )

    compose_prompt = f"""Original task: {task}

The task was decomposed into steps, each solved with multi-agent voting.
Here are the step results:

{steps_block}

Compose these into ONE coherent, complete answer to the original task.
Preserve all important details. Remove redundancy. Make it flow naturally."""

    result = await claude(compose_prompt, model=MODELS["composer"],
                          system="You are a skilled editor. Compose step results into a clear, complete answer.",
                          timeout=timeout)
    cost = result.get("cost", 0)
    return result.get("content", "") or steps_block, cost


# ── Phase 4: VERIFY ──────────────────────────────────────────────────

async def verify(task: str, answer: str, verbose: bool = False,
                 timeout: float = TIMEOUT_SECONDS) -> dict:
    """Check if the composed answer actually solves the original task."""
    verify_prompt = f"""Original task: {task}

Proposed answer:
{answer[:4000]}

Evaluate this answer:
1. Does it fully address the original task?
2. Are there any errors, gaps, or missing pieces?
3. Is it actionable and specific?

YOU MUST reply in this EXACT format (all 4 fields required):

VERDICT: PASS or FAIL
CONFIDENCE: X/10
ISSUES: <list any problems, or "none">
LEARNING: [mistake|strategy|pattern|constraint] <Extract ONE reusable insight from this task/answer>

Example: "LEARNING: [strategy] When decomposing auth tasks, always separate token generation from validation steps"

The LEARNING field is MANDATORY. Use one of these exact categories: mistake, strategy, pattern, constraint."""

    result = await claude(verify_prompt, model=MODELS["verifier"],
                          system="You are a strict quality reviewer. Only PASS if the answer is genuinely complete and correct.",
                          timeout=timeout)
    text = result.get("content", "")
    cost = result.get("cost", 0)

    verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    conf_match = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)", text)
    issues_match = re.search(r"ISSUES:\s*(.+?)(?:\nLEARNING:|\Z)", text, re.DOTALL)

    return {
        "passed": bool(verdict_match and verdict_match.group(1).upper() == "PASS"),
        "confidence": float(conf_match.group(1)) if conf_match else 5.0,
        "issues": issues_match.group(1).strip() if issues_match else "",
        "raw": text,
        "session_id": result.get("session_id"),
        "cost": cost,
    }


# ── Orchestrator: The MAKER Loop ─────────────────────────────────────

def _progress(msg: str, verbose: bool, progress: bool) -> None:
    """Print progress message if progress or verbose mode is on."""
    if verbose or progress:
        print(msg, file=sys.stderr, flush=True)


async def run(task: str, tags: list[str] | None = None,
              verbose: bool = False, mode: str = "maker",
              tools: bool = False, tools_rw: bool = False, cwd: str | None = None,
              workers: int = 3, worker_model: str = "sonnet",
              timeout: float = TIMEOUT_SECONDS, max_loops: int = MAX_LOOPS,
              progress: bool = False) -> dict:
    """Main entry point.

    mode="maker": decompose → vote per step → compose → verify → learn → loop
    mode="opinion": v1 parallel opinions (3 agents + consensus/judge)
    """
    if mode == "opinion":
        return await run_opinion(task, tags=tags, num_workers=workers, verbose=verbose,
                                 tools=tools, tools_rw=tools_rw, cwd=cwd, worker_model=worker_model)

    t0 = time.monotonic()
    total_cost = 0
    answer = ""  # Initialize before loop to avoid NameError on early exit

    # 1. RECALL
    _progress("[1/6] Recalling learnings...", verbose, progress)
    learnings = recall(tags)
    learnings_text = format_learnings(learnings)
    if verbose and learnings:
        print(f"\n[recall] Loaded {len(learnings)} learnings", file=sys.stderr)

    # Check for resumable checkpoint
    checkpoint = _load_checkpoint(task)
    resume_step = 0
    resumed_results = []
    if checkpoint:
        resume_step = checkpoint["step_num"] + 1
        resumed_results = checkpoint["step_results"]
        total_cost = checkpoint["total_cost"]
        if verbose:
            print(f"\n[resume] Found checkpoint at step {checkpoint['step_num']}. "
                  f"Resuming from step {resume_step}.", file=sys.stderr)

    for loop in range(max_loops):
        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[loop {loop+1}/{max_loops}]", file=sys.stderr)

        # 2. DECOMPOSE
        _progress("[2/6] Decomposing task...", verbose, progress)
        hook_ctx = {"task": task, "learnings_text": learnings_text, "loop": loop,
                     "tags": tags, "cwd": cwd, "tools": tools, "tools_rw": tools_rw}
        overrides = await run_hooks("pre_decompose", hook_ctx)
        task = overrides.get("task", task)
        learnings_text = overrides.get("learnings_text", learnings_text)

        if verbose:
            print(f"\n[decompose] Breaking task into atomic steps...", file=sys.stderr)
        steps, decompose_cost = await decompose(task, learnings_text, verbose=verbose, timeout=timeout)
        total_cost += decompose_cost

        overrides = await run_hooks("post_decompose", {"steps": steps, "task": task, "loop": loop})
        steps = overrides.get("steps", steps)

        if verbose:
            print(f"[decompose] {len(steps)} steps:", file=sys.stderr)
            for i, s in enumerate(steps):
                print(f"  {i+1}. {s[:100]}", file=sys.stderr)

        base_context = f"Overall task: {task}"
        if learnings_text:
            base_context += f"\n\n{learnings_text}"

        # 3. VOTE each step (with per-step verification + checkpointing)
        _progress(f"[3/6] Voting on {len(steps)} steps...", verbose, progress)
        step_results = resumed_results if (loop == 0 and resumed_results) else []
        compressed_history = []
        start_step = resume_step if (loop == 0 and resumed_results) else 0
        # Clear resume state after first use
        if loop == 0:
            resume_step = 0
            resumed_results = []

        for i in range(start_step, len(steps)):
            step = steps[i]
            _progress(f"  Step {i+1}/{len(steps)}", verbose, progress)
            if verbose:
                print(f"\n[step {i+1}/{len(steps)}] Voting on: {step[:80]}...", file=sys.stderr)

            context = _build_context(base_context, step_results, compressed_history)

            overrides = await run_hooks("pre_vote", {"step": step, "step_num": i + 1,
                                                       "total_steps": len(steps), "context": context,
                                                       "worker_model": worker_model})
            step = overrides.get("step", step)
            step_worker_model = overrides.get("worker_model", worker_model)

            # Vote with judge selection
            result = await vote_step(step, i + 1, len(steps), context,
                                    verbose=verbose, tools=tools, tools_rw=tools_rw, cwd=cwd,
                                    worker_model=step_worker_model, timeout=timeout)

            overrides = await run_hooks("post_vote", {"step": step, "step_num": i + 1,
                                                       "result": result})
            if "result" in overrides:
                result = overrides["result"]

            # Per-step verification — catch errors immediately
            step_ok, step_issue, verify_cost = await _verify_step(
                step, result["answer"], i + 1, verbose, timeout)
            total_cost += verify_cost

            if not step_ok:
                # Retry this step once with the issue as feedback
                if verbose:
                    print(f"  [step {i+1}] Re-voting with feedback: {step_issue}", file=sys.stderr)
                feedback_context = context + f"\n\nPREVIOUS ATTEMPT ISSUE: {step_issue}\nFix this issue."
                result = await vote_step(step, i + 1, len(steps), feedback_context,
                                        verbose=verbose, tools=tools, tools_rw=tools_rw, cwd=cwd,
                                        worker_model=step_worker_model, timeout=timeout)
                total_cost += result.get("cost", 0)

            step_results.append(result)
            total_cost += result.get("cost", 0)

            for sid in result.get("session_ids", []):
                if sid:
                    save_session(f"step-{i+1}", sid, step, 7.0, tags)

            # Checkpoint every N steps
            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                _save_checkpoint(task, i, step_results, total_cost, loop)
                if verbose:
                    print(f"  [checkpoint] Saved at step {i+1}", file=sys.stderr)

            # Compress older steps to keep context manageable
            if len(step_results) > 5 and len(step_results) % 10 == 0:
                chunk_end = len(step_results) - 5
                chunk_start = max(0, chunk_end - 10)
                summary, compress_cost = await _compress_steps(
                    step_results, chunk_start, chunk_end, timeout)
                total_cost += compress_cost
                compressed_history.append(f"Steps {chunk_start+1}-{chunk_end}: {summary}")
                if verbose:
                    print(f"  [compress] Compressed steps {chunk_start+1}-{chunk_end}", file=sys.stderr)

        if verbose:
            total_votes = sum(r["total_sampled"] for r in step_results)
            total_flagged = sum(r["flagged"] for r in step_results)
            print(f"\n[execute] Done. {total_votes} total samples, {total_flagged} red-flagged", file=sys.stderr)
            all_flag_reasons = {}
            for r in step_results:
                for reason, count in r.get("flagged_reasons", {}).items():
                    all_flag_reasons[reason] = all_flag_reasons.get(reason, 0) + count
            if all_flag_reasons:
                print(f"[execute] Flag reasons:", file=sys.stderr)
                for reason, count in sorted(all_flag_reasons.items(), key=lambda x: -x[1]):
                    print(f"  - {reason}: {count}x", file=sys.stderr)

        # 4. COMPOSE
        _progress("[4/6] Composing answer...", verbose, progress)
        if verbose:
            print(f"\n[compose] Merging step results...", file=sys.stderr)
        answer, compose_cost = await compose(task, step_results, verbose=verbose, timeout=timeout)
        total_cost += compose_cost

        overrides = await run_hooks("post_compose", {"answer": answer, "task": task,
                                                       "step_results": step_results})
        answer = overrides.get("answer", answer)

        # 5. VERIFY
        _progress("[5/6] Verifying answer...", verbose, progress)
        overrides = await run_hooks("pre_verify", {"answer": answer, "task": task, "loop": loop})
        answer = overrides.get("answer", answer)

        if verbose:
            print(f"\n[verify] Checking answer against original task...", file=sys.stderr)
        verification = await verify(task, answer, verbose=verbose, timeout=timeout)
        total_cost += verification.get("cost", 0)

        if verbose:
            status = "PASS" if verification["passed"] else "FAIL"
            print(f"[verify] {status} (confidence: {verification['confidence']}/10)", file=sys.stderr)
            if verification["issues"] and verification["issues"].lower() != "none":
                print(f"[verify] Issues: {verification['issues'][:200]}", file=sys.stderr)

        overrides = await run_hooks("post_verify", {"verification": verification,
                                                       "answer": answer, "task": task, "loop": loop})
        if "verification" in overrides:
            verification = overrides["verification"]

        # 6. LEARN
        _progress("[6/6] Extracting learnings...", verbose, progress)
        new_learnings = learn(verification["raw"], tags)
        if verbose and new_learnings:
            print(f"[learn] Extracted {len(new_learnings)} learnings", file=sys.stderr)

        await run_hooks("post_learn", {"learnings": new_learnings, "verification": verification,
                                        "task": task, "loop": loop})

        if verification["passed"]:
            _clear_checkpoints(task)
            elapsed = round(time.monotonic() - t0, 2)
            if verbose:
                total_cost_display = round(total_cost, 3)
                print(f"\n[done] PASSED on loop {loop+1}, {elapsed}s, ${total_cost_display} total", file=sys.stderr)
                print(f"[sessions] Resumable sessions saved to: python -m swarm --sessions", file=sys.stderr)

            await run_hooks("post_loop", {"answer": answer, "cost": total_cost, "passed": True,
                                           "loop": loop + 1, "elapsed": elapsed, "task": task})
            return {"answer": answer, "cost": total_cost}

        # Failed — feed issues back
        if verbose:
            print(f"\n[loop] Verification FAILED. Feeding issues back for re-planning...", file=sys.stderr)
        failure_context = f"LEARNING: [mistake] Previous attempt failed verification: {verification['issues']}"
        learn(failure_context, tags)
        learnings = recall(tags)
        learnings_text = format_learnings(learnings)

    # Exhausted all loops
    _clear_checkpoints(task)
    elapsed = round(time.monotonic() - t0, 2)
    if verbose:
        total_cost_display = round(total_cost, 3)
        print(f"\n[done] Exhausted {max_loops} loops. Returning best effort. {elapsed}s, ${total_cost_display} total", file=sys.stderr)

    await run_hooks("post_loop", {"answer": answer, "cost": total_cost, "passed": False,
                                   "loop": max_loops, "elapsed": elapsed, "task": task})
    return {"answer": answer, "cost": total_cost}
