"""MAKER loop: decompose, vote, compose, verify, learn."""

import asyncio
import json
import re
import sys
import time

from .agent import claude, red_flag
from .config import (
    BATCH_SIZE, K_AHEAD, MAX_LOOPS, MAX_SAMPLES, MODELS, ROLES, TIMEOUT_SECONDS,
)
from .hooks import run_hooks
from .memory import format_learnings, learn, recall, save_session
from .opinion import run_opinion


# ── Phase 1: DECOMPOSE ────────────────────────────────────────────────────

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


# ── Phase 2: VOTE per step ───────────────────────────────────────────────

async def vote_step(
    step: str,
    step_num: int,
    total_steps: int,
    context: str,
    verbose: bool = False,
    tools: bool = False,
    tools_rw: bool = False,
    cwd: str | None = None,
    worker_model: str = "haiku",
    timeout: float = TIMEOUT_SECONDS,
    k_ahead: int = K_AHEAD,
) -> dict:
    """Adaptive first-to-ahead-by-K voting for a single step."""
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

        if len(all_responses) < 2:
            continue

        agreement, agreement_cost = await _check_agreement(all_responses, step, k_ahead)
        total_cost += agreement_cost
        if agreement:
            if verbose:
                print(f"  [vote step {step_num}] CONSENSUS after {total_sampled} samples "
                      f"({agreement['agree_count']}/{len(all_responses)} agree)", file=sys.stderr)
            return {
                "step": step,
                "step_num": step_num,
                "answer": agreement["merged"],
                "agree_count": agreement["agree_count"],
                "total_sampled": total_sampled,
                "flagged": flagged_count,
                "flagged_reasons": flagged_reasons,
                "rounds": round_num,
                "session_ids": [r.get("session_id") for r in all_responses if r.get("session_id")],
                "cost": total_cost,
            }

    # Fallback: no K-agreement — use Sonnet tiebreaker if workers were Haiku
    if verbose:
        print(f"  [vote step {step_num}] NO CONSENSUS after {MAX_SAMPLES} samples.", file=sys.stderr)

    if all_responses and worker_model != "sonnet":
        if verbose:
            print(f"  [vote step {step_num}] Escalating to Sonnet tiebreaker...", file=sys.stderr)
        responses_text = "\n\n---\n\n".join(
            f"RESPONSE {i+1}:\n{r.get('content', '')[:1500]}"
            for i, r in enumerate(all_responses)
        )
        judge_result = await claude(
            f"Step: {step}\n\nPick the BEST response (most correct, complete, actionable). "
            f"Output ONLY the response number (1-{len(all_responses)}).\n\n{responses_text}",
            model="sonnet", system="You are a strict judge. Pick the single best response.",
            timeout=timeout)
        total_cost += judge_result.get("cost", 0)
        try:
            pick_idx = int(judge_result.get("content", "").strip().split()[0]) - 1
            best = all_responses[max(0, min(pick_idx, len(all_responses) - 1))]
        except (ValueError, IndexError):
            best = max(all_responses, key=lambda r: len(r.get("content", "")))
    else:
        print(f"[vote step {step_num}] WARNING: Fallback to longest response (no consensus)", file=sys.stderr)
        best = max(all_responses, key=lambda r: len(r.get("content", ""))) if all_responses else {"content": ""}

    return {
        "step": step,
        "step_num": step_num,
        "answer": best.get("content", ""),
        "agree_count": 1,
        "total_sampled": total_sampled,
        "flagged": flagged_count,
        "flagged_reasons": flagged_reasons,
        "rounds": round_num,
        "session_ids": [r.get("session_id") for r in all_responses if r.get("session_id")],
        "cost": total_cost,
    }


async def _check_agreement(responses: list[dict], step: str, k: int) -> tuple[dict | None, float]:
    """Ask cheap model if K+ responses substantively agree."""
    summaries = "\n---\n".join(
        f"Response {i+1}:\n{r['content'][:1500]}"
        for i, r in enumerate(responses)
    )

    check_prompt = f"""Step being executed: {step}

Here are {len(responses)} responses:

{summaries}

How many of these responses substantively agree on the approach and answer?
Reply in this EXACT format:

AGREE_COUNT: <number>
AGREE_INDICES: <comma-separated indices of agreeing responses, 1-based>
MERGED: <if AGREE_COUNT >= {k}, merge the agreeing responses into one clean answer. Otherwise write NONE>"""

    result = await claude(check_prompt, model="haiku",
                          system="You are a concise evaluator. Count agreement precisely.")
    cost = result.get("cost", 0)
    text = result.get("content", "")

    count_match = re.search(r"AGREE_COUNT:\s*(\d+)", text)
    if not count_match:
        return None, cost

    agree_count = int(count_match.group(1))
    if agree_count < k:
        return None, cost

    merged_match = re.search(r"MERGED:\s*(.+)", text, re.DOTALL)
    if not merged_match or merged_match.group(1).strip().upper() == "NONE":
        return None, cost

    return {"agree_count": agree_count, "merged": merged_match.group(1).strip()}, cost


# ── Phase 3: COMPOSE ──────────────────────────────────────────────────────

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


# ── Phase 4: VERIFY ──────────────────────────────────────────────────────

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


# ── Orchestrator: The MAKER Loop ─────────────────────────────────────────

def _progress(msg: str, verbose: bool, progress: bool) -> None:
    """Print progress message if progress or verbose mode is on."""
    if verbose or progress:
        print(msg, file=sys.stderr, flush=True)


async def run(task: str, tags: list[str] | None = None,
              verbose: bool = False, mode: str = "maker",
              tools: bool = False, tools_rw: bool = False, cwd: str | None = None,
              workers: int = 3, worker_model: str = "haiku", max_cost: float = 1.00,
              timeout: float = TIMEOUT_SECONDS, k_ahead: int = K_AHEAD, max_loops: int = MAX_LOOPS,
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
    answer = ""  # Initialize before loop to avoid NameError on early cost ceiling hit

    # 1. RECALL
    _progress("[1/6] Recalling learnings...", verbose, progress)
    learnings = recall(tags)
    learnings_text = format_learnings(learnings)
    if verbose and learnings:
        print(f"\n[recall] Loaded {len(learnings)} learnings", file=sys.stderr)

    for loop in range(max_loops):
        if total_cost > max_cost:
            if verbose:
                print(f"\n[cost-limit] WARNING: Total cost ${round(total_cost, 3)} exceeds limit ${max_cost}. "
                      f"Returning best answer so far.", file=sys.stderr)
            return {"answer": answer, "cost": total_cost}

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

        # 3. VOTE each step
        _progress(f"[3/6] Voting on {len(steps)} steps...", verbose, progress)
        step_results = []
        for i, step in enumerate(steps):
            _progress(f"  Step {i+1}/{len(steps)}", verbose, progress)
            if verbose:
                print(f"\n[step {i+1}/{len(steps)}] Voting on: {step[:80]}...", file=sys.stderr)

            context = base_context
            if step_results:
                prior = "\n".join(f"Step {r['step_num']} result: {r['answer'][:300]}"
                                  for r in step_results)
                context += f"\n\nPrior step results:\n{prior}"

            overrides = await run_hooks("pre_vote", {"step": step, "step_num": i + 1,
                                                       "total_steps": len(steps), "context": context,
                                                       "worker_model": worker_model})
            step = overrides.get("step", step)
            step_worker_model = overrides.get("worker_model", worker_model)

            result = await vote_step(step, i + 1, len(steps), context,
                                    verbose=verbose, tools=tools, tools_rw=tools_rw, cwd=cwd,
                                    worker_model=step_worker_model, timeout=timeout, k_ahead=k_ahead)

            overrides = await run_hooks("post_vote", {"step": step, "step_num": i + 1,
                                                       "result": result})
            if "result" in overrides:
                result = overrides["result"]

            step_results.append(result)
            total_cost += result.get("cost", 0)

            for sid in result.get("session_ids", []):
                if sid:
                    save_session(f"step-{i+1}", sid, step, 7.0, tags)

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
    elapsed = round(time.monotonic() - t0, 2)
    if verbose:
        total_cost_display = round(total_cost, 3)
        print(f"\n[done] Exhausted {max_loops} loops. Returning best effort. {elapsed}s, ${total_cost_display} total", file=sys.stderr)

    await run_hooks("post_loop", {"answer": answer, "cost": total_cost, "passed": False,
                                   "loop": max_loops, "elapsed": elapsed, "task": task})
    return {"answer": answer, "cost": total_cost}
