"""
swarm.py v2 — MAKER-informed multi-agent collective intelligence.

Based on "Solving a Million-Step LLM Task with Zero Errors" (Meyerson et al., 2025).
Key insight: decompose → adaptive vote per step → red-flag → compose → verify → learn → loop.

    python swarm.py "Design an auth system"
    python swarm.py -v --tags coding "Write a rate limiter"
    python swarm.py --resume                        # resume best session
    python swarm.py --sessions                      # list resumable sessions
    python swarm.py --mode opinion "Is Rust better?" # v1 mode: parallel opinions

Flow:  RECALL → DECOMPOSE → VOTE each step → COMPOSE → VERIFY → LEARN
         ↑                                                         |
         └──────────── re-plan with learnings if verify fails ─────┘
"""

import asyncio
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────

MODELS = {
    "planner":   "sonnet",   # decomposes tasks into atomic steps
    "worker":    "haiku",    # executes steps — cheap, voting makes it reliable
    "composer":  "sonnet",   # merges step results into final answer
    "verifier":  "sonnet",   # checks result against original task
}

K_AHEAD = 3              # votes ahead needed to win (from MAKER paper)
MAX_SAMPLES = 10          # max vote samples per step before fallback
BATCH_SIZE = 3            # parallel workers per vote round
MAX_LOOPS = 3             # max verify→re-plan loops
TIMEOUT_SECONDS = 120
MEMORY_DIR = Path.home() / ".swarm"
MEMORY_FILE = MEMORY_DIR / "learnings.jsonl"
SESSIONS_FILE = MEMORY_DIR / "sessions.jsonl"

ROLES = [
    "You are a pragmatist. Find the simplest working solution.",
    "You are a skeptic. Find what could go wrong. Identify edge cases.",
    "You are an innovator. Find unconventional approaches others would miss.",
    "You are a systems thinker. Consider second-order effects and feedback loops.",
    "You are a historian. What has worked or failed before in similar situations?",
    "You are a contrarian. Argue against the obvious answer. Find the hidden truth.",
]


# ── Memory ─────────────────────────────────────────────────────────────────

def recall(tags: list[str] | None = None, limit: int = 5) -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    seen = {}
    for line in MEMORY_FILE.read_text().splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not entry.get("active", True):
            continue
        seen[entry["id"]] = entry
    learnings = list(seen.values())
    if tags:
        tag_set = set(tags)
        learnings = [l for l in learnings if set(l.get("tags", [])) & tag_set]
    learnings.sort(key=lambda l: l.get("confidence", 0.7), reverse=True)
    return learnings[:limit]


def learn(text: str, tags: list[str] | None = None) -> list[dict]:
    pattern = r"LEARNING:\s*\[(mistake|strategy|pattern|constraint)\]\s*(.+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        return []
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for category, content in matches:
        content = content.strip()
        if len(content) < 20 or len(content) > 500:
            continue
        entries.append({
            "id": uuid.uuid4().hex[:12],
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category.lower(),
            "tags": tags or [],
            "content": content,
            "confidence": 0.7,
            "times_confirmed": 0,
            "active": True,
        })
    if entries:
        with open(MEMORY_FILE, "a") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
    return entries


def format_learnings(learnings: list[dict]) -> str:
    if not learnings:
        return ""
    lines = ["## LESSONS FROM PAST RUNS (apply these):"]
    for l in learnings:
        cat = l.get("category", "insight").upper()
        conf = l.get("confidence", 0.7)
        confirmed = l.get("times_confirmed", 0)
        lines.append(f"- [{cat}] {l['content']} (confidence: {conf:.2f}, confirmed {confirmed}x)")
    return "\n".join(lines)


# ── Sessions ───────────────────────────────────────────────────────────────

def save_session(role: str, session_id: str, prompt: str,
                 confidence: float, tags: list[str] | None = None):
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "role": role,
        "session_id": session_id,
        "prompt": prompt[:200],
        "confidence": confidence,
        "tags": tags or [],
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with open(SESSIONS_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_sessions(limit: int = 10) -> list[dict]:
    if not SESSIONS_FILE.exists():
        return []
    sessions = []
    for line in SESSIONS_FILE.read_text().splitlines():
        try:
            sessions.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return sessions[-limit:]


def get_best_session() -> dict | None:
    sessions = load_sessions(20)
    if not sessions:
        return None
    latest_ts = sessions[-1]["ts"][:16]
    last_run = [s for s in sessions if s["ts"][:16] == latest_ts]
    return max(last_run, key=lambda s: s.get("confidence", 0)) if last_run else None


# ── Claude Code Agent ──────────────────────────────────────────────────────

async def claude(
    prompt: str,
    model: str = "sonnet",
    system: str | None = None,
    session_id: str | None = None,
    timeout: float = TIMEOUT_SECONDS,
    tools: bool = False,
    tools_rw: bool = False,
    cwd: str | None = None,
) -> dict:
    """Run a Claude Code agent. Returns {content, session_id, cost, elapsed, error}."""
    cmd = ["claude", "-p", prompt, "--output-format", "json", "--model", model]
    if system:
        cmd += ["--append-system-prompt", system]
    if session_id:
        cmd += ["--resume", session_id]
    if tools or tools_rw:
        cmd += ["--permission-mode", "acceptEdits"]
        if tools_rw:
            cmd += ["--allowed-tools", "Read,Glob,Grep,Bash,Edit,Write"]
        else:
            cmd += ["--allowed-tools", "Read,Glob,Grep,Bash"]

    t0 = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {"content": "", "session_id": None, "cost": 0,
                    "elapsed": round(time.monotonic() - t0, 2),
                    "error": f"Timed out after {timeout}s"}
        elapsed = round(time.monotonic() - t0, 2)

        if proc.returncode != 0:
            return {"content": "", "session_id": None, "cost": 0, "elapsed": elapsed,
                    "error": stderr.decode().strip()[:500] or f"exit code {proc.returncode}"}

        data = json.loads(stdout.decode())
        return {
            "content": data.get("result", ""),
            "session_id": data.get("session_id"),
            "cost": data.get("cost_usd", 0),
            "elapsed": elapsed,
            "error": None,
        }
    except Exception as e:
        return {"content": "", "session_id": None, "cost": 0,
                "elapsed": round(time.monotonic() - t0, 2),
                "error": f"{type(e).__name__}: {e}"}


# ── Red-Flagging (MAKER insight #3) ───────────────────────────────────────

def red_flag(result: dict, max_len: int = 3000, is_code_task: bool = False) -> str | None:
    """Check if a response should be discarded. Returns reason or None if OK.

    Checks:
    1. Errors
    2. Empty/trivially short responses
    3. Response length exceeding max
    4. Low self-reported confidence
    5. Refusal patterns (I cannot, I'm unable, as an AI)
    6. Code-only responses (>80% indented lines for non-code tasks)
    """
    if result.get("error"):
        return f"error: {result['error']}"
    content = result.get("content", "")
    if not content or len(content.strip()) < 10:
        return "empty or trivially short response"
    if len(content) > max_len:
        return f"response too long ({len(content)} chars > {max_len})"

    # Check for refusal patterns
    lower_content = content.lower()
    refusal_patterns = ["i cannot", "i'm unable", "as an ai"]
    for pattern in refusal_patterns:
        if pattern in lower_content:
            return f"likely refusal: contains '{pattern}'"

    # Check for code-only responses (>80% indented lines) for non-code tasks
    if not is_code_task:
        lines = content.split("\n")
        if len(lines) > 3:  # only check if enough lines to be meaningful
            indented = sum(1 for line in lines if line and (line[0] in " \t"))
            indent_ratio = indented / len(lines)
            if indent_ratio > 0.8:
                return f"mostly code with no explanation ({indent_ratio:.0%} indented lines)"

    # Low self-reported confidence
    conf_match = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)", content)
    if conf_match and float(conf_match.group(1)) < 3:
        return f"self-reported confidence too low ({conf_match.group(1)}/10)"
    return None


# ── Phase 1: DECOMPOSE ────────────────────────────────────────────────────

async def decompose(task: str, learnings_text: str, verbose: bool = False) -> tuple[list[str], float]:
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

    result = await claude(task, model=MODELS["planner"], system=system)
    cost = result.get("cost", 0)
    if result["error"]:
        if verbose:
            print(f"  [decompose] ERROR: {result['error']}", file=sys.stderr)
        return [task], cost  # fallback: treat whole task as one step

    # Parse JSON array from response
    content = result.get("content", "")
    # Try to find JSON array in the response
    json_match = re.search(r'\[.*\]', content, re.DOTALL)
    if json_match:
        try:
            steps = json.loads(json_match.group())
            if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
                return steps, cost
        except json.JSONDecodeError:
            pass

    # Fallback: split numbered lines
    lines = [re.sub(r'^\d+[\.\)]\s*', '', l.strip())
             for l in content.splitlines() if re.match(r'^\d+[\.\)]', l.strip())]
    return (lines if lines else [task]), cost


# ── Phase 2: VOTE per step (MAKER insight #2) ─────────────────────────────

async def vote_step(
    step: str,
    step_num: int,
    total_steps: int,
    context: str,
    verbose: bool = False,
    tools: bool = False,
    tools_rw: bool = False,
    cwd: str | None = None,
) -> dict:
    """Adaptive first-to-ahead-by-K voting for a single step.

    Samples workers in parallel batches. Red-flags bad responses.
    Keeps going until one answer leads by K votes.
    Uses cheap Haiku for agreement checks.
    """
    all_responses = []   # valid (non-red-flagged) responses
    flagged_count = 0
    flagged_reasons = {}  # track which red-flag reasons fired most
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
        # Launch a batch of workers in parallel
        batch_count = min(BATCH_SIZE, MAX_SAMPLES - total_sampled)
        tasks = []
        for i in range(batch_count):
            role = ROLES[(total_sampled + i) % len(ROLES)]
            tasks.append(claude(step_prompt, model=MODELS["worker"], system=role,
                               tools=tools, tools_rw=tools_rw, cwd=cwd))

        results = await asyncio.gather(*tasks)
        total_sampled += batch_count

        # Accumulate costs and red-flag filter
        for r in results:
            total_cost += r.get("cost", 0)
            flag = red_flag(r)
            if flag:
                flagged_count += 1
                # Track which reasons fire
                flagged_reasons[flag] = flagged_reasons.get(flag, 0) + 1
                if verbose:
                    print(f"  [vote step {step_num}] RED-FLAG: {flag}", file=sys.stderr)
            else:
                all_responses.append(r)

        if verbose:
            print(f"  [vote step {step_num}] round {round_num}: "
                  f"{len(all_responses)} valid, {flagged_count} flagged, "
                  f"{total_sampled} total sampled", file=sys.stderr)

        # Need at least 2 valid responses to check agreement
        if len(all_responses) < 2:
            continue

        # Check if we have K agreeing responses
        agreement, agreement_cost = await _check_agreement(all_responses, step, K_AHEAD)
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

    # Fallback: no K-agreement reached, use longest response
    if verbose:
        print(f"  [vote step {step_num}] NO CONSENSUS after {MAX_SAMPLES} samples. Using best response.", file=sys.stderr)
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
    """Ask cheap model if K+ responses substantively agree. Returns (merged_answer, cost) or (None, cost)."""
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

async def compose(task: str, step_results: list[dict], verbose: bool = False) -> tuple[str, float]:
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
                          system="You are a skilled editor. Compose step results into a clear, complete answer.")
    cost = result.get("cost", 0)
    return result.get("content", "") or steps_block, cost


# ── Phase 4: VERIFY ───────────────────────────────────────────────────────

async def verify(task: str, answer: str, verbose: bool = False) -> dict:
    """Check if the composed answer actually solves the original task."""
    verify_prompt = f"""Original task: {task}

Proposed answer:
{answer[:4000]}

Evaluate this answer:
1. Does it fully address the original task?
2. Are there any errors, gaps, or missing pieces?
3. Is it actionable and specific?

Reply in this EXACT format:
VERDICT: PASS or FAIL
CONFIDENCE: X/10
ISSUES: <list any problems, or "none">
LEARNING: [category] <any reusable insight from this evaluation>"""

    result = await claude(verify_prompt, model=MODELS["verifier"],
                          system="You are a strict quality reviewer. Only PASS if the answer is genuinely complete and correct.")
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


# ── Orchestrator: The MAKER Loop ──────────────────────────────────────────

async def run(task: str, tags: list[str] | None = None,
              verbose: bool = False, mode: str = "maker",
              tools: bool = False, tools_rw: bool = False, cwd: str | None = None, workers: int = 3) -> str:
    """Main entry point.

    mode="maker": decompose → vote per step → compose → verify → learn → loop
    mode="opinion": v1 parallel opinions (3 agents + consensus/judge)
    """
    if mode == "opinion":
        return await _run_opinion(task, tags=tags, num_workers=workers, verbose=verbose,
                                  tools=tools, tools_rw=tools_rw, cwd=cwd)

    t0 = time.monotonic()
    total_cost = 0

    # 1. RECALL
    learnings = recall(tags)
    learnings_text = format_learnings(learnings)
    if verbose and learnings:
        print(f"\n[recall] Loaded {len(learnings)} learnings", file=sys.stderr)

    for loop in range(MAX_LOOPS):
        if verbose:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"[loop {loop+1}/{MAX_LOOPS}]", file=sys.stderr)

        # 2. DECOMPOSE
        if verbose:
            print(f"\n[decompose] Breaking task into atomic steps...", file=sys.stderr)
        steps, decompose_cost = await decompose(task, learnings_text, verbose=verbose)
        total_cost += decompose_cost
        if verbose:
            print(f"[decompose] {len(steps)} steps:", file=sys.stderr)
            for i, s in enumerate(steps):
                print(f"  {i+1}. {s[:100]}", file=sys.stderr)

        # Build context for workers (original task + learnings + prior step results)
        base_context = f"Overall task: {task}"
        if learnings_text:
            base_context += f"\n\n{learnings_text}"

        # 3. VOTE each step
        step_results = []
        for i, step in enumerate(steps):
            if verbose:
                print(f"\n[step {i+1}/{len(steps)}] Voting on: {step[:80]}...", file=sys.stderr)

            # Include results from prior steps as context
            context = base_context
            if step_results:
                prior = "\n".join(f"Step {r['step_num']} result: {r['answer'][:300]}"
                                  for r in step_results)
                context += f"\n\nPrior step results:\n{prior}"

            result = await vote_step(step, i + 1, len(steps), context,
                                    verbose=verbose, tools=tools, tools_rw=tools_rw, cwd=cwd)
            step_results.append(result)
            total_cost += result.get("cost", 0)

            # Save best session for resume
            for sid in result.get("session_ids", []):
                if sid:
                    save_session(f"step-{i+1}", sid, step, 7.0, tags)

        if verbose:
            total_votes = sum(r["total_sampled"] for r in step_results)
            total_flagged = sum(r["flagged"] for r in step_results)
            print(f"\n[execute] Done. {total_votes} total samples, {total_flagged} red-flagged", file=sys.stderr)
            # Aggregate flag reasons across all steps
            all_flag_reasons = {}
            for r in step_results:
                for reason, count in r.get("flagged_reasons", {}).items():
                    all_flag_reasons[reason] = all_flag_reasons.get(reason, 0) + count
            if all_flag_reasons:
                print(f"[execute] Flag reasons:", file=sys.stderr)
                for reason, count in sorted(all_flag_reasons.items(), key=lambda x: -x[1]):
                    print(f"  - {reason}: {count}x", file=sys.stderr)

        # 4. COMPOSE
        if verbose:
            print(f"\n[compose] Merging step results...", file=sys.stderr)
        answer, compose_cost = await compose(task, step_results, verbose=verbose)
        total_cost += compose_cost

        # 5. VERIFY
        if verbose:
            print(f"\n[verify] Checking answer against original task...", file=sys.stderr)
        verification = await verify(task, answer, verbose=verbose)
        total_cost += verification.get("cost", 0)

        if verbose:
            status = "PASS" if verification["passed"] else "FAIL"
            print(f"[verify] {status} (confidence: {verification['confidence']}/10)", file=sys.stderr)
            if verification["issues"] and verification["issues"].lower() != "none":
                print(f"[verify] Issues: {verification['issues'][:200]}", file=sys.stderr)

        # 6. LEARN — extract from verification output
        new_learnings = learn(verification["raw"], tags)
        if verbose and new_learnings:
            print(f"[learn] Extracted {len(new_learnings)} learnings", file=sys.stderr)

        # Check if we passed
        if verification["passed"]:
            elapsed = round(time.monotonic() - t0, 2)
            if verbose:
                total_cost_display = round(total_cost, 3)
                print(f"\n[done] PASSED on loop {loop+1}, {elapsed}s, ${total_cost_display} total", file=sys.stderr)
                print(f"[sessions] Resumable sessions saved to: python swarm.py --sessions", file=sys.stderr)
            return {"answer": answer, "cost": total_cost}

        # Failed — feed issues back as learnings for next loop
        if verbose:
            print(f"\n[loop] Verification FAILED. Feeding issues back for re-planning...", file=sys.stderr)
        failure_context = f"LEARNING: [mistake] Previous attempt failed verification: {verification['issues']}"
        learn(failure_context, tags)
        learnings = recall(tags)  # reload with new learnings
        learnings_text = format_learnings(learnings)

    # Exhausted all loops — return best effort
    if verbose:
        elapsed = round(time.monotonic() - t0, 2)
        total_cost_display = round(total_cost, 3)
        print(f"\n[done] Exhausted {MAX_LOOPS} loops. Returning best effort. {elapsed}s, ${total_cost_display} total", file=sys.stderr)
    return {"answer": answer, "cost": total_cost}


# ── v1 Opinion Mode (kept for comparison / simple questions) ──────────────

async def _run_opinion(task: str, tags: list[str] | None = None,
                       num_workers: int = 3, verbose: bool = False,
                       tools: bool = False, tools_rw: bool = False, cwd: str | None = None) -> tuple[str, float]:
    """v1 mode: parallel diverse opinions + consensus/judge. Returns (answer, total_cost)."""
    t0 = time.monotonic()
    learnings = recall(tags)
    learnings_text = format_learnings(learnings)
    total_cost = 0

    # Spawn workers
    if verbose:
        print(f"\n[opinion] Spawning {num_workers} workers...", file=sys.stderr)

    tasks = []
    for i in range(num_workers):
        role = ROLES[i % len(ROLES)]
        system = role
        if learnings_text:
            system += f"\n\n{learnings_text}"
        system += "\n\nAt the end, include: CONFIDENCE: X/10"
        tasks.append(claude(task, model="sonnet", system=system,
                            tools=tools, tools_rw=tools_rw, cwd=cwd))

    results = await asyncio.gather(*tasks)
    successful = [r for r in results if not r.get("error")]
    # Accumulate worker costs
    for r in results:
        total_cost += r.get("cost", 0)

    if not successful:
        return {"answer": "[ERROR] All workers failed", "cost": total_cost}

    # Consensus check
    summaries = "\n---\n".join(r["content"][:1500] for r in successful)
    check = await claude(
        f"Do these responses agree?\n\n{summaries}\n\nAGREE: YES or NO\nIf YES: MERGED: <combined>",
        model="haiku", system="Concise evaluator.")
    text = check.get("content", "")
    total_cost += check.get("cost", 0)

    if re.search(r"AGREE:\s*YES", text, re.IGNORECASE):
        m = re.search(r"MERGED:\s*(.+)", text, re.DOTALL)
        answer = m.group(1).strip() if m else successful[0]["content"]
    else:
        # Judge
        judge_prompt = f"Task: {task}\n\nResponses:\n{summaries}\n\nSynthesize the best answer."
        judge_result = await claude(judge_prompt, model="sonnet",
                                    system="You are a judge. Score each, synthesize the best.")
        answer = judge_result.get("content", successful[0]["content"])
        total_cost += judge_result.get("cost", 0)

    learn(answer, tags)
    elapsed = round(time.monotonic() - t0, 2)
    if verbose:
        total_cost_display = round(total_cost, 3)
        print(f"[done] Opinion mode complete, {elapsed}s, ${total_cost_display} total", file=sys.stderr)
    return {"answer": answer, "cost": total_cost}


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    global TIMEOUT_SECONDS, K_AHEAD, MAX_LOOPS

    import argparse
    p = argparse.ArgumentParser(
        description="Swarm v2: MAKER-informed multi-agent collective intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python swarm.py "Design an auth system" -v
  python swarm.py --tags coding "Write a rate limiter" -v
  python swarm.py --mode opinion "Is Rust better than Go?"
  python swarm.py --resume
  python swarm.py --sessions""",
    )
    p.add_argument("prompt", nargs="?", help="The task or question")
    p.add_argument("--mode", choices=["maker", "opinion"], default="maker",
                   help="maker=decompose+vote+verify loop (default), opinion=v1 parallel opinions")
    p.add_argument("--workers", type=int, default=3,
                   help="Number of workers for opinion mode (default: 3)")
    p.add_argument("--tags", type=str, default="", help="Comma-separated memory tags")
    p.add_argument("-t", "--timeout", type=int, default=TIMEOUT_SECONDS,
                   help=f"Timeout per agent call (default: {TIMEOUT_SECONDS}s)")
    p.add_argument("-k", "--k-ahead", type=int, default=K_AHEAD,
                   help=f"Votes ahead to win (default: {K_AHEAD})")
    p.add_argument("--max-loops", type=int, default=MAX_LOOPS,
                   help=f"Max verify→re-plan loops (default: {MAX_LOOPS})")
    p.add_argument("--tools", action="store_true",
                   help="Give workers tool access (Read, Glob, Grep, Bash - read-only)")
    p.add_argument("--tools-rw", action="store_true",
                   help="Give workers read-write tool access (Read, Glob, Grep, Bash, Edit, Write)")
    p.add_argument("--cwd", type=str, default=None,
                   help="Working directory for workers (default: current dir)")
    p.add_argument("--stdin", action="store_true", help="Read prompt from stdin")
    p.add_argument("-v", "--verbose", action="store_true", help="Print detailed progress")
    p.add_argument("--json", action="store_true", help="Output as JSON")
    p.add_argument("--resume", nargs="?", const="BEST", metavar="SESSION_ID",
                   help="Resume a worker session (default: best from last run)")
    p.add_argument("--sessions", action="store_true", help="List resumable sessions")
    args = p.parse_args()

    # Apply config overrides
    TIMEOUT_SECONDS = args.timeout
    K_AHEAD = args.k_ahead
    MAX_LOOPS = args.max_loops

    # Warn if using read-write tools
    if args.tools_rw:
        print("⚠️  WARNING: --tools-rw enables parallel workers to EDIT and WRITE files. "
              "Risk of data corruption if multiple workers write to same files simultaneously.",
              file=sys.stderr)

    # Sessions mode
    if args.sessions:
        sessions = load_sessions(20)
        if not sessions:
            print("No sessions saved yet.")
        else:
            print(f"{'ID':<40} {'Role':<12} {'Conf':>5}  {'Time':<20}  Prompt")
            print("-" * 100)
            for s in sessions:
                print(f"{s['session_id']:<40} {s['role']:<12} {s['confidence']:>4.1f}  "
                      f"{s['ts'][:19]:<20}  {s['prompt'][:40]}...")
        return

    # Resume mode
    if args.resume is not None:
        if args.resume == "BEST":
            session = get_best_session()
            if not session:
                print("No sessions to resume.", file=sys.stderr)
                sys.exit(1)
            session_id = session["session_id"]
            print(f"[resume] {session['role']} (confidence={session['confidence']}/10)", file=sys.stderr)
        else:
            session_id = args.resume

        follow_up = args.prompt or "Continue and refine your previous answer. Go deeper."
        result = asyncio.run(claude(follow_up, session_id=session_id))
        if result["error"]:
            print(f"[error] {result['error']}", file=sys.stderr)
            sys.exit(1)
        print(result["content"])
        return

    # Run swarm
    prompt = sys.stdin.read().strip() if args.stdin else args.prompt
    if not prompt:
        p.print_help()
        sys.exit(1)

    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None

    cwd = args.cwd or os.getcwd()
    result = asyncio.run(run(prompt, tags=tags, verbose=args.verbose, mode=args.mode,
                             tools=args.tools, tools_rw=args.tools_rw, cwd=cwd, workers=args.workers))

    # Handle both string and dict returns
    if isinstance(result, dict):
        answer = result.get("answer", "")
        cost = result.get("cost", 0)
    else:
        answer = result
        cost = 0

    if args.json:
        output = {"answer": answer, "prompt": prompt, "tags": tags, "mode": args.mode, "cost": round(cost, 3)}
        print(json.dumps(output))
    else:
        print(answer)


if __name__ == "__main__":
    main()
