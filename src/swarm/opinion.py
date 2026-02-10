"""Opinion mode: parallel diverse agents + consensus/judge."""

import re
import sys
import time

import asyncio

from .agent import claude
from .config import ROLES
from .memory import format_learnings, learn, recall


async def run_opinion(task: str, tags: list[str] | None = None,
                      num_workers: int = 3, verbose: bool = False,
                      tools: bool = False, tools_rw: bool = False, cwd: str | None = None,
                      worker_model: str = "haiku") -> dict:
    """Parallel diverse opinions + consensus/judge. Returns {answer, cost}."""
    t0 = time.monotonic()
    learnings = recall(tags)
    learnings_text = format_learnings(learnings)
    total_cost = 0

    if verbose:
        print(f"\n[opinion] Spawning {num_workers} workers...", file=sys.stderr)

    tasks = []
    for i in range(num_workers):
        role = ROLES[i % len(ROLES)]
        system = role
        if learnings_text:
            system += f"\n\n{learnings_text}"
        system += "\n\nAt the end, include: CONFIDENCE: X/10"
        tasks.append(claude(task, model=worker_model, system=system,
                            tools=tools, tools_rw=tools_rw, cwd=cwd))

    results = await asyncio.gather(*tasks)
    successful = [r for r in results if not r.get("error")]
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
