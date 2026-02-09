Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Swarm: Multi-Agent Collective Intelligence System

 Context

 The user wants to build a system where multiple fast LLM agents think
 about the same problem in parallel, a judge scores and synthesizes their
  outputs (집단지성), and learnings are persisted so the system compounds
  knowledge over time and never repeats mistakes (시행착오). This should
 be a simple, elegant Python package — both a CLI tool and an importable
 library.

 Architecture Overview

 User Input → Load Config → Load Relevant Learnings →
   Spawn N Workers (parallel via asyncio.gather) → Collect Responses →
   Judge Scores + Synthesizes + Extracts Learnings → Save to Memory →
   Return Best Answer

 Workers are intentionally isolated (no worker-to-worker communication) —
  this is what produces diversity. The judge does three things in one
 call: scores each response, synthesizes a merged answer, and extracts
 reusable learnings.

 File Structure (3 files, ~300 lines total)

 teammate_revived3/
 ├── swarm/
 │   ├── __init__.py       # ~15 lines  - re-exports public API
 │   ├── core.py           # ~250 lines - ALL logic: config, memory,
 workers, judge, orchestrator
 │   └── __main__.py       # ~60 lines  - CLI entry point (python -m
 swarm)
 ├── pyproject.toml        # Project metadata + single dependency
 (anthropic)
 └── plan.md               # Existing

 Memory files are created at runtime:
 - ~/.swarm/learnings.jsonl — Append-only learning store (what agents
 query)
 - ~/.swarm/runs.jsonl — Optional full run log (audit trail, enabled with
  --save-runs)

 Implementation Plan

 Step 1: Create pyproject.toml

 File: /Users/julius/Documents/teammate_revived3/pyproject.toml

 [project]
 name = "swarm"
 version = "0.1.0"
 requires-python = ">=3.10"
 dependencies = ["anthropic>=0.40.0"]

 [project.scripts]
 swarm = "swarm.__main__:main"

 Single dependency: anthropic SDK (includes httpx, anyio, pydantic).

 Step 2: Create swarm/__init__.py

 File: /Users/julius/Documents/teammate_revived3/swarm/__init__.py

 Re-exports: SwarmConfig, SwarmResult, WorkerResult, JudgeVerdict,
 Learning, SwarmError, Memory, run

 Step 3: Create swarm/core.py — The heart of the system

 File: /Users/julius/Documents/teammate_revived3/swarm/core.py

 3a. Data Structures (dataclasses)

 - SwarmConfig — All knobs: worker_model, judge_model, num_workers
 (default 3), worker_temperature (default 0.9), judge_temperature
 (default 0.1), max_tokens (default 4096), timeout_seconds (default 120),
  memory_path (default ~/.swarm/learnings.jsonl), tags (list[str]),
 optional system_prompt and judge_prompt overrides
 - WorkerResult — worker_id, content, model, input_tokens, output_tokens,
  elapsed_seconds, error (None if success)
 - JudgeVerdict — scores (dict[int, float]), reasoning, synthesis,
 best_worker_id, key_insight, failure_modes (list[str]), token counts
 - Learning — learning_id, timestamp, source_run_id, category (one of:
 mistake/strategy/pattern/constraint), tags, content (1-3 sentences),
 confidence (float), times_confirmed (int), active (bool)
 - SwarmResult — answer, verdict, workers, learnings_used, config, total
 token counts, elapsed_seconds
 - SwarmError — Custom exception for total worker failure

 3b. Memory class (static methods, ~60 lines)

 class Memory:
     @staticmethod
     def load(memory_path, tags=None, limit=10) -> list[Learning]:
         # Read JSONL, deduplicate by learning_id (take last occurrence),
         # filter active=True, optionally filter by tag overlap,
         # return sorted by confidence desc, limited to `limit`
         # Handles: file not found (return []), corrupted lines (skip)

     @staticmethod
     def save(memory_path, prompt, workers, verdict, config) ->
 list[Learning]:
         # Parse LEARNING: lines from verdict using regex
         # Create Learning entries with category, tags, confidence=0.7
         # Append as JSONL lines
         # Check for confirmation of existing learnings (keyword overlap)

     @staticmethod
     def format_for_prompt(learnings) -> str:
         # Returns formatted block like:
         # - [MISTAKE] Always preserve backward compat... (confidence:
 0.85, confirmed 6x)
         # - [STRATEGY] Use parametrized tests... (confidence: 0.92,
 confirmed 11x)

 Key memory design decisions:
 - Tag-based retrieval for v1 (simple, zero-cost, no extra LLM calls)
 - Judge extracts learnings inline as part of scoring (no extra LLM call)
 - Append-only with dedup on read — reader takes last occurrence of each
 learning_id
 - Soft pruning — when times_contradicted > times_confirmed and
 observations >= 3, mark active=false
 - Cap at 5 learnings per prompt to avoid attention dilution
 - Cold start handled — first runs have no learnings, system works fine
 without them

 3c. Worker Logic (~40 lines)

 async def _safe_worker(worker_id, prompt, system, config, client) ->
 WorkerResult:
     # Wraps single API call in try/except — NEVER raises
     # Catches: TimeoutError, anthropic.APIError, any Exception
     # Returns WorkerResult with error field set on failure

 async def run_workers(prompt, system, config, client) ->
 list[WorkerResult]:
     # asyncio.gather(*[_safe_worker(i, ...) for i in
 range(config.num_workers)])
     # Per-worker fault isolation — one failure doesn't kill others

 Workers get temperature=0.9 (diversity) and include learnings in their
 system prompt.

 3d. Judge Logic (~40 lines)

 async def run_judge(prompt, workers, config, client, learnings) ->
 JudgeVerdict:
     # Sends all successful worker responses to judge
     # Judge prompt forces structured output:
     #   SCORES: Worker 0: X/10 - reason ...
     #   BEST_WORKER: id
     #   KEY_INSIGHT: one sentence
     #   FAILURE_MODES: - bullet list
     #   SYNTHESIS: merged answer
     #   LEARNING: [category] content (0-3 learnings)
     # Parses with best-effort regex — lenient, never crashes

 def _parse_verdict(raw, num_workers) -> JudgeVerdict:
     # Regex extraction of scores, best_worker, synthesis, learnings
     # Falls back gracefully if judge deviates from format

 Judge gets temperature=0.1 (consistency).

 3e. Orchestrator (~30 lines)

 async def run(prompt, *, config=None, client=None) -> SwarmResult:
     # 1. Default config if None
     # 2. Create AsyncAnthropic client if None (from ANTHROPIC_API_KEY
 env)
     # 3. Load relevant learnings from memory
     # 4. Build worker system prompt (base + learnings)
     # 5. Run workers in parallel
     # 6. Check quorum (all failed? raise SwarmError. Only 1? skip judge,
  return directly)
     # 7. Run judge on successful workers
     # 8. Save learnings to memory
     # 9. Return SwarmResult

 Step 4: Create swarm/__main__.py — CLI

 File: /Users/julius/Documents/teammate_revived3/swarm/__main__.py

 Uses argparse (stdlib, no extra deps). Key flags:

 python -m swarm "What causes inflation?"
 python -m swarm -n 5 -w claude-haiku-4-5-20251001 -j
 claude-sonnet-4-5-20250929 "Design a REST API"
 python -m swarm --show-workers --show-scores --tags coding,python "Write
  an LRU cache"
 python -m swarm --no-memory --json "Quick question"
 cat problem.txt | python -m swarm --stdin

 Flags: -n/--workers, -w/--worker-model, -j/--judge-model, --worker-temp,
  --judge-temp, --max-tokens, --tags, --memory-path, --no-memory,
 --show-workers, --show-scores, --json, --stdin, --verbose

 Error Handling Strategy

 1. Per-worker fault isolation: _safe_worker never raises. Errors stored
 in WorkerResult.error
 2. Quorum check: If all workers fail → raise SwarmError. If only 1
 succeeds → skip judge, return directly
 3. Judge failure: Fall back to returning longest successful worker
 response with a warning
 4. No custom retries: Rely on anthropic SDK's built-in retry (default 2
 retries on 429/500/503)
 5. Memory corruption: Skip malformed JSONL lines silently, missing file
 returns empty list

 Judge: Pick vs. Synthesize

 Both. The judge scores each response (needed for learning extraction)
 AND synthesizes a merged answer (combines the best elements). This is
 the key design choice — picking alone discards complementary strengths,
 synthesis alone loses traceability for learning.

 Cost Profile (per query)
 ┌─────────────────────────┬─────────┬────────┬────────────────┐
 │         Config          │ Workers │ Judge  │ Total (approx) │
 ├─────────────────────────┼─────────┼────────┼────────────────┤
 │ 3 Haiku + Sonnet judge  │ ~$0.003 │ ~$0.01 │ ~$0.013        │
 ├─────────────────────────┼─────────┼────────┼────────────────┤
 │ 5 Sonnet + Sonnet judge │ ~$0.05  │ ~$0.02 │ ~$0.07         │
 ├─────────────────────────┼─────────┼────────┼────────────────┤
 │ 10 Haiku + Opus judge   │ ~$0.01  │ ~$0.15 │ ~$0.16         │
 └─────────────────────────┴─────────┴────────┴────────────────┘
 Verification Plan

 1. Install: pip install -e . from project root
 2. Set API key: export ANTHROPIC_API_KEY=sk-...
 3. Basic CLI test: python -m swarm "What is 2+2?" -n 3 --show-workers
 --show-scores --verbose
 4. Memory test: Run twice with same --tags math — second run should show
  learnings being loaded
 5. Library test: python -c "import asyncio, swarm;
 print(asyncio.run(swarm.run('Hello')).answer)"
 6. Fault tolerance test: Use an invalid model name for 1 worker config
 to verify graceful degradation
 7. JSON output test: python -m swarm --json "test" | python -m json.tool

 What This Does NOT Include (Intentionally)

 - No task decomposition (Rule #2 from plan.md) — that's a layer on top,
 for v2
 - No iterative refinement loops — can be built by calling run() in a
 loop
 - No web UI — CLI + library is the foundation
 - No vector/embedding-based memory retrieval — tag-based is sufficient
 for v1
 - No streaming — simpler without it, can add later