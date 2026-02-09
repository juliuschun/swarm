All 10 agents have returned. Let me now do what your system is designed to do — synthesize their collective intelligence into something better than any one of them produced alone.

  ---
  Synthesis: The Collective Intelligence of 10 Agents

  The Cynical Truth (Agents 2, 7, 10 agree)

  80% of tasks don't need multi-agent. A single well-prompted Opus is cheaper ($0.035) than 5 Haiku + 1 Sonnet judge ($0.064). The real win isn't parallelism — it's knowing when to use it
  and compounding learnings.

  Your 3 principles are sound, but the order matters:
  ┌──────────┬───────────────────────────────────────────┬───────────────────────────────────────────────────────────┐
  │ Priority │                 Principle                 │                            Why                            │
  ├──────────┼───────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ #1       │ Record & compound learnings (your rule 3) │ A single agent with memory beats 10 amnesiac agents       │
  ├──────────┼───────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ #2       │ Intelligent decomposition (your rule 2)   │ Only split when ROI > 1.2x                                │
  ├──────────┼───────────────────────────────────────────┼───────────────────────────────────────────────────────────┤
  │ #3       │ Parallel crowd intelligence (your rule 1) │ Use sparingly; enforce diversity or you get mode collapse │
  └──────────┴───────────────────────────────────────────┴───────────────────────────────────────────────────────────┘
  The Architecture: 5 Unix Pipes

  Every agent converged on the same core insight: composable bash commands with NDJSON. Here's the distilled version — 5 commands, not 7:

  echo "task" | swarm-think 3 | swarm-judge | swarm-do | swarm-learn
  ┌───────────────┬───────────────────────────────────────────────┬────────────────┬───────────────────────┐
  │    Command    │                     Does                      │     Input      │        Output         │
  ├───────────────┼───────────────────────────────────────────────┼────────────────┼───────────────────────┤
  │ swarm-think N │ Spawn N diverse agents in parallel            │ Goal text      │ NDJSON answers        │
  ├───────────────┼───────────────────────────────────────────────┼────────────────┼───────────────────────┤
  │ swarm-judge   │ Score + pick/merge best answer                │ NDJSON answers │ Single winning answer │
  ├───────────────┼───────────────────────────────────────────────┼────────────────┼───────────────────────┤
  │ swarm-do      │ Execute the decision (or decompose → recurse) │ Decision       │ Result + outcome      │
  ├───────────────┼───────────────────────────────────────────────┼────────────────┼───────────────────────┤
  │ swarm-learn   │ Extract learnings, persist to memory          │ Result         │ Learning entry        │
  ├───────────────┼───────────────────────────────────────────────┼────────────────┼───────────────────────┤
  │ swarm-recall  │ Inject past learnings into prompt             │ Query text     │ Relevant learnings    │
  └───────────────┴───────────────────────────────────────────────┴────────────────┴───────────────────────┘
  The key insight from Agent 6 (Unix): everything flows as NDJSON through pipes. Each command does one thing. They compose.

  The 3 Critical Design Decisions

  1. Diversity via Prompt Roles, NOT Temperature (Agents 2, 8 agree)

  Mode collapse is the #1 killer. All agents using the same model will converge to the same answer. The fix costs nothing:

  # BAD: 3 identical agents with different temperature
  # GOOD: 3 agents with different ROLES
  ROLES=(
    "You are a pragmatist. Find the simplest working solution."
    "You are a skeptic. Find what could go wrong."
    "You are an innovator. Find unconventional approaches."
  )

  Agent 8 confirmed this with ML research: prompt diversity is the LLM equivalent of bagging's feature subsets. It gives 15-25% quality improvement vs temperature variation's ~2%.

  2. N=3, Not 5 or 10 (Agents 3, 7 agree)

  The cost analysis is brutal:
  - N=3 + 1 judge = 4 API calls, captures 90% of ensemble benefit
  - N=5 + 1 judge = 6 API calls, captures 95% (+5% for +50% cost)
  - N=10 = waste

  3. Learn BEFORE You Think (Agents 2, 4, 10 agree)

  The single most impactful thing: before spawning any agent, query past learnings and inject them into the prompt. A single agent with context beats 3 agents without.

  # The real pipeline:
  CONTEXT=$(swarm-recall "auth bugs")
  echo "fix auth bug" | swarm-think 3 --context "$CONTEXT" | swarm-judge | swarm-do | swarm-learn

  The Feedback Loop (Agent 9's Key Contribution)

  Don't just record learnings — measure if the system is getting better:

  TASK → RECALL → THINK → JUDGE → DO → EVALUATE → LEARN
    ↑                                                  |
    └──────────── weights adjusted ────────────────────┘

  Three safety mechanisms prevent the system from poisoning itself:
  1. Confidence gate: Don't learn from uncertain outcomes (conf < 0.7)
  2. Exploration: Always 20% random agent selection (prevents collapse)
  3. Staleness reset: Clear learnings older than 30 days

  What NOT To Build (The Anti-Features)

  Every cynical agent agreed on these:

  - No database — Markdown + grep is sufficient and git-friendly
  - No complex graph/state machine — Bash pipes are the state machine
  - No manager agent that "intelligently delegates" — Use hardcoded rules
  - No iterative debate rounds — Just parallel vote (13% improvement vs 2.8% for debate, per research)
  - No ensemble-of-judges — One judge is enough; judging the judge is a trap
  - No decomposition by default — Only when time_saved / overhead > 1.2

  The Minimal Implementation

  The entire system is ~200 lines of bash across 5 files:

  swarm-recall   # 30 lines: grep .swarm-memory for relevant learnings
  swarm-think    # 50 lines: spawn N agents with diverse roles, collect NDJSON
  swarm-judge    # 40 lines: feed all answers to one Sonnet, pick winner
  swarm-do       # 30 lines: execute decision, capture outcome
  swarm-learn    # 50 lines: extract learning entry, append to memory

  Memory format (from Agent 4, simplified):

  ## L-20260209-001: CLI flags must be kebab-case
  - **Tags**: #cli #spawn #critical
  - **Outcome**: FAILURE→SUCCESS
  - **What happened**: Used --dangerouslySkipPermissions, failed. Fix: --dangerously-skip-permissions
  - **Pattern**: Always use kebab-case for Claude CLI flags

  Recommended Next Step

  Don't build all 5 commands at once. Build them in this order:

  1. swarm-learn + swarm-recall first — memory is your highest-ROI investment
  2. swarm-think + swarm-judge second — the parallel consensus engine
  3. swarm-do last — execution is the part you already have working