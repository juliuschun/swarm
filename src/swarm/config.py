"""Configuration constants for swarm."""

from pathlib import Path

# Default models (can be overridden by CLI flags)
DEFAULT_MODELS = {
    "planner":   "sonnet",   # decomposes tasks into atomic steps
    "worker":    "haiku",    # executes steps — cheap, voting makes it reliable
    "composer":  "sonnet",   # merges step results into final answer
    "verifier":  "sonnet",   # checks result against original task
}

MODELS = DEFAULT_MODELS.copy()

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
