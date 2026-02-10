"""Configuration constants for swarm."""

from pathlib import Path

# Default models (can be overridden by CLI flags)
DEFAULT_MODELS = {
    "planner":   "opus",     # decomposes tasks — needs best judgment
    "worker":    "sonnet",   # executes steps — voting makes it reliable
    "judge":     "opus",     # picks best response — needs best judgment
    "composer":  "opus",     # merges step results — needs coherent synthesis
    "verifier":  "opus",     # checks result — needs best judgment to catch issues
}

MODELS = DEFAULT_MODELS.copy()

K_START = 2               # initial K — escalate if no winner
K_MAX = 5                 # max K before judge fallback
MAX_SAMPLES = 12          # max vote samples per step before judge
BATCH_SIZE = 3            # parallel workers per vote round
MAX_LOOPS = 3             # max verify→re-plan loops
TIMEOUT_SECONDS = 120
CHECKPOINT_INTERVAL = 10  # checkpoint every N steps
MEMORY_DIR = Path.home() / ".swarm"
MEMORY_FILE = MEMORY_DIR / "learnings.jsonl"
SESSIONS_FILE = MEMORY_DIR / "sessions.jsonl"
CHECKPOINTS_DIR = MEMORY_DIR / "checkpoints"

ROLES = [
    "You are a pragmatist. Find the simplest working solution.",
    "You are a skeptic. Find what could go wrong. Identify edge cases.",
    "You are an innovator. Find unconventional approaches others would miss.",
    "You are a systems thinker. Consider second-order effects and feedback loops.",
    "You are a historian. What has worked or failed before in similar situations?",
    "You are a contrarian. Argue against the obvious answer. Find the hidden truth.",
]
