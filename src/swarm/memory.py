"""Memory system: learnings and session management."""

import json
import re
import sys
import uuid
from datetime import datetime, timezone

from .config import MEMORY_DIR, MEMORY_FILE, SESSIONS_FILE


def recall(tags: list[str] | None = None, limit: int = 5) -> list[dict]:
    """Load learnings from memory, optionally filtered by tags."""
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


def learn(text: str, tags: list[str] | None = None, verbose: bool = False) -> list[dict]:
    """Extract and save learnings from text."""
    pattern = r"LEARNING:\s*\[(mistake|strategy|pattern|constraint)\]\s*(.+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    if not matches:
        if verbose:
            print(f"[learn] No LEARNING: patterns found in text (length={len(text)})", file=sys.stderr)
        return []
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for category, content in matches:
        content = content.strip()
        if len(content) < 15 or len(content) > 500:
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
    """Format learnings for injection into agent prompts."""
    if not learnings:
        return ""
    lines = ["## LESSONS FROM PAST RUNS (apply these):"]
    for l in learnings:
        cat = l.get("category", "insight").upper()
        conf = l.get("confidence", 0.7)
        confirmed = l.get("times_confirmed", 0)
        lines.append(f"- [{cat}] {l['content']} (confidence: {conf:.2f}, confirmed {confirmed}x)")
    return "\n".join(lines)


def save_session(role: str, session_id: str, prompt: str,
                 confidence: float, tags: list[str] | None = None):
    """Save a worker session for later resume."""
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
    """Load recent sessions."""
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
    """Get the highest-confidence session from the last run."""
    sessions = load_sessions(20)
    if not sessions:
        return None
    latest_ts = sessions[-1]["ts"][:16]
    last_run = [s for s in sessions if s["ts"][:16] == latest_ts]
    return max(last_run, key=lambda s: s.get("confidence", 0)) if last_run else None
