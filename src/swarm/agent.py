"""Claude Code agent wrapper and red-flagging."""

import asyncio
import json
import re
import time

from .config import TIMEOUT_SECONDS


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
        if len(lines) > 3:
            indented = sum(1 for line in lines if line and (line[0] in " \t"))
            indent_ratio = indented / len(lines)
            if indent_ratio > 0.8:
                return f"mostly code with no explanation ({indent_ratio:.0%} indented lines)"

    # Low self-reported confidence
    conf_match = re.search(r"CONFIDENCE:\s*(\d+(?:\.\d+)?)", content)
    if conf_match and float(conf_match.group(1)) < 3:
        return f"self-reported confidence too low ({conf_match.group(1)}/10)"
    return None
