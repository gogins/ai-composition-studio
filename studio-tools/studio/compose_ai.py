"""Rewrite a staged .csd from a user prompt using an LLM API.

Used by the local compose API when STUDIO_COMPOSE_HOOK is not set.
Requires ANTHROPIC_API_KEY (or STUDIO_COMPOSE_API_KEY with an Anthropic key).
Optional: STUDIO_COMPOSE_MODEL (default claude-sonnet-4-5).
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Tuple

from . import naming, staging
from .paths import StudioError

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_DEFAULT_MODEL = "claude-sonnet-4-5"
_CSD_BLOCK = re.compile(
    r"<CsoundSynthesizer>[\s\S]*?</CsoundSynthesizer>", re.IGNORECASE
)

_SYSTEM = """\
You are Michael Gogins's Csound composition assistant for the AI Composition Studio.
You rewrite a complete Csound .csd to satisfy the user's prompt.

Rules:
- Output ONE complete Csound file only: <CsoundSynthesizer> … </CsoundSynthesizer>.
- No markdown fences, no commentary before or after the .csd.
- Keep <CsInstruments> and <CsScore>. Set 0dbfs = 1. Prefer sr = 48000, ksmps = 32, nchnls = 2.
- Do not put -odac or -o file in <CsOptions>; use -d only if needed.
- P-fields: p1 instrument, p2 time, p3 duration, p4 MIDI key, p5 MIDI velocity,
  p6 depth, p7 pan, p8 height, p9 phase, p10 pitch-class set, p11 homogeneity.
- Realize the prompt musically. If asked for a duration (e.g. 2 minutes), the
  score must last that long (last events ending near that time).
- Keep the stylistic character of the previous .csd when the prompt is a revision,
  unless the prompt asks for something different.
"""


class ComposeAIError(StudioError):
    """The language-model rewrite failed."""


def ai_configured() -> bool:
    return bool(_api_key())


def _api_key() -> str:
    return (
        (os.environ.get("STUDIO_COMPOSE_API_KEY") or "").strip()
        or (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    )


def model_name() -> str:
    return (os.environ.get("STUDIO_COMPOSE_MODEL") or _DEFAULT_MODEL).strip()


def rewrite_csd(staging_dir: Path, prompt: str) -> Dict[str, Any]:
    """Call the LLM and replace <name>.csd in the staging directory."""
    meta = staging.load_meta(staging_dir)
    name = meta["composition"]
    csd_path = Path(staging_dir) / naming.csd_name(name)
    previous = csd_path.read_text(encoding="utf-8") if csd_path.is_file() else ""
    key = _api_key()
    if not key:
        raise ComposeAIError(
            "no AI key: set ANTHROPIC_API_KEY (or STUDIO_COMPOSE_API_KEY) in the "
            "environment that runs studio serve, or set STUDIO_COMPOSE_HOOK"
        )
    text, usage = _anthropic_complete(key, prompt, previous, name)
    csd = _extract_csd(text)
    problems = []
    from . import render as render_module

    problems = render_module.structural_problems(csd)
    if problems:
        raise ComposeAIError(
            "the model did not return a complete .csd: " + "; ".join(problems)
        )
    csd_path.write_text(csd if csd.endswith("\n") else csd + "\n", encoding="utf-8")
    return {
        "model": model_name(),
        "provider": "anthropic",
        "bytes": len(csd.encode("utf-8")),
        "usage": usage,
    }


def _extract_csd(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:csound|csd)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    match = _CSD_BLOCK.search(text)
    if not match:
        raise ComposeAIError(
            "the model response did not contain a <CsoundSynthesizer> … </CsoundSynthesizer> block"
        )
    return match.group(0).strip() + "\n"


def _anthropic_complete(
    key: str, prompt: str, previous_csd: str, name: str
) -> Tuple[str, Dict[str, Any]]:
    user = (
        f"Composition name: {name}\n\n"
        f"User prompt (follow this):\n{prompt.strip()}\n\n"
    )
    if previous_csd.strip():
        user += (
            "Previous .csd (revise this; keep what still fits the prompt):\n"
            f"{previous_csd.strip()}\n"
        )
    else:
        user += "There is no previous .csd; write a new complete piece.\n"
    user += "\nReturn only the complete .csd file."

    body = {
        "model": model_name(),
        "max_tokens": int(os.environ.get("STUDIO_COMPOSE_MAX_TOKENS", "16000")),
        "system": _SYSTEM,
        "messages": [{"role": "user", "content": user}],
    }
    request = urllib.request.Request(
        _ANTHROPIC_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=int(os.environ.get("STUDIO_COMPOSE_TIMEOUT", "600"))
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:800]
        raise ComposeAIError(f"Anthropic HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ComposeAIError(f"could not reach Anthropic: {error}") from error

    parts = []
    for block in payload.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text") or "")
    text = "".join(parts).strip()
    if not text:
        raise ComposeAIError("Anthropic returned no text")
    usage = payload.get("usage") or {}
    return text, usage
