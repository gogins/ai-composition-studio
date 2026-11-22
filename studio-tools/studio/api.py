"""Compose API used by the local player (and later by a hosted backend).

The player posts a prompt; this module stages a revision (or a new
composition), records the prompt, rewrites the .csd with an LLM (or
STUDIO_COMPOSE_HOOK), then renders, accepts and publishes.

AI rewrite uses ANTHROPIC_API_KEY when STUDIO_COMPOSE_HOOK is unset.
Set STUDIO_COMPOSE_ALLOW_UNCHANGED=1 only for tests that keep the prior .csd.
"""
from __future__ import annotations

import os
import shlex
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from . import compose_ai, naming, publish as publish_module, render as render_module, staging, versioning
from .paths import Studio, StudioError

API_VERSION = "1"
COMPOSE_LOCK = threading.Lock()

# Short stub used only when creating a composition with ALLOW_UNCHANGED and no AI.
_STUB_CSD = """\
<CsoundSynthesizer>
<CsOptions>
-d
</CsOptions>
<CsInstruments>
sr = 48000
ksmps = 32
nchnls = 2
0dbfs = 1
instr 1
  a1 poscil 0.2, cpsmidinn(p4)
  aenv linen a1, 0.01, p3, 0.05
  out aenv, aenv
endin
</CsInstruments>
<CsScore>
i 1 0 1.5 60
</CsScore>
</CsoundSynthesizer>
"""


class ComposeError(StudioError):
    """A compose request could not be completed."""


def status() -> Dict[str, Any]:
    hook = (os.environ.get("STUDIO_COMPOSE_HOOK") or "").strip()
    ai = compose_ai.ai_configured()
    if hook:
        message = "Local compose is available; STUDIO_COMPOSE_HOOK will rewrite the .csd."
    elif ai:
        message = (
            f"Local compose is available; prompts rewrite the .csd with {compose_ai.model_name()}, "
            "then render, accept and publish."
        )
    else:
        message = (
            "Local compose API is up, but no AI is configured. Set ANTHROPIC_API_KEY "
            "in the terminal that runs studio serve (or STUDIO_COMPOSE_HOOK)."
        )
    return {
        "ok": True,
        "api": API_VERSION,
        "mode": "local",
        "compose_hook": bool(hook),
        "compose_ai": ai,
        "compose_model": compose_ai.model_name() if ai else None,
        "message": message,
    }


def compose(
    studio: Studio,
    *,
    prompt: str,
    composition: Optional[str] = None,
    name: Optional[str] = None,
    title: Optional[str] = None,
    allow_unplayable: bool = False,
) -> Dict[str, Any]:
    """Record a prompt, rewrite the .csd, and publish a new version. Thread-safe."""
    text = (prompt or "").strip()
    if not text:
        raise ComposeError("prompt is empty")
    with COMPOSE_LOCK:
        return _compose(
            studio,
            prompt=text,
            composition=composition,
            name=name,
            title=title,
            allow_unplayable=allow_unplayable,
        )


def _allow_unchanged() -> bool:
    return os.environ.get("STUDIO_COMPOSE_ALLOW_UNCHANGED", "").strip() in ("1", "true", "yes")


def _compose(
    studio: Studio,
    *,
    prompt: str,
    composition: Optional[str],
    name: Optional[str],
    title: Optional[str],
    allow_unplayable: bool,
) -> Dict[str, Any]:
    if composition:
        naming.validate_composition_name(composition)
        if not studio.composition_dir(composition).is_dir():
            raise ComposeError(f"no composition named {composition!r}")
        staging_dir = staging.stage_revision(studio, composition)
        kind = "revision"
        target = composition
    else:
        if not name:
            raise ComposeError("give composition (to revise) or name (to create)")
        naming.validate_composition_name(name)
        staging_dir = staging.new_composition(studio, name, title)
        kind = "new"
        target = name

    ai_info: Optional[Dict[str, Any]] = None
    try:
        prompt_path = staging.add_prompt(staging_dir, prompt, "user")
        before = _csd_fingerprint(staging_dir, target)
        ai_info = _rewrite_from_prompt(staging_dir, prompt, kind=kind)
        csd = staging_dir / naming.csd_name(target)
        if not csd.is_file():
            raise ComposeError(
                f"{csd.name} is missing after compose; the AI or hook must write the .csd"
            )
        after = _csd_fingerprint(staging_dir, target)
        if kind == "revision" and before == after and not _allow_unchanged():
            raise ComposeError(
                "the .csd did not change after the prompt. Check the AI rewrite / compose hook."
            )
        render_module.render(
            staging_dir,
            allow_unplayable=allow_unplayable,
            check_browser=os.environ.get("STUDIO_SKIP_BROWSER_CHECK") != "1",
        )
        model = (ai_info or {}).get("model") or os.environ.get("STUDIO_COMPOSE_MODEL") or "player"
        agent = os.environ.get("STUDIO_COMPOSE_AGENT") or (
            "studio-compose-ai" if ai_info else "studio-serve"
        )
        accepted = versioning.accept(
            studio,
            staging_dir,
            ai_model=model,
            ai_agent=agent,
            title=title,
            notes="Submitted through the studio compose API (player).",
        )
        staging_dir = None  # accept removes the staging area
        published = publish_module.publish(studio, [target])
        ordinal = accepted.get("ordinal")
        return {
            "ok": True,
            "kind": kind,
            "composition": target,
            "title": title or target,
            "ordinal": ordinal,
            "version": "current",
            "archived_as": accepted.get("archived_as"),
            "prompt_file": prompt_path.name,
            "published": bool(published.get("written")),
            "ai": ai_info,
            "message": (
                f"Published {target} ordinal {ordinal} (now current)."
                if kind == "revision"
                else f"Created and published {target}."
            ),
        }
    except Exception:
        if staging_dir is not None and staging_dir.is_dir():
            try:
                staging.discard(studio, staging_dir)
            except Exception:
                pass
        raise


def _csd_fingerprint(staging_dir: Path, name: str) -> Optional[str]:
    path = Path(staging_dir) / naming.csd_name(name)
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _rewrite_from_prompt(staging_dir: Path, prompt: str, *, kind: str) -> Optional[Dict[str, Any]]:
    """Hook first, else built-in Anthropic rewrite, else unchanged/stub for tests."""
    raw = (os.environ.get("STUDIO_COMPOSE_HOOK") or "").strip()
    if raw:
        request = staging_dir / ".compose-request.txt"
        request.write_text(prompt if prompt.endswith("\n") else prompt + "\n", encoding="utf-8")
        staging.run_command(
            staging_dir, shlex.split(raw), timeout=int(os.environ.get("STUDIO_COMPOSE_TIMEOUT", "3600"))
        )
        return {"model": "hook", "provider": "STUDIO_COMPOSE_HOOK"}

    if compose_ai.ai_configured():
        return compose_ai.rewrite_csd(staging_dir, prompt)

    if _allow_unchanged():
        csd = staging_dir / naming.csd_name(staging.load_meta(staging_dir)["composition"])
        if kind == "new" and not csd.is_file():
            csd.write_text(_STUB_CSD, encoding="utf-8")
        return None

    raise ComposeError(
        "Submit needs an AI rewrite. In the same terminal as studio serve, export "
        "ANTHROPIC_API_KEY=… (or set STUDIO_COMPOSE_HOOK to a command that writes the .csd), "
        "then restart studio serve and reload the player."
    )
