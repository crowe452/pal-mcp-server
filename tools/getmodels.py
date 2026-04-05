"""
Get Models tool - Live frontier model lookup from OpenRouter.

Returns the single best model from each of: Google, xAI, OpenAI, Anthropic,
plus one wildcard pick from any other provider. Five choices, live data.
No stale registry — hits the OpenRouter API directly (1-hour cache).
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from mcp.types import TextContent

from tools.shared.base_tool import BaseTool
from tools.shared.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)

CACHE_FILE = Path("/tmp/openrouter_models_cache.json")
CACHE_TTL = 3600  # 1 hour

# Providers we always show, in display order
CORE_PROVIDERS = [
    ("google", "Google"),
    ("x-ai", "xAI"),
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
]


class GetModelsTool(BaseTool):
    """Utility tool that fetches the latest frontier models from OpenRouter."""

    def get_name(self) -> str:
        return "getmodels"

    def get_description(self) -> str:
        return (
            "Get the latest frontier AI models from OpenRouter. "
            "Returns 5 picks: the best from Google, xAI, OpenAI, Anthropic, "
            "plus one wildcard from any other provider. Live data, 1-hour cache."
        )

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    def get_annotations(self) -> Optional[dict[str, Any]]:
        return {"readOnlyHint": True}

    def requires_model(self) -> bool:
        return False

    def get_system_prompt(self) -> str:
        return ""

    def get_request_model(self):
        return None

    async def prepare_prompt(self, request) -> str:
        return ""

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """Fetch latest models from OpenRouter API."""
        try:
            models = await self._get_models()
            if not models:
                return [TextContent(type="text", text="Failed to fetch models. Check OPENROUTER_API_KEY.")]

            picks = self._pick_frontier(models)
            output = self._format(picks)
            return [TextContent(type="text", text=output)]

        except Exception as e:
            logger.error(f"Error in getmodels: {e}")
            raise ToolExecutionError(str(e)) from e

    async def _get_models(self) -> list[dict]:
        """Fetch models from OpenRouter, using cache if fresh."""
        # Check cache
        if CACHE_FILE.exists():
            age = time.time() - CACHE_FILE.stat().st_mtime
            if age < CACHE_TTL:
                try:
                    data = json.loads(CACHE_FILE.read_text())
                    return data.get("data", [])
                except (json.JSONDecodeError, KeyError):
                    pass

        # Fetch live
        import urllib.request

        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            return []

        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = resp.read().decode()
                data = json.loads(raw)
                # Cache it
                CACHE_FILE.write_text(raw)
                return data.get("data", [])
        except Exception as e:
            logger.warning(f"OpenRouter fetch failed: {e}")
            # Try stale cache as fallback
            if CACHE_FILE.exists():
                try:
                    return json.loads(CACHE_FILE.read_text()).get("data", [])
                except Exception:
                    pass
            return []

    def _pick_frontier(self, models: list[dict]) -> list[dict]:
        """Pick the single best model from each core provider + one wildcard."""
        # Score each model: prefer large context, recent creation
        def score(m: dict) -> float:
            ctx = m.get("context_length", 0)
            created = m.get("created", 0)
            # Bonus for large context, recency, and "pro" in name
            s = ctx / 100_000
            if created:
                s += created / 1_000_000_000  # epoch bonus
            name = m.get("id", "").lower()
            if "pro" in name and "preview" not in name:
                s += 5
            elif "pro" in name:
                s += 3
            # Penalty for mini/lite/nano/free
            for tag in ["mini", "lite", "nano", "free", "flash"]:
                if tag in name:
                    s -= 10
            return s

        picks = []
        used_ids = set()

        for prefix, label in CORE_PROVIDERS:
            candidates = [
                m for m in models
                if m.get("id", "").startswith(f"{prefix}/")
                and "free" not in m.get("id", "").lower()
            ]
            candidates.sort(key=score, reverse=True)
            if candidates:
                best = candidates[0]
                picks.append({"label": label, "model": best})
                used_ids.add(best["id"])

        # Wildcard: best model NOT from core providers
        core_prefixes = tuple(f"{p}/" for p, _ in CORE_PROVIDERS)
        wildcards = [
            m for m in models
            if not m.get("id", "").startswith(core_prefixes)
            and m.get("id", "") not in used_ids
            and "free" not in m.get("id", "").lower()
        ]
        wildcards.sort(key=score, reverse=True)
        if wildcards:
            picks.append({"label": "Wildcard", "model": wildcards[0]})

        return picks

    def _format(self, picks: list[dict]) -> str:
        """Format picks as a clean table."""
        lines = ["**Frontier Models (live from OpenRouter)**", ""]
        lines.append(f"{'#':<3} {'Provider':<12} {'Model ID':<50} {'Context':>10}")
        lines.append("-" * 78)

        for i, pick in enumerate(picks, 1):
            m = pick["model"]
            mid = m.get("id", "?")
            ctx = m.get("context_length", 0)
            ctx_str = f"{ctx:,}" if ctx else "?"
            lines.append(f"{i:<3} {pick['label']:<12} {mid:<50} {ctx_str:>10}")

        lines.append("")
        lines.append("Pick a number or name a model directly.")
        return "\n".join(lines)
