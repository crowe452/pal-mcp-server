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

    # Only exclude models that are NOT general-purpose text reasoning models.
    # We do NOT exclude tiers (sonnet, haiku, flash, mini) — just non-text models.
    EXCLUDE_SUBSTRINGS = [
        "lyria", "imagen", "music", "audio-", "-tts", "-stt", "whisper",
        "-embed", "moderation", "dall-e", "stable-diffusion",
        "/auto", "openrouter/",
        "-image", "-vision",
        "-ocr",
        ":free",  # free tier variants on OpenRouter
        "gemma",  # open weights family, not Gemini
    ]
    EXCLUDE_REGEX = []  # none needed currently

    def _pick_frontier(self, models: list[dict]) -> list[dict]:
        """Pick the most expensive model from each core provider + top 5 others.

        One rule: highest completion price wins. Providers price their best
        models highest. OpenRouter maintains the pricing. We just sort by it.
        Tiebreak: shortest model ID (base model over variant suffixes).
        """
        import re

        def is_excluded(mid: str) -> bool:
            mid_lower = mid.lower()
            if any(pat in mid_lower for pat in self.EXCLUDE_SUBSTRINGS):
                return True
            if any(re.search(pat, mid_lower) for pat in self.EXCLUDE_REGEX):
                return True
            return False

        def completion_price(m: dict) -> float:
            try:
                return float(m.get("pricing", {}).get("completion", "0"))
            except (ValueError, TypeError):
                return 0.0

        picks = []
        used_ids = set()

        def version_family(mid: str) -> str:
            """Extract version family: 'openai/gpt-5.4-pro' -> '5.4'"""
            name = mid.split("/")[-1]
            match = re.search(r"(\d+\.?\d*)", name)
            return match.group(1) if match else "0"

        for prefix, label in CORE_PROVIDERS:
            candidates = [
                m for m in models
                if m.get("id", "").startswith(f"{prefix}/")
                and not is_excluded(m.get("id", ""))
            ]
            # Group by version family, pick newest family, then most expensive in it
            from collections import defaultdict
            families = defaultdict(list)
            for c in candidates:
                families[version_family(c["id"])].append(c)

            # Newest family first (by max created timestamp)
            sorted_fams = sorted(
                families.items(),
                key=lambda kv: max(m.get("created", 0) for m in kv[1]),
                reverse=True,
            )
            if sorted_fams:
                # Within newest family: most expensive, then shortest ID
                fam = sorted_fams[0][1]
                fam.sort(key=lambda m: (-completion_price(m), len(m.get("id", ""))))
                best = fam[0]
                picks.append({"label": label, "model": best})
                used_ids.add(best["id"])

        # Rest of field: best model from each non-core provider (same logic: newest family + most expensive)
        core_prefixes = tuple(f"{p}/" for p, _ in CORE_PROVIDERS)
        others = [
            m for m in models
            if not m.get("id", "").startswith(core_prefixes)
            and m.get("id", "") not in used_ids
            and not is_excluded(m.get("id", ""))
        ]

        # Group by provider, pick the best from each using same family+price logic
        from collections import defaultdict
        by_provider = defaultdict(list)
        for m in others:
            provider = m.get("id", "").split("/")[0]
            by_provider[provider].append(m)

        provider_picks = []
        for provider, provider_models in by_provider.items():
            # Group by version family within this provider
            families = defaultdict(list)
            for m in provider_models:
                families[version_family(m["id"])].append(m)

            # Newest family first
            sorted_fams = sorted(
                families.items(),
                key=lambda kv: max(m.get("created", 0) for m in kv[1]),
                reverse=True,
            )
            if sorted_fams:
                fam = sorted_fams[0][1]
                fam.sort(key=lambda m: (-completion_price(m), len(m.get("id", ""))))
                best = fam[0]
                provider_picks.append({"label": provider, "model": best, "is_other": True})

        # Sort all provider picks by price descending (best providers float up)
        provider_picks.sort(key=lambda p: -completion_price(p["model"]))
        picks.extend(provider_picks[:5])

        return picks

    def _format(self, picks: list[dict]) -> str:
        """Format picks as frontier labs + others."""
        frontier = [p for p in picks if not p.get("is_other")]
        others = [p for p in picks if p.get("is_other")]

        lines = ["**Frontier Labs (live from OpenRouter)**", ""]
        lines.append(f"{'#':<3} {'Provider':<12} {'Model ID':<50} {'Context':>10}")
        lines.append("-" * 78)

        for i, pick in enumerate(frontier, 1):
            m = pick["model"]
            mid = m.get("id", "?")
            ctx = m.get("context_length", 0)
            ctx_str = f"{ctx:,}" if ctx else "?"
            lines.append(f"{i:<3} {pick['label']:<12} {mid:<50} {ctx_str:>10}")

        if others:
            lines.append("")
            lines.append("**Also Available**")
            lines.append("-" * 78)
            for i, pick in enumerate(others, len(frontier) + 1):
                m = pick["model"]
                mid = m.get("id", "?")
                ctx = m.get("context_length", 0)
                ctx_str = f"{ctx:,}" if ctx else "?"
                lines.append(f"{i:<3} {pick['label']:<12} {mid:<50} {ctx_str:>10}")

        lines.append("")
        lines.append("Pick a number or name a model directly.")
        return "\n".join(lines)
