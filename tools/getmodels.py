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

    # Models that are NOT general-purpose reasoning/coding models
    EXCLUDE_PATTERNS = [
        "lyria", "imagen", "music", "audio", "tts", "stt", "whisper",
        "embed", "moderation", "dall-e", "stable-diffusion",
        "auto", "router", "openrouter/",  # meta/routing models
        "vision-only", "ocr",
    ]

    # Model family keywords that indicate flagship reasoning models
    FLAGSHIP_KEYWORDS = {
        "google": ["gemini"],
        "x-ai": ["grok"],
        "openai": ["gpt", "o3", "o4"],
        "anthropic": ["claude"],
    }

    def _pick_frontier(self, models: list[dict]) -> list[dict]:
        """Pick the single best model from each core provider + one wildcard."""

        def is_excluded(mid: str) -> bool:
            mid_lower = mid.lower()
            return any(pat in mid_lower for pat in self.EXCLUDE_PATTERNS)

        def is_flagship_family(mid: str, provider_prefix: str) -> bool:
            """Check if model belongs to the flagship family for this provider."""
            mid_lower = mid.lower()
            keywords = self.FLAGSHIP_KEYWORDS.get(provider_prefix, [])
            return any(kw in mid_lower for kw in keywords)

        def score(m: dict) -> float:
            ctx = m.get("context_length", 0)
            created = m.get("created", 0)
            name = m.get("id", "").lower()
            s = 0.0

            # Context window (normalized)
            s += ctx / 100_000

            # Recency bonus
            if created:
                s += created / 1_000_000_000

            # "opus" is flagship for Anthropic
            if "opus" in name:
                s += 20

            # "pro" bonus (but not "preview" penalty — previews are often the latest)
            if "pro" in name:
                s += 10

            # Penalty for mini/lite/nano/free/flash variants
            for tag in ["mini", "lite", "nano", "free", "flash"]:
                if tag in name:
                    s -= 20

            # Penalty for old version numbers when newer exist
            # e.g., gpt-5.1 < gpt-5.4, gemini-2.5 < gemini-3.1
            # Extract version-like numbers
            import re
            versions = re.findall(r"(\d+\.?\d*)", name.split("/")[-1])
            if versions:
                try:
                    max_ver = max(float(v) for v in versions)
                    s += max_ver * 2  # higher version = better
                except ValueError:
                    pass

            return s

        picks = []
        used_ids = set()

        for prefix, label in CORE_PROVIDERS:
            candidates = [
                m for m in models
                if m.get("id", "").startswith(f"{prefix}/")
                and not is_excluded(m.get("id", ""))
                and is_flagship_family(m.get("id", ""), prefix)
            ]
            candidates.sort(key=score, reverse=True)
            if candidates:
                best = candidates[0]
                picks.append({"label": label, "model": best})
                used_ids.add(best["id"])

        # Rest of field: top 5 from other providers
        core_prefixes = tuple(f"{p}/" for p, _ in CORE_PROVIDERS)
        others = [
            m for m in models
            if not m.get("id", "").startswith(core_prefixes)
            and m.get("id", "") not in used_ids
            and not is_excluded(m.get("id", ""))
        ]
        others.sort(key=score, reverse=True)
        # Deduplicate by provider prefix (one per provider)
        seen_providers = set()
        for m in others:
            provider = m.get("id", "").split("/")[0]
            if provider not in seen_providers:
                seen_providers.add(provider)
                picks.append({"label": provider, "model": m, "is_other": True})
                if len(seen_providers) >= 5:
                    break

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
