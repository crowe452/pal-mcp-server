"""
Get Models tool - Live frontier model lookup from OpenRouter.

Top 4 is strict: positions 1-3 are the three non-Anthropic frontier labs
(Google, xAI, OpenAI) in that fixed order. Position 4 is a WILDCARD chosen
from any OTHER provider (not the top 3, not Anthropic) by the same
newest-family + most-expensive rule used for the core picks.

Why Anthropic is excluded from the top 4: this tool exists so Claude Code
can reach models OUTSIDE Anthropic. Claude Code already runs Claude, so
listing Anthropic in a premium slot is dead weight. Anthropic still shows
up in "Also Available" so it's visible, just not recommended.

The top-4 table has a strict shape (rank / provider / name / slug / context
/ $/M in / $/M out) so callers can rely on it. The "Also Available"
section stays flexible.

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

# Top-3 core providers, in fixed display order. Anthropic is deliberately
# NOT in this list — see module docstring.
CORE_PROVIDERS = [
    ("google", "Google"),
    ("x-ai", "xAI"),
    ("openai", "OpenAI"),
]

# Providers excluded from the wildcard slot (position 4). Top-3 labs
# already claim their own slots, and Anthropic is banned from top-4
# entirely.
WILDCARD_EXCLUDED_PREFIXES = {"google", "x-ai", "openai", "anthropic"}


class GetModelsTool(BaseTool):
    """Utility tool that fetches the latest frontier models from OpenRouter."""

    def get_name(self) -> str:
        return "getmodels"

    def get_description(self) -> str:
        return (
            "Get the latest frontier AI models from OpenRouter. "
            "Top 4: Google, xAI, OpenAI, plus one wildcard from any other "
            "non-Anthropic provider. Each top-4 row includes model slug and "
            "pricing so callers can invoke without guessing. Also Available "
            "lists the rest of the field. Live data, 1-hour cache."
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

    @staticmethod
    def _completion_price(m: dict) -> float:
        try:
            return float(m.get("pricing", {}).get("completion", "0"))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _prompt_price(m: dict) -> float:
        try:
            return float(m.get("pricing", {}).get("prompt", "0"))
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _version_family(mid: str) -> str:
        """Extract version family: 'openai/gpt-5.4-pro' -> '5.4'."""
        import re
        name = mid.split("/")[-1]
        match = re.search(r"(\d+\.?\d*)", name)
        return match.group(1) if match else "0"

    def _is_excluded(self, mid: str) -> bool:
        import re
        mid_lower = mid.lower()
        if any(pat in mid_lower for pat in self.EXCLUDE_SUBSTRINGS):
            return True
        if any(re.search(pat, mid_lower) for pat in self.EXCLUDE_REGEX):
            return True
        return False

    def _best_in_pool(self, pool: list[dict]) -> Optional[dict]:
        """Pick the flagship model from a candidate pool.

        Rule: group by version family, take newest family (by max created
        timestamp), then within that family pick highest completion price,
        tiebreak on shortest model id (prefers base model over suffixed
        variants). Returns None if pool is empty.
        """
        from collections import defaultdict

        if not pool:
            return None
        families = defaultdict(list)
        for c in pool:
            families[self._version_family(c.get("id", ""))].append(c)
        sorted_fams = sorted(
            families.items(),
            key=lambda kv: max(m.get("created", 0) for m in kv[1]),
            reverse=True,
        )
        if not sorted_fams:
            return None
        fam = sorted_fams[0][1]
        fam.sort(key=lambda m: (-self._completion_price(m), len(m.get("id", ""))))
        return fam[0]

    def _pick_frontier(self, models: list[dict]) -> list[dict]:
        """Pick the top-4 + the rest of the field.

        Top 4 is strict: Google, xAI, OpenAI, then one wildcard from any
        other non-Anthropic provider. The wildcard is whichever non-core
        non-Anthropic provider has the highest-priced flagship. Anthropic
        is deliberately excluded from the top-4 pool.

        The rest of the field ("Also Available") includes Anthropic and
        every other provider with an eligible flagship, sorted by price
        descending, capped at 5.
        """
        from collections import defaultdict

        picks: list[dict] = []
        used_ids: set[str] = set()

        # Positions 1-3: the three core frontier labs in fixed order
        for prefix, label in CORE_PROVIDERS:
            pool = [
                m for m in models
                if m.get("id", "").startswith(f"{prefix}/")
                and not self._is_excluded(m.get("id", ""))
            ]
            best = self._best_in_pool(pool)
            if best is not None:
                picks.append({"label": label, "model": best, "slot": "core"})
                used_ids.add(best["id"])
            else:
                # No eligible model — leave a placeholder slot so the
                # table still has 4 rows and callers can see the gap.
                picks.append({"label": label, "model": None, "slot": "core"})

        # Build provider → flagship map for everything outside the top-3
        # (includes Anthropic; we filter Anthropic out later just for the
        # wildcard slot).
        core_prefixes = tuple(f"{p}/" for p, _ in CORE_PROVIDERS)
        by_provider: dict[str, list[dict]] = defaultdict(list)
        for m in models:
            mid = m.get("id", "")
            if mid in used_ids:
                continue
            if mid.startswith(core_prefixes):
                continue
            if self._is_excluded(mid):
                continue
            provider = mid.split("/")[0]
            by_provider[provider].append(m)

        provider_flagships: list[dict] = []
        for provider, pool in by_provider.items():
            best = self._best_in_pool(pool)
            if best is not None:
                provider_flagships.append({"label": provider, "model": best})

        # Position 4: highest-priced non-core non-Anthropic flagship
        wildcard_pool = [
            p for p in provider_flagships
            if p["label"].lower() not in WILDCARD_EXCLUDED_PREFIXES
        ]
        wildcard_pool.sort(key=lambda p: -self._completion_price(p["model"]))
        if wildcard_pool:
            wc = wildcard_pool[0]
            picks.append({
                "label": wc["label"],
                "model": wc["model"],
                "slot": "wildcard",
            })
            used_ids.add(wc["model"].get("id", ""))
        else:
            picks.append({"label": "Wildcard", "model": None, "slot": "wildcard"})

        # "Also Available": every other flagship (including Anthropic) by
        # price descending, capped at 5, excluding whatever already
        # landed in the top 4.
        others = [
            {"label": p["label"], "model": p["model"], "slot": "other"}
            for p in provider_flagships
            if p["model"].get("id", "") not in used_ids
        ]
        others.sort(key=lambda p: -self._completion_price(p["model"]))
        picks.extend(others[:5])

        return picks

    def _format(self, picks: list[dict]) -> str:
        """Render picks as a strict top-4 table + flexible "Also Available".

        The top-4 table shape is LOAD-BEARING — callers parse it to pick a
        model. Columns: rank, provider, name, slug, context, $/M in, $/M
        out. The "Also Available" section is free-form and can evolve.
        """
        top = [p for p in picks if p.get("slot") in ("core", "wildcard")]
        others = [p for p in picks if p.get("slot") == "other"]

        def fmt_ctx(m: Optional[dict]) -> str:
            if not m:
                return "—"
            ctx = m.get("context_length", 0)
            return f"{ctx:,}" if ctx else "?"

        def fmt_price_per_mtok(price_per_token: float) -> str:
            """OpenRouter prices are per-token as strings. Convert to $/M."""
            if price_per_token <= 0:
                return "—"
            per_m = price_per_token * 1_000_000
            if per_m >= 100:
                return f"${per_m:.0f}"
            if per_m >= 10:
                return f"${per_m:.1f}"
            return f"${per_m:.2f}"

        # Column widths tuned so the table renders cleanly in a mono font
        # without wrapping on an 80-col terminal most of the time. Callers
        # should not rely on exact widths — only on column order.
        lines: list[str] = []
        lines.append("**Top 4 (live from OpenRouter)**")
        lines.append("")
        header = (
            f"{'#':<2} {'Provider':<11} {'Model':<34} "
            f"{'Slug':<36} {'Context':>10} {'$/Min':>7} {'$/Mout':>7}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for i, pick in enumerate(top, 1):
            m = pick.get("model")
            label = pick.get("label", "?")
            if pick.get("slot") == "wildcard":
                # Title-case the bare provider prefix ("perplexity" -> "Perplexity"),
                # then mark with * so callers see it's the floating slot.
                label = "*" + label[:1].upper() + label[1:]
            if len(label) > 11:
                label = label[:11]
            if m is None:
                lines.append(
                    f"{i:<2} {label:<11} {'(none available)':<34} "
                    f"{'—':<36} {'—':>10} {'—':>7} {'—':>7}"
                )
                continue
            name = m.get("name", m.get("id", "?"))
            slug = m.get("id", "?")
            ctx_str = fmt_ctx(m)
            p_in = fmt_price_per_mtok(self._prompt_price(m))
            p_out = fmt_price_per_mtok(self._completion_price(m))
            # Truncate name/slug if they'd blow the column, but keep slug
            # intact — it's the load-bearing field. Name is cosmetic.
            if len(name) > 34:
                name = name[:33] + "…"
            lines.append(
                f"{i:<2} {label:<11} {name:<34} "
                f"{slug:<36} {ctx_str:>10} {p_in:>7} {p_out:>7}"
            )

        lines.append("")
        lines.append("`*` = wildcard slot (floats to whichever non-core provider has the priciest flagship).")
        lines.append("Prices shown are $/M tokens (input / output). Context is in tokens.")

        if others:
            lines.append("")
            lines.append("**Also Available**")
            lines.append("")
            for i, pick in enumerate(others, len(top) + 1):
                m = pick["model"]
                name = m.get("name", m.get("id", "?"))
                slug = m.get("id", "?")
                ctx_str = fmt_ctx(m)
                p_out = fmt_price_per_mtok(self._completion_price(m))
                lines.append(f"{i}. {name} — `{slug}` ({ctx_str} ctx, {p_out}/Mout)")

        lines.append("")
        lines.append("Pick a model by slug (e.g. `openai/gpt-5.4-pro`) when calling zen chat / thinkdeep.")
        return "\n".join(lines)
