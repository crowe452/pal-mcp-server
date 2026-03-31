"""
List Models tool - Display available AI models with capabilities and pricing.

This is a utility tool that does not call any AI model. It queries the
internal provider registry to return a formatted table of all available
models, their context windows, and capability flags.
"""

import logging
from typing import Any, Optional

from mcp.types import TextContent

from providers.registry import ModelProviderRegistry
from tools.shared.base_tool import BaseTool
from tools.shared.exceptions import ToolExecutionError

logger = logging.getLogger(__name__)


class ListModelsTool(BaseTool):
    """Utility tool that lists available AI models without calling any model."""

    def get_name(self) -> str:
        return "listmodels"

    def get_description(self) -> str:
        return (
            "List all available AI models with their capabilities. "
            "Shows model ID, name, context window size, provider, and capability flags "
            "(thinking, code generation, images, JSON mode). "
            "Use to discover which models are available before calling other tools. "
            "Supports filtering by provider name and minimum context window size."
        )

    def get_input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": (
                        "Filter by provider name (case-insensitive substring match). "
                        "Examples: 'openai', 'anthropic', 'google', 'openrouter', 'xai'."
                    ),
                },
                "min_context": {
                    "type": "integer",
                    "description": (
                        "Minimum context window size in tokens. "
                        "Default: 0 (show all models). "
                        "Common values: 100000, 200000, 1000000."
                    ),
                    "default": 0,
                    "minimum": 0,
                },
            },
            "required": [],
        }

    def get_annotations(self) -> Optional[dict[str, Any]]:
        return {"readOnlyHint": True}

    def requires_model(self) -> bool:
        return False

    # Abstract method stubs — never called because execute() is fully
    # overridden and requires_model() returns False.

    def get_system_prompt(self) -> str:
        return ""

    def get_request_model(self):
        return None

    async def prepare_prompt(self, request) -> str:
        return ""

    async def execute(self, arguments: dict[str, Any]) -> list[TextContent]:
        """List available models from the provider registry."""
        try:
            provider_filter = (arguments.get("provider") or "").strip().lower()
            min_context = arguments.get("min_context", 0) or 0

            logger.info(
                f"listmodels called: provider={provider_filter or 'all'}, "
                f"min_context={min_context}"
            )

            # Get all available models from the registry
            available_models = ModelProviderRegistry.get_available_models(
                respect_restrictions=True
            )

            if not available_models:
                return [
                    TextContent(
                        type="text",
                        text="No models available. Check that at least one API key is configured.",
                    )
                ]

            # Build model info list with capabilities
            rows = []
            for model_name, provider_type in available_models.items():
                # Apply provider filter — match against provider type OR model name prefix
                # (e.g., "openai" matches openai/gpt-5 even when routed via openrouter)
                if provider_filter:
                    model_prefix = model_name.split("/")[0].lower() if "/" in model_name else ""
                    if (
                        provider_filter not in provider_type.value.lower()
                        and provider_filter not in model_prefix
                    ):
                        continue

                # Get capabilities from the provider
                provider = ModelProviderRegistry.get_provider_for_model(model_name)
                if not provider:
                    continue

                try:
                    caps = provider.get_capabilities(model_name)
                except Exception:
                    caps = None

                context_window = caps.context_window if caps else 0
                max_output = caps.max_output_tokens if caps else 0

                # Apply context window filter
                if context_window < min_context:
                    continue

                # Build capability flags
                flags = []
                if caps:
                    if caps.supports_extended_thinking:
                        flags.append("thinking")
                    if getattr(caps, "allow_code_generation", False):
                        flags.append("code-gen")
                    if caps.supports_images:
                        flags.append("images")
                    if caps.supports_json_mode:
                        flags.append("json")

                # Get intelligence score for sorting
                score = caps.intelligence_score if caps else 0

                # Get aliases
                aliases = caps.aliases if caps and caps.aliases else []

                rows.append(
                    {
                        "model": model_name,
                        "provider": provider_type.value,
                        "context": context_window,
                        "max_output": max_output,
                        "score": score,
                        "flags": flags,
                        "aliases": aliases,
                        "description": caps.description if caps else "",
                    }
                )

            if not rows:
                filter_desc = []
                if provider_filter:
                    filter_desc.append(f"provider='{provider_filter}'")
                if min_context > 0:
                    filter_desc.append(f"min_context={min_context:,}")
                return [
                    TextContent(
                        type="text",
                        text=f"No models match filters: {', '.join(filter_desc)}.",
                    )
                ]

            # Sort: intelligence score descending, then context window descending, then name
            rows.sort(key=lambda r: (-r["score"], -r["context"], r["model"]))

            # Format as table
            lines = []
            lines.append(f"**{len(rows)} models available**")
            if provider_filter:
                lines.append(f"Filtered by provider: {provider_filter}")
            if min_context > 0:
                lines.append(f"Filtered by min context: {min_context:,} tokens")
            lines.append("")

            # Column headers
            lines.append(
                f"{'Model':<45} {'Score':>5} {'Context':>10} "
                f"{'MaxOut':>8} {'Provider':<12} {'Capabilities'}"
            )
            lines.append("-" * 120)

            for row in rows:
                ctx_str = _format_tokens(row["context"])
                out_str = _format_tokens(row["max_output"])
                flags_str = ", ".join(row["flags"]) if row["flags"] else "-"

                lines.append(
                    f"{row['model']:<45} {row['score']:>5} {ctx_str:>10} "
                    f"{out_str:>8} {row['provider']:<12} {flags_str}"
                )

                # Show aliases on next line if present
                if row["aliases"]:
                    alias_str = ", ".join(row["aliases"])
                    lines.append(f"  aliases: {alias_str}")

            result = "\n".join(lines)
            logger.info(f"listmodels returned {len(rows)} models")
            return [TextContent(type="text", text=result)]

        except Exception as e:
            logger.error(f"Error in listmodels: {e}")
            raise ToolExecutionError(str(e)) from e


def _format_tokens(n: int) -> str:
    """Format token count for display: 200000 -> '200K', 1000000 -> '1.0M'."""
    if n <= 0:
        return "-"
    if n >= 1_000_000:
        val = n / 1_000_000
        return f"{val:.1f}M" if val != int(val) else f"{int(val)}M"
    if n >= 1_000:
        val = n / 1_000
        return f"{val:.0f}K" if val == int(val) else f"{val:.1f}K"
    return str(n)
