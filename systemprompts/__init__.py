"""
System prompts for PAL MCP tools (Maya fork — 5 tools)
"""

from .chat_prompt import CHAT_PROMPT
from .codereview_prompt import CODEREVIEW_PROMPT
from .consensus_prompt import CONSENSUS_PROMPT
from .debug_prompt import DEBUG_ISSUE_PROMPT
from .generate_code_prompt import GENERATE_CODE_PROMPT
from .thinkdeep_prompt import THINKDEEP_PROMPT

__all__ = [
    "CHAT_PROMPT",
    "CODEREVIEW_PROMPT",
    "CONSENSUS_PROMPT",
    "DEBUG_ISSUE_PROMPT",
    "GENERATE_CODE_PROMPT",
    "THINKDEEP_PROMPT",
]
