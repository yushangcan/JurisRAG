"""Agent 只保留直接影响开销与停止行为的少量配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass

from shared.config import PROJECT_ROOT  # noqa: F401


def _bounded(name: str, default: int, maximum: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} 必须为正数")
    return min(value, maximum)


@dataclass(frozen=True)
class AgentSettings:
    enabled: bool = True
    max_iterations: int = 2
    max_llm_calls: int = 2
    llm_timeout_seconds: int = 20
    request_timeout_seconds: int = 50
    max_question_chars: int = 2000
    max_concurrent_requests: int = 2


def get_agent_settings() -> AgentSettings:
    return AgentSettings(
        enabled=os.getenv("ENABLE_AGENT", "true").lower() == "true",
        max_iterations=_bounded("AGENT_MAX_ITERATIONS", 2, 2),
        max_llm_calls=_bounded("AGENT_MAX_LLM_CALLS", 2, 2),
        llm_timeout_seconds=_bounded("AGENT_LLM_TIMEOUT_SECONDS", 20, 25),
        request_timeout_seconds=_bounded("AGENT_REQUEST_TIMEOUT_SECONDS", 50, 60),
        max_question_chars=_bounded("AGENT_MAX_QUESTION_CHARS", 2000, 2000),
        max_concurrent_requests=_bounded("AGENT_MAX_CONCURRENT_REQUESTS", 2, 8),
    )
