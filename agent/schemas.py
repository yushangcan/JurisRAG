"""模型输出和工具动作使用的 Pydantic Schema。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Domain = Literal["government_procurement", "construction_bidding", "uncertain"]
Answerability = Literal["direct", "conditional", "insufficient"]


def _normalize_bounded_strings(value: Any, limit: int) -> list[str]:
    """把模型生成的字符串列表清洗成稳定、有限的状态字段。

    大模型遵守 JSON Schema 的能力并非百分之百可靠，偶尔会返回超过上限的
    数组。如果直接交给 Pydantic 的 ``max_length`` 校验，整个 HTTP 请求会因
    一项非关键字段过长而失败。这里在正式校验前完成四件事：

    1. 非列表值转换为空列表；
    2. 去掉空白项，并统一转成字符串；
    3. 保留第一次出现的顺序并去重；
    4. 只保留状态机实际允许消费的前 ``limit`` 项。

    截断只用于查询提示和公开执行摘要，不会修改法规证据本身。
    """

    if not isinstance(value, (list, tuple)):
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item).strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
        if len(normalized) >= limit:
            break
    return normalized


class ExactReference(BaseModel):
    """从用户问题中识别出的正式法规引用。"""

    document_title: str = ""
    article_number: str = ""
    document_number: str = ""


class IntentAnalysis(BaseModel):
    """意图节点的完整结构化输出。"""

    in_scope: bool = True
    primary_intent: Literal[
        "exact_reference",
        "rule_explanation",
        "legality_assessment",
        "procedure_guidance",
        "scope_applicability",
        "multi_issue",
        "out_of_scope",
    ]
    secondary_intents: list[str] = Field(default_factory=list)
    normalized_question: str
    domain: Domain = "uncertain"
    domain_signals: list[str] = Field(default_factory=list)
    terminology_map: dict[str, str] = Field(default_factory=dict)
    legal_issues: list[str] = Field(default_factory=list, max_length=6)
    required_case_facts: list[str] = Field(default_factory=list, max_length=6)
    exact_references: list[ExactReference] = Field(default_factory=list, max_length=4)
    initial_queries: list[str] = Field(default_factory=list, max_length=5)
    reason_summary: str = ""

    @field_validator("terminology_map", mode="before")
    @classmethod
    def normalize_terminology_values(cls, value):
        """兼容模型把一个术语解释拆成字符串数组，但状态中统一保存字符串。"""

        if not isinstance(value, dict):
            return {}
        return {
            str(key): "；".join(str(item) for item in raw) if isinstance(raw, list) else str(raw)
            for key, raw in value.items()
        }

    @field_validator(
        "legal_issues",
        "required_case_facts",
        "initial_queries",
        mode="before",
    )
    @classmethod
    def normalize_intent_lists(cls, value, info):
        """防止意图模型偶发返回过长列表导致整条 Agent 链路中断。"""

        limit = 5 if info.field_name == "initial_queries" else 6
        return _normalize_bounded_strings(value, limit)


class ToolAction(BaseModel):
    """规划模型建议、策略层批准后才能执行的一个 MCP 动作。"""

    tool_name: Literal[
        "exact_legal_search",
        "keyword_search",
        "semantic_search",
        "hyde_search",
        "multi_query_search",
        "lookup_document_scope",
    ]
    arguments: dict[str, Any]
    target_issues: list[str] = Field(default_factory=list, max_length=6)
    expected_evidence: str = ""
    reason_summary: str = ""

    @field_validator("target_issues", mode="before")
    @classmethod
    def normalize_target_issues(cls, value):
        """规划模型列出过多争点时保序截断，工具调用仍可继续执行。"""

        return _normalize_bounded_strings(value, 6)


class ActionDecision(BaseModel):
    """模型一次只选择一个动作，避免单轮无界并行工具调用。"""

    action: ToolAction


class AnswerDraft(BaseModel):
    """回答节点输出；citation_ids 必须映射到当前真实证据。"""

    answer: str
    citation_ids: list[str] = Field(default_factory=list, max_length=8)
    missing_user_facts: list[str] = Field(default_factory=list, max_length=6)


class AnswerValidation(BaseModel):
    """LLM 只审查草稿，不得在此节点补充新法律知识。"""

    passed: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    overstated_claims: list[str] = Field(default_factory=list)
    domain_mismatches: list[str] = Field(default_factory=list)
    missing_issues: list[str] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    correction_action: Literal[
        "none", "rewrite_answer", "retrieve_more", "conditional_answer", "refuse"
    ] = "none"
    reason_summary: str = ""


class ExactSearchArgs(BaseModel):
    document_title: str = ""
    article_number: str = ""
    document_number: str = ""
    top_k: int = Field(default=5, ge=1, le=10)


class TextSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class KeywordSearchArgs(BaseModel):
    keywords: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)


class HydeSearchArgs(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)


class MultiQueryArgs(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=5)
    top_k: int = Field(default=5, ge=1, le=10)


class ScopeLookupArgs(BaseModel):
    title_or_number: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=5, ge=1, le=10)


class IdListArgs(BaseModel):
    ids: list[str] = Field(min_length=1, max_length=20)


class CoverageArgs(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    evidence_ids: list[str] = Field(min_length=1, max_length=20)
