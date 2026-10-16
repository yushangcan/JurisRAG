"""边界清晰的请求、检索计划和模型输出，不依赖 Agent 框架。"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExactReference(StrictModel):
    document_title: str = ""
    article_number: str = ""
    document_number: str = ""


class EvidenceNeed(StrictModel):
    issue: str
    query: str
    sources: list[ExactReference] = Field(default_factory=list)


class QueryPlan(StrictModel):
    normalized_question: str
    primary_intent: str = "rule_explanation"
    domain: str = "uncertain"
    exact_references: list[ExactReference] = Field(default_factory=list)
    needs: list[EvidenceNeed] = Field(default_factory=list)
    cases: list[str] = Field(default_factory=list)
    required_case_facts: list[str] = Field(default_factory=list)
    boundary_reason: str = ""
    boundary_action: Literal["refuse", "clarify"] = "refuse"
    safe_exact: bool = False


class SearchPlan(StrictModel):
    queries: list[Annotated[str, StringConstraints(min_length=1, max_length=300, strip_whitespace=True)]] = Field(min_length=1, max_length=3)


class AnswerClaim(StrictModel):
    text: str = Field(min_length=1, max_length=800)
    evidence_id: str = Field(min_length=1, max_length=300)
    quote: str = Field(min_length=8, max_length=3000)


class GroundedAnswer(StrictModel):
    claims: list[AnswerClaim] = Field(min_length=1, max_length=10)


class HistoryMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)
