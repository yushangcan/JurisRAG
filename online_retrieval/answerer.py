"""Turn retrieved evidence into a cautious, multi-source answer."""

from __future__ import annotations

import json
import re
from typing import Any

from shared.llm import build_chat_model


def _citation(item: dict[str, Any]) -> dict[str, Any]:
    """Expose only citation fields a frontend or interviewer needs."""

    metadata = item["metadata"]
    return {
        "document_title": metadata.get("title", ""),
        "category": metadata.get("category", ""),
        # 法规返回真实条号；普通通知/附件返回标题路径。最后才退回旧版
        # chunk_index，避免把没有“第X条”的文件伪装成法条。
        "article_or_position": (
            metadata.get("article_number")
            or metadata.get("section_path")
            or metadata.get("section_title")
            or metadata.get("chunk_index", "")
        ),
        "structure_type": metadata.get("structure_type", ""),
        "section_title": metadata.get("section_title", ""),
        "section_path": metadata.get("section_path", ""),
        "chapter": metadata.get("chapter", ""),
        "page_number": metadata.get("page_number", ""),
        "page_start": metadata.get("page_start", ""),
        "page_end": metadata.get("page_end", ""),
        "page_numbers": metadata.get("page_numbers", ""),
        "quote": item["text"],
        "source_portal": metadata.get("source_portal", ""),
        "source_url": metadata.get("source_url", ""),
        "attachment_url": metadata.get("attachment_url", ""),
        "distance": item["distance"],
        # 父块用于回答，下面字段说明究竟是哪一个细粒度子块命中了问题。
        "parent_id": item.get("parent_id", ""),
        "matched_child_chunk_id": item.get("matched_child_chunk_id", ""),
        "matched_child_quote": item.get("matched_child_text", ""),
        "retrieval_channels": item.get("retrieval_channels", []),
        "exact_matches": item.get("exact_matches", []),
        "hybrid_score": item.get("hybrid_score", 0.0),
        "reranker_score": item.get("reranker_score"),
    }


def _fallback_answer(question: str, evidence: list[dict[str, Any]]) -> str:
    """Create a transparent answer when no external LLM is configured."""

    lines = [f"基于知识库检索，问题“{question}”的相关依据如下："]
    for number, item in enumerate(evidence, start=1):
        title = item["metadata"].get("title", "未命名法规")
        lines.append(f"{number}. 《{title}》：{item['text']}")
    lines.append("以上是检索到的原文依据，当前基线未替你作出超出引用内容的法律结论。")
    return "\n".join(lines)


_QUANTIFIED_FACT_PATTERN = re.compile(
    r"\d+(?:\s*[—–-]\s*\d+)?\s*(?:页|天|日|年|万元|元|%|％|分钟|小时|个工作日)"
)
_ARTICLE_REFERENCE_PATTERN = re.compile(r"第\s*[一二三四五六七八九十百千万零〇两0-9]+\s*条")


def _unsupported_generated_facts(
    answer: str,
    question: str,
    evidence_text: str,
) -> list[str]:
    """找出问题和证据中都没有出现的量化事实或占位内容。

    这不是完整的事实核验器，但能拦截法律回答中风险很高的“模型自创期限、
    金额、页数区间”和 ``XXX`` 占位文号。法条编号由引用约束负责。
    """

    supported_text = f"{question}\n{evidence_text}".replace(" ", "")
    unsupported = [
        match.group(0)
        for match in _QUANTIFIED_FACT_PATTERN.finditer(answer)
        if match.group(0).replace(" ", "") not in supported_text
    ]
    if "XXX" in answer.upper():
        unsupported.append("XXX占位内容")
    if "封闭式列举" in answer and "封闭式列举" not in evidence_text:
        unsupported.append("证据未支持否决情形属于封闭式列举")
    return list(dict.fromkeys(unsupported))


def _parse_review_json(content: str) -> dict[str, Any]:
    """解析回答审查器输出的 JSON，兼容 Markdown 代码围栏。"""

    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("回答审查结果不是 JSON 对象")
    return parsed


def _review_answer_entailment(
    model: Any,
    question: str,
    evidence_text: str,
    answer: str,
) -> tuple[str, str | None]:
    """让模型逐句检查回答是否由引用证据真实支持，并在必要时重写。"""

    prompt = (
        "你是法律RAG回答审查器，不补充外部知识。请只输出JSON对象，字段为"
        "supported、problems、corrected_answer。逐句核对回答中的法律命题是否由对应"
        "[证据N]原文支持。特别检查：是否把否决全部投标用于否决单个投标；是否把"
        "列举情形擅自说成封闭清单；是否认为写入招标文件就自动合法；是否把页数要求"
        "无依据地定性为纯形式性；是否引用证据外法规或条号。若有任一问题，"
        "supported=false，并在corrected_answer中给出只基于证据的条件式回答；否则"
        "supported=true且corrected_answer保持原回答。\n"
        f"问题：{question}\n\n待审查回答：{answer}\n\n证据：\n{evidence_text}"
    )
    try:
        response = model.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        review = _parse_review_json(content)
    except Exception as exc:
        return answer, f"回答证据审查失败，保留已通过确定性校验的回答: {type(exc).__name__}: {exc}"
    if review.get("supported") is True:
        return answer, None
    corrected = str(review.get("corrected_answer", "")).strip()
    if not corrected:
        return answer, "回答证据审查发现问题，但未返回可用的修正版"
    return corrected, None


def _compact_reference(value: str) -> str:
    """删除引用中的空格和常见书名装饰，用于同一性比较。"""

    return re.sub(r"[\s《》]", "", value).lower()


def _unsupported_legal_references(
    answer: str,
    evidence: list[dict[str, Any]],
) -> list[str]:
    """检查回答中的法规标题和条号是否真实存在于本次引用集合。"""

    allowed_titles = {
        _compact_reference(str(item.get("metadata", {}).get("title", "")))
        for item in evidence
    }
    allowed_articles = {
        _compact_reference(str(item.get("metadata", {}).get("article_number", "")))
        for item in evidence
        if item.get("metadata", {}).get("article_number")
    }
    unsupported: list[str] = []
    for title in re.findall(r"《([^》]+)》", answer):
        compact_title = _compact_reference(title)
        if not any(
            compact_title == allowed
            or compact_title in allowed
            or allowed in compact_title
            for allowed in allowed_titles
            if allowed
        ):
            unsupported.append(f"未引用法规《{title}》")
    for article in _ARTICLE_REFERENCE_PATTERN.findall(answer):
        if _compact_reference(article) not in allowed_articles:
            unsupported.append(f"未引用条号{article}")
    return list(dict.fromkeys(unsupported))


def _llm_answer(question: str, evidence: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Ask the optional model to synthesize only from supplied evidence.

    The return value is ``(answer, warning)``.  A model/API failure must not
    break legal retrieval, so callers can fall back to the evidence-only mode
    while still telling the frontend why the fallback happened.
    """

    model = build_chat_model()
    if model is None:
        return None, None

    evidence_text = "\n\n".join(
        (
            f"证据{index}: 《{item['metadata'].get('title', '')}》"
            f"{item['metadata'].get('article_number', '')}\n{item['text']}"
        )
        for index, item in enumerate(evidence, start=1)
    )
    prompt = (
        "你是谨慎的法规检索助手。只能依据给定证据回答。必须遵守以下规则：\n"
        "1. 第一段直接回答问题。证据能支持判断规则但不能认定个案事实时，应给出"
        "条件式结论，例如‘不能仅凭该事实当然认定，应进一步判断……’，不能把整题"
        "简单处理成‘证据不足’。\n"
        "用户询问‘合法吗’但个案事实和适用场景不完整时，第一句不得直接写‘合法’或"
        "‘不合法’，必须说明不能仅凭现有事实直接认定，并给出判断条件。\n"
        "2. 区分政府采购与工程建设招标；问题未说明场景时，明确提示两类规则可能不同。\n"
        "3. 只能引用证据标题中出现的法规、条号和文号，不得引用模型记忆中的其他来源，"
        "不得生成XXX等占位文号。引用时标记[证据1]、[证据2]。\n"
        "4. 不得从‘法规没有明确列举’直接推导为合法或违法；应结合设置权限、与项目"
        "特点和实际需要的关联、是否属于实质性要求、是否限制排斥以及否决依据说明。\n"
        "5. 除非证据明确表明某个否决清单是封闭且穷尽的，不得声称‘未被法条列举，"
        "所以不属于否决情形’。招标文件合法设置且醒目标明的实质性要求可能影响响应性"
        "判断，但不得因此跳过对该要求本身合法性、合理性的审查。\n"
        "不得在证据未明确说明时把页数要求定性为纯形式性、非功能性、非行业惯例或"
        "当然不属于实质性要求。\n"
        "招标文件把某项要求写成‘实质性要求’并不自动证明该要求合法；仍须先审查"
        "设置权限、项目关联性、公平性以及是否不合理限制排斥。\n"
        "证据若规定的是‘所有投标均不符合时否决全部投标’，不得用它证明可以否决"
        "某一个投标；必须严格区分否决全部投标和否决单个投标。\n"
        "6. 每个法律判断必须由所标记的证据原文实际支持，不能用否决情形条文证明"
        "公平性或项目关联性等该条没有表达的命题。\n"
        "7. 最后列出仍需用户补充的关键事实，不重复粘贴全部法条。\n"
        f"问题：{question}\n\n证据：\n{evidence_text}"
    )
    try:
        response = model.invoke(prompt)
    except Exception as exc:
        # Only report the exception type/status text.  Never include settings or
        # request headers because they may contain the user's API key.
        return None, f"模型调用失败，已切换为本地检索回答: {type(exc).__name__}: {exc}"
    content = response.content if isinstance(response.content, str) else str(response.content)
    unsupported = [
        *_unsupported_generated_facts(content, question, evidence_text),
        *_unsupported_legal_references(content, evidence),
    ]
    if unsupported:
        correction_prompt = (
            "请重写下面的法规回答。删除所有未被问题或证据明确支持的数字、期限、"
            "行业惯例、示例数据和占位文号；不得新增法规或条号。保留直接的条件式结论，"
            "只用[证据N]引用给定证据。\n"
            f"检测到的不受支持内容：{'、'.join(unsupported)}\n"
            f"原回答：{content}\n\n问题：{question}\n\n证据：\n{evidence_text}"
        )
        try:
            corrected = model.invoke(correction_prompt)
            corrected_content = (
                corrected.content
                if isinstance(corrected.content, str)
                else str(corrected.content)
            )
            corrected_issues = [
                *_unsupported_generated_facts(
                    corrected_content, question, evidence_text
                ),
                *_unsupported_legal_references(corrected_content, evidence),
            ]
            if not corrected_issues:
                content = corrected_content
            else:
                return None, "模型回答包含证据外的量化事实，已切换为原文证据回答"
        except Exception as exc:
            return None, (
                "模型回答事实校验未通过，且重写失败，已切换为原文证据回答: "
                f"{type(exc).__name__}: {exc}"
            )
    reviewed_content, review_warning = _review_answer_entailment(
        model, question, evidence_text, content
    )
    reviewed_issues = [
        *_unsupported_generated_facts(reviewed_content, question, evidence_text),
        *_unsupported_legal_references(reviewed_content, evidence),
    ]
    if reviewed_issues:
        return None, (
            "模型回答在证据审查后仍包含无依据内容，已切换为原文证据回答: "
            + "、".join(reviewed_issues)
        )
    return reviewed_content, review_warning


def answer_question(question: str, top_k: int = 5) -> dict[str, Any]:
    """Retrieve evidence and return answer, multiple citations and disclaimer."""

    from online_retrieval.retriever import hybrid_search

    retrieval = hybrid_search(question, top_k=top_k)
    evidence = retrieval["results"]
    evidence_sufficient = bool(retrieval["trace"].get("evidence_sufficient"))
    if not evidence_sufficient:
        # 拒答发生在调用最终回答模型之前。候选只是“可能相关”，证据门没有
        # 放行，因此不能把它们包装成正式引用误导用户。
        return {
            "question": question,
            "answer": (
                "当前知识库经过多路检索后，仍未找到足以支持该问题结论的法规证据，"
                "因此我不能据此作答。请补充准确的法规名称、条号、文号或更多事实，"
                "并以官方最新文本为准。"
            ),
            "citations": [],
            "source_count": 0,
            "source_titles": [],
            "generation_mode": "abstention",
            "warning": retrieval["trace"].get("warning"),
            "retrieval_trace": retrieval["trace"],
            "insufficient_evidence": True,
            "refusal_reason": retrieval["trace"].get("refusal_reason"),
            "retrieved_candidate_count": len(evidence),
            "disclaimer": "未找到充分证据时系统拒绝给出法律结论；法规时效请核验官方最新记录。",
        }

    unique_sources: list[str] = []
    for item in evidence:
        title = item["metadata"].get("title", "")
        if title and title not in unique_sources:
            unique_sources.append(title)

    model_answer, model_warning = _llm_answer(question, evidence)
    answer = model_answer or _fallback_answer(question, evidence)
    warnings = [warning for warning in (retrieval["trace"].get("warning"), model_warning) if warning]
    return {
        "question": question,
        "answer": answer,
        "citations": [_citation(item) for item in evidence],
        "source_count": len(unique_sources),
        "source_titles": unique_sources,
        "generation_mode": "llm" if model_answer else "retrieval_fallback",
        "warning": " | ".join(warnings) if warnings else None,
        "retrieval_trace": retrieval["trace"],
        "insufficient_evidence": False,
        "refusal_reason": None,
        "disclaimer": "回答仅基于当前知识库检索结果，不构成法律意见；法规时效请核验官方最新记录。",
    }
