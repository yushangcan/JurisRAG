"""证据不足时生成三种用途不同的二次检索表达。

问题式用于重新描述用户意图，答案式（HyDE）模拟法规中可能出现的回答，
关键词式保留实体、条号和法律术语。三者不是用来直接回答用户，只用于扩大
候选召回。配置了 LLM 时一次调用同时生成三种表达；离线模式仍能生成可解释
的问题式和关键词式，答案式明确标记为未启用。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from shared.llm import build_chat_model


@dataclass(frozen=True)
class QueryVariant:
    """一条检索表达及其用途，便于 trace 展示而不是只返回字符串。"""

    style: str
    text: str
    generated_by: str


_QUESTION_FILLERS = (
    "请问",
    "我想知道",
    "麻烦说明",
    "根据相关规定",
    "哪些",
    "什么情况下",
    "如何",
    "怎么",
    "是否",
    "能否",
    "可以吗",
    "需要吗",
    "为什么",
    "是什么",
    "规定了什么",
    "有何",
    "多少",
    "多久",
)


def _core_expression(query: str) -> str:
    """去掉低信息量问句外壳，保留法规名、主体、动作、数字和条号。"""

    core = query.strip()
    for filler in _QUESTION_FILLERS:
        core = core.replace(filler, " ")
    core = re.sub(r"[？?！!。；;，,：:]+", " ", core)
    core = re.sub(r"\s+", " ", core).strip(" 的呢吗呀 ")
    return core or query.strip()


def _deterministic_variants(query: str) -> list[QueryVariant]:
    """无模型时生成三条不会编造具体法律结论的安全表达。"""

    core = _core_expression(query)
    question = f"关于“{core}”，政府采购或招标投标法律如何规定？"
    hypothetical_answer = (
        f"相关法规可能从适用主体、条件、程序、期限和法律后果等方面规定“{core}”。"
    )
    # 关键词式故意不补“应当”“处罚”等结论性词，防止把猜测写进检索词。
    keywords = re.sub(r"[“”《》]", " ", core)
    keywords = re.sub(r"\s+", " ", keywords).strip()
    return [
        QueryVariant("question", question, "deterministic"),
        QueryVariant("answer", hypothetical_answer, "deterministic"),
        QueryVariant("keywords", keywords, "deterministic"),
    ]


def legal_issue_variants(query: str) -> list[QueryVariant]:
    """补充招标文件合法性判断需要的通用法律争点。

    用户通常会说“页数限制能不能废标”，法规却使用“项目特点和实际需要”、
    “实质性要求”“限制排斥”等表达。这里补的是判断维度，不预设合法或违法
    结论，因此即使模型不可用也能召回设置规则和否决规则两侧的证据。
    """

    if not any(term in query for term in ("招标文件", "采购文件", "投标文件")):
        return []
    return [
        QueryVariant(
            "question",
            (
                "招标文件设置资格、技术、商务或编制要求时，如何判断其是否与项目"
                "具体特点和实际需要相适应、是否与合同履行有关，以及是否构成不合理"
                "条件限制或排斥潜在投标人？"
            ),
            "legal_issue_rule",
        ),
        QueryVariant(
            "keywords",
            (
                "招标文件 投标文件 编制要求 投标无效情形 实质性要求 项目特点 "
                "实际需要 合同履行 不合理条件 限制 排斥 否决投标"
            ),
            "legal_issue_rule",
        ),
    ]


def _parse_json_content(content: str) -> dict[str, Any]:
    """兼容模型偶尔包裹的 Markdown 代码围栏，并严格要求 JSON 对象。"""

    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
    content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("检索改写结果不是 JSON 对象")
    return parsed


def _evidence_digest(evidence: list[dict[str, Any]]) -> str:
    """把当前头部候选压缩成规划模型可审阅的短证据清单。"""

    rows: list[str] = []
    for index, item in enumerate(evidence[:8], start=1):
        metadata = item.get("metadata", {})
        position = metadata.get("article_number") or metadata.get("section_title", "")
        text = re.sub(r"\s+", " ", str(item.get("text", ""))).strip()[:700]
        rows.append(
            f"候选{index}：《{metadata.get('title', '')}》{position}\n{text}"
        )
    return "\n\n".join(rows) or "当前没有候选证据"


def _as_string_list(value: Any) -> list[str]:
    """把模型 JSON 中的字符串或数组统一成去空值的字符串列表。"""

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _issue_coverage(value: Any, candidate_count: int) -> list[dict[str, Any]]:
    """规范并校验模型为每个法律争点标注的候选证据编号。"""

    if not isinstance(value, list):
        return []
    coverage: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        issue = str(row.get("issue", "")).strip()
        candidate_ids: list[int] = []
        for identifier in row.get("candidate_ids", []):
            try:
                number = int(identifier)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= candidate_count and number not in candidate_ids:
                candidate_ids.append(number)
        if issue:
            coverage.append({"issue": issue, "candidate_ids": candidate_ids})
    return coverage


def plan_retrieval(
    query: str,
    evidence: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[QueryVariant], str | None]:
    """让模型审阅“问题 + 当前证据”，决定是否继续检索。

    与旧版只看分数不同，规划模型必须说明口语术语、适用场景、法律争点和
    缺失证据。模型不可用时返回 ``planner_used=False``，调用方再使用确定性
    指标降级，保证网络故障不会让检索接口完全不可用。
    """

    fallback = _deterministic_variants(query) + legal_issue_variants(query)
    model = build_chat_model()
    if model is None:
        return (
            {
                "planner_used": False,
                "should_continue": True,
                "answerability": "unknown",
                "normalized_question": query,
                "domain": "unknown",
                "terminology_map": {},
                "legal_issues": [],
                "missing_evidence": [],
                "reason": "未配置可用的规划模型",
            },
            fallback,
            "未配置可用的 LLM，检索决策已降级为确定性指标",
        )

    prompt = (
        "你是RAG检索规划器，不向用户回答法律结论。请审阅用户问题和当前候选证据，"
        "只输出JSON对象，字段必须为：should_continue、normalized_question、domain、"
        "answerability、terminology_map、legal_issues、issue_coverage、missing_evidence、"
        "reason、question、"
        "hypothetical_answer、keywords。\n"
        "规则：\n"
        "1. 识别口语和法律术语，例如甲方可能是采购人或招标人，但场景不明时不能擅自选定。\n"
        "仅出现‘招标文件、投标文件、招标人’不足以区分政府采购与工程建设招标；"
        "没有政府采购、供应商、财政部门、工程建设、施工、勘察设计等明确上下文时，"
        "domain必须为uncertain。\n"
        "2. 将问题拆成作出结论所需的法律争点，并逐项检查候选证据是否覆盖。\n"
        "issue_coverage必须是数组，每项格式为"
        "{\"issue\":\"争点原文\",\"candidate_ids\":[1,2]}。只能填写候选证据中"
        "明确表达该规则的编号，不能使用模型记忆；没有直接支持时candidate_ids必须为空。\n"
        "3. 若证据足以给出有依据的条件式回答，即使仍需用户补充个案事实，也设置"
        "answerability=conditional；证据直接回答问题时设为direct；连可适用的一般法律"
        "标准都没有时才设为insufficient。answerability只能取direct、conditional、"
        "insufficient。should_continue独立表示补检是否可能提高答案质量，它可以和"
        "answerability=conditional同时为true。\n"
        "一般法律咨询不要求法规逐字出现用户描述的具体数字或行为，也不要求必须找到"
        "同案判例、专门答复或司法解释。只要现有法律法规已经提供可适用的一般判断标准，"
        "例如项目关联性、公平竞争、实质性要求和法律后果，就足以作条件式回答，"
        "此时必须设置should_continue=false。不得把‘没有150页专门规定’本身列为证据缺口。\n"
        "不得把列举的否决情形假定为封闭清单，不得在证据未说明时把页数等形式要求"
        "直接定性为非实质性或非法。\n"
        "只有每个必要法律争点都至少有一个候选证据编号时，should_continue才可以为false。"
        "但缺少针对具体页数、金额或个案的专门规定，不妨碍answerability=conditional。\n"
        "4. question是标准法律术语改写；hypothetical_answer只写法规可能使用的主体、"
        "条件、程序表达，不得虚构条号或结论；keywords用空格分隔多个争点检索词。\n"
        "5. domain只能是government_procurement、construction_bidding、uncertain之一。\n"
        f"用户问题：{query}\n\n当前候选证据：\n{_evidence_digest(evidence)}"
    )
    try:
        response = model.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_json_content(content)
        should_continue = parsed.get("should_continue")
        if not isinstance(should_continue, bool):
            raise ValueError("should_continue 不是布尔值")
        domain = str(parsed.get("domain", "uncertain")).strip()
        if domain not in {"government_procurement", "construction_bidding", "uncertain"}:
            domain = "uncertain"
        answerability = str(parsed.get("answerability", "insufficient")).strip().lower()
        if answerability not in {"direct", "conditional", "insufficient"}:
            answerability = "insufficient"
        terminology_map = parsed.get("terminology_map", {})
        if not isinstance(terminology_map, dict):
            terminology_map = {}
        legal_issues = _as_string_list(parsed.get("legal_issues"))
        coverage = _issue_coverage(parsed.get("issue_coverage"), min(len(evidence), 8))
        def issue_key(value: str) -> str:
            return re.sub(r"[\s，。；：、（）()‘’“”'\"]", "", value)

        covered_issue_names = {
            issue_key(row["issue"]) for row in coverage if row["candidate_ids"]
        }
        uncovered_issues = [
            issue
            for issue in legal_issues
            if not any(
                issue_key(issue) == covered
                or issue_key(issue) in covered
                or covered in issue_key(issue)
                for covered in covered_issue_names
            )
        ]
        # 布尔结论必须和逐争点证据映射一致；模型不能一边承认某争点没有候选，
        # 一边凭参数记忆声称证据充分并提前停止。
        if uncovered_issues:
            should_continue = True
        plan = {
            "planner_used": True,
            "should_continue": should_continue,
            "answerability": answerability,
            "normalized_question": str(parsed.get("normalized_question", query)).strip() or query,
            "domain": domain,
            "terminology_map": {
                str(key): str(value) for key, value in terminology_map.items()
            },
            "legal_issues": legal_issues,
            "issue_coverage": coverage,
            "uncovered_issues": uncovered_issues,
            "missing_evidence": _as_string_list(parsed.get("missing_evidence")),
            "reason": str(parsed.get("reason", "")).strip(),
        }
        variants = [
            QueryVariant("question", str(parsed.get("question", "")).strip(), "llm_planner"),
            QueryVariant(
                "answer",
                str(parsed.get("hypothetical_answer", "")).strip()[:500],
                "llm_planner",
            ),
            QueryVariant("keywords", str(parsed.get("keywords", "")).strip(), "llm_planner"),
        ]
        variants = [variant for variant in variants if variant.text]
        present_styles = {variant.style for variant in variants}
        for variant in fallback:
            if variant.style not in present_styles:
                variants.append(variant)
        # 规划模型负责理解问题，但不能覆盖通用的“条件能否设置 + 违反后果”
        # 双侧检索。即使三种 style 都已生成，法律争点查询仍单独保留。
        variants.extend(
            variant
            for variant in legal_issue_variants(query)
            if variant.text not in {existing.text for existing in variants}
        )
        return plan, variants, None
    except Exception as exc:
        return (
            {
                "planner_used": False,
                "should_continue": True,
                "answerability": "unknown",
                "normalized_question": query,
                "domain": "unknown",
                "terminology_map": {},
                "legal_issues": [],
                "missing_evidence": [],
                "reason": f"规划模型失败: {type(exc).__name__}",
            },
            fallback,
            f"规划模型失败，检索决策已降级为确定性指标: {type(exc).__name__}: {exc}",
        )


def build_retry_variants(query: str) -> tuple[list[QueryVariant], str | None]:
    """返回问题式、答案式、关键词式表达以及可审计警告。

    LLM 未配置或调用失败时，不影响本地检索：三种确定性表达仍会执行；
    其中答案式只补充中性的法规语言结构，不编造条号、金额或法律结论。
    """

    fallback = _deterministic_variants(query) + legal_issue_variants(query)
    model = build_chat_model()
    if model is None:
        return fallback, "未配置可用的 LLM，已使用确定性多路检索表达"

    try:
        prompt = (
            "你是法律知识库内部的检索改写器，不负责向用户作答。"
            "请只输出一个JSON对象，字段必须是question、hypothetical_answer、keywords。"
            "question：把口语主体和动作改成法规术语，并写出需要判断的法律争点；"
            "hypothetical_answer：不超过120字，只写可能出现在法规中的主体、条件、程序词，"
            "不得编造法规名、条号、日期、金额或结论；"
            "keywords：用空格分隔主体、行为、设置权限、项目关联性、公平性、"
            "实质性要求和法律后果等检索词。不得假设问题属于政府采购或工程建设招标。"
            f"\n原问题：{query}"
        )
        response = model.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)
        parsed = _parse_json_content(content)
        variants = [
            QueryVariant("question", str(parsed.get("question", "")).strip(), "llm"),
            QueryVariant(
                "answer",
                str(parsed.get("hypothetical_answer", "")).strip()[:500],
                "llm",
            ),
            QueryVariant("keywords", str(parsed.get("keywords", "")).strip(), "llm"),
        ]
        variants = [variant for variant in variants if variant.text]
        present_styles = {variant.style for variant in variants}
        # 模型漏字段时用不含具体法律结论的确定性表达补齐三种通道。
        for variant in fallback:
            if variant.style not in present_styles:
                variants.append(variant)
        # 法律争点规则允许和模型同类型表达并存；一个保留用户语义，另一个保证
        # “设置是否合法”和“能否否决”两侧法规都进入候选集。
        variants.extend(
            variant
            for variant in legal_issue_variants(query)
            if variant.text not in {existing.text for existing in variants}
        )
        return variants, None
    except Exception as exc:
        return fallback, (
            "检索改写模型失败，已使用确定性问题式和关键词式: "
            f"{type(exc).__name__}: {exc}"
        )


def variants_for_trace(variants: list[QueryVariant]) -> list[dict[str, str]]:
    """把 dataclass 转成可 JSON 序列化的 trace 数据。"""

    return [asdict(variant) for variant in variants]
