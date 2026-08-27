"""确定性信号提取、意图降级和提示词构造。"""

from __future__ import annotations

import re

from agent.schemas import ExactReference, IntentAnalysis
from online_retrieval.retriever import ARTICLE_PATTERN, DOCUMENT_NUMBER_PATTERN, TITLE_PATTERN


GOVERNMENT_SIGNALS = ("政府采购", "供应商", "采购人", "财政部门", "采购代理机构")
CONSTRUCTION_SIGNALS = ("工程建设", "施工", "勘察", "设计招标", "监理", "工程项目")

# 这些表达要求模型结合多条规则判断法律后果，不能被“出现了法规名和条号”误导到
# 原文快速路径。列表宁可保守一些：漏走快速路径只影响速度，误走则可能影响结论。
LEGAL_JUDGMENT_MARKERS = (
    "合法",
    "违法",
    "合规",
    "能否",
    "可以否决",
    "是否有效",
    "有效吗",
    "合理吗",
    "如何处理",
    "怎么办",
    "责任",
    "处罚",
    "救济",
    "投诉",
    "质疑",
)


def is_safe_exact_reference_question(question: str, analysis: IntentAnalysis) -> bool:
    """判断问题是否只是“明确法规 + 明确条号”的低风险原文查询。

    快速路径必须同时满足：只有一个正式引用、法规名和条号都存在、不含评价/后果
    表达。最终是否进入快速路径还要等精确工具返回等值命中结果后再次确认。
    """

    references = analysis.exact_references
    return (
        analysis.primary_intent == "exact_reference"
        and len(references) == 1
        and bool(references[0].document_title.strip())
        and bool(references[0].article_number.strip())
        and not any(marker in question for marker in LEGAL_JUDGMENT_MARKERS)
        and not any(marker in question for marker in ("以及", "同时", "分别", "比较"))
    )


def deterministic_intent(question: str) -> IntentAnalysis:
    """模型不可用时仍能给出保守、可解释的意图和首轮查询。"""

    titles = TITLE_PATTERN.findall(question)
    articles = [match.group(0).replace(" ", "") for match in ARTICLE_PATTERN.finditer(question)]
    numbers = [match.group(0) for match in DOCUMENT_NUMBER_PATTERN.finditer(question)]
    # 用户常省略书名号，例如“政府采购法第二十二条”。只从紧邻条号的文本中
    # 提取标题，不扫描或猜测问题中没有出现的其他法规。
    if not titles:
        inline_title = re.search(
            r"((?:中华人民共和国)?[\u4e00-\u9fff]{2,24}?(?:法实施条例|实施条例|条例|办法|规定|法))"
            r"(?=\s*第\s*[一二三四五六七八九十百千万零〇两0-9]+\s*条)",
            question,
        )
        if inline_title:
            title = re.sub(r"^(?:请问|根据|依照|关于)", "", inline_title.group(1))
            if title:
                titles = [title]
    references: list[ExactReference] = []
    reference_count = max(len(titles), len(articles), len(numbers), 0)
    for index in range(reference_count):
        references.append(
            ExactReference(
                document_title=titles[index] if index < len(titles) else "",
                article_number=articles[index] if index < len(articles) else "",
                document_number=numbers[index] if index < len(numbers) else "",
            )
        )

    government = [signal for signal in GOVERNMENT_SIGNALS if signal in question]
    construction = [signal for signal in CONSTRUCTION_SIGNALS if signal in question]
    if government and not construction:
        domain = "government_procurement"
        domain_signals = government
    elif construction and not government:
        domain = "construction_bidding"
        domain_signals = construction
    else:
        domain = "uncertain"
        domain_signals = [*government, *construction]

    legality = any(marker in question for marker in LEGAL_JUDGMENT_MARKERS)
    procedure = any(marker in question for marker in ("流程", "程序", "期限", "如何办理", "怎么做"))
    scope = any(marker in question for marker in ("是否适用", "适用范围", "效力", "废止", "现行有效"))
    legal_markers = (
        "招标", "投标", "采购", "供应商", "评标", "中标", "法规", "法律", "条例", "办法", "第"
    )
    clearly_unrelated = any(
        marker in question for marker in ("天气", "菜谱", "电影推荐", "写代码", "数学题", "旅游攻略")
    ) and not any(marker in question for marker in legal_markers)
    if clearly_unrelated:
        intent = "out_of_scope"
    elif references:
        intent = "exact_reference"
    elif legality:
        intent = "legality_assessment"
    elif procedure:
        intent = "procedure_guidance"
    elif scope:
        intent = "scope_applicability"
    else:
        intent = "rule_explanation"

    terminology: dict[str, str] = {}
    normalized = question
    if "甲方" in question:
        terminology["甲方"] = "采购人或招标人，具体取决于项目场景"
        normalized = question.replace("甲方", "采购人或招标人")

    if legality:
        issues = [
            "招标或采购文件设置该项要求的权限及其与项目特点、实际需要的关联",
            "该项要求是否构成不合理限制排斥或差别待遇",
            "未满足该项要求时能否依法作为否决投标或无效响应的依据",
        ]
    elif "内部制度" in question:
        issues = [
            "内部制度能否作为对投标人或供应商的文件要求",
            "相关要求是否公开、明确、与项目相关且可以验证",
            "相关要求是否违法形成资格、评分或否决条件",
        ]
    else:
        issues = [normalized]

    required_facts: list[str] = []
    if domain == "uncertain" and any(term in question for term in ("招标", "投标", "甲方")):
        required_facts.append("项目属于政府采购还是工程建设招标")
    if legality:
        required_facts.append("招标文件或采购文件中该项要求及法律后果的完整原文")

    return IntentAnalysis(
        in_scope=not clearly_unrelated,
        primary_intent=intent,
        normalized_question=normalized,
        domain=domain,
        domain_signals=domain_signals,
        terminology_map=terminology,
        legal_issues=issues,
        required_case_facts=list(dict.fromkeys(required_facts)),
        exact_references=references,
        initial_queries=[normalized, *issues][:5],
        reason_summary="确定性规则根据正式引用、评价词和领域信号生成",
    )


def intent_prompt(question: str, fallback: IntentAnalysis) -> str:
    """要求模型只返回可公开的结构化分析，不输出隐藏思维链。"""

    return (
        "你是政府采购与招标投标法律检索 Agent 的意图分析器，不回答法律结论。"
        "只输出 JSON 对象，字段必须符合：in_scope、primary_intent、secondary_intents、"
        "normalized_question、domain、domain_signals、terminology_map、legal_issues、"
        "required_case_facts、exact_references、initial_queries、reason_summary。\n"
        "primary_intent只能是exact_reference、rule_explanation、legality_assessment、"
        "procedure_guidance、scope_applicability、multi_issue、out_of_scope之一。"
        "domain只能是government_procurement、construction_bidding、uncertain。\n"
        "规则：没有政府采购/供应商/财政部门或工程建设/施工/勘察设计等明确信号时，"
        "不能仅凭招标文件、投标文件、甲方认定领域；甲方应映射为采购人或招标人候选。"
        "合法性问题要拆出设置权限与项目关联性、公平竞争限制、法律后果等必要争点。"
        "用户明确写出的法规、条号和文号必须原样放入exact_references，不能凭记忆新增。"
        "reason_summary只写一句可公开决策摘要。\n"
        f"确定性预分析：{fallback.model_dump_json(ensure_ascii=False)}\n"
        f"用户问题：{question}"
    )


def compact_keywords(values: list[str], limit: int = 220) -> str:
    """把争点转成关键词串，去掉问句虚词但保留数字和法律动作。"""

    text = " ".join(values)
    text = re.sub(r"[？?，。；：、（）()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]
