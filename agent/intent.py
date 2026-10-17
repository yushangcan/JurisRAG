"""低成本规划与能力边界。领域外问题、缺少法源和缺少事实分别处理。"""

from __future__ import annotations

import re

from agent.cases import match_cases
from agent.schemas import ExactReference, QueryPlan
from shared.legal_text import article_value, references, title_matches

LEGAL_JUDGMENT_MARKERS = ("合法", "违法", "能否", "可否", "是否有效", "可以否决", "责任", "处罚", "怎么办", "如何处理")
DOMAIN_TERMS = ("采购", "招标", "投标", "供应商", "财政", "评标", "评审", "联合体", "询价", "中标", "质疑", "投诉", "差别待遇", "排斥供应商")


def contextual_question(question: str, history: list[dict[str, str]] | None = None) -> str:
    """只在显式承接上题时使用最近一条用户问题，旧助手回答永远不是证据。"""
    if not history or not re.match(r"^(那|那么|上述|刚才|这个|该项目|补充|我是|我们是|项目属于)", question):
        return question
    previous = next((row["content"] for row in reversed(history) if row.get("role") == "user"), "")
    current_refs = references(question)
    if any(ref["document_title"] or ref["document_number"] for ref in current_refs):
        return question  # 新的明确引用覆盖上题，不能把上题条号也作为本轮请求。
    previous_refs = references(previous)
    if current_refs and not any(ref["document_title"] or ref["document_number"] for ref in previous_refs):
        for row in reversed(history[:-1]):
            if row.get("role") == "user" and any(ref["document_title"] or ref["document_number"] for ref in references(row["content"])):
                previous_refs = references(row["content"])
                break
    titles = list(dict.fromkeys(f"《{ref['document_title']}》" if ref["document_title"] else ref["document_number"] for ref in previous_refs if ref["document_title"] or ref["document_number"]))
    if current_refs and len(titles) == 1:
        return f"{titles[0]}{question}"
    return f"{previous[:1000]}\n补充问题：{question}" if previous else question


def plan_question(question: str) -> QueryPlan:
    refs = [ExactReference(**row) for row in references(question)]
    government = any(word in question for word in ("政府采购", "采购人", "财政部门", "质疑"))
    construction_terms = ("工程建设", "工程项目", "工程招标", "政府采购工程", "施工", "勘察", "设计招标", "招标投标法")
    construction = any(word in question for word in construction_terms)
    if "\n补充问题：" in question:
        supplement = question.rsplit("\n补充问题：", 1)[1]
        government_clarified = "政府采购" in supplement and not re.search(r"不(?:属于|适用|是).{0,3}政府采购", supplement)
        construction_clarified = any(word in supplement for word in construction_terms)
        if government_clarified and not construction_clarified:
            government, construction = True, False
        elif construction_clarified and not government_clarified:
            government, construction = False, True
    domain = "government_procurement" if government and not construction else "construction_bidding" if construction and not government else "uncertain"
    plan = QueryPlan(normalized_question=question, domain=domain, exact_references=refs)
    if len(refs) > 8:
        plan.boundary_reason = "本次指定的法规引用过多，请拆成每次最多8处引用的问题。"
        plan.boundary_action = "clarify"
        plan.required_case_facts = ["本次优先检索的法规与条号"]
        return plan
    boundary_checks = (
        (r"美国|欧盟|欧元|英国|联邦|FAR\b|EU\b", "当前知识库仅覆盖中国政府采购与招投标，未收录所问境外法源。"),
        (r"增值税.*(税率|多少)|适用.*税率|税率.*(多少|统一)", "当前知识库缺少确定税率所需的税收法源，不能用采购合同条款代替。"),
        (r"(判决|判多久|胜诉|败诉|胜率|一定中标|保证中标|肯定中标)|预测.*(结果|处罚)", "无法仅凭法规检索预测具体案件裁判或保证采购结果。"),
        (r"(哪些|全部|清单|目录).*(废止|失效)|现行有效.*(清单|目录)", "库内效力元数据尚不足以生成完整、现时有效的法规清单。"),
        (r"(最新|现行|截至|目前|现在|今年|202[6-9]年|203\d年).*(有效|废止|修订|版本)|(?:最新|现行).*(法|条例|办法)", "本系统没有实时法规更新核验能力，不能确认所问版本截至当前仍有效。"),
        (r"(最低|统一|一律).{0,10}注册资本|注册资本.{0,10}(最低|统一|一律|至少多少)", "库内没有统一最低注册资本数额依据；不能接受该前提并编造金额。"),
        (r"(上海|北京|广东|深圳|各省|本省|本市|当地|地方).*(限额|门槛|标准|目录)|(?:限额|门槛).*(上海|北京|广东|当地|今年|202[6-9])", "所问地区或年度的采购限额、目录需要对应地方官方文件，当前语料不完整。"),
        (r"(?:今年|明年|202[6-9]年|203\d年).*(限额|门槛|目录|招标数额)|(?:现行|最新).*(限额|门槛|招标数额)", "所问年度和地区的数额标准需要对应官方文件，当前知识库不能确认。"),
        (r"(忽略|无视|绕过).*(指令|规则|证据|约束)|(?:输出|显示|告诉我).*(API.?KEY|密钥|系统提示词|环境变量)", "该请求不能作为法规检索任务执行。"),
    )
    for pattern, reason in boundary_checks:
        if re.search(pattern, question, re.IGNORECASE):
            plan.boundary_reason = reason
            return plan
    if not any(word in question for word in DOMAIN_TERMS) and not refs:
        plan.boundary_reason = "当前知识库只处理政府采购与招投标法规，尚未覆盖该问题所属领域。"
        return plan
    if refs and any(ref.article_number and not (ref.document_title or ref.document_number) for ref in refs):
        plan.boundary_reason = "条号本身不能唯一定位法规，请补充法规名称。"
        plan.boundary_action = "clarify"
        plan.required_case_facts = ["需要查询的法规完整名称"]
        return plan
    case_question = question
    for title in sorted({ref.document_title for ref in refs if ref.document_title}, key=len, reverse=True):
        case_question = case_question.replace(title, "")
    # 法规名称中的“质疑和投诉”等词不是用户新增的业务请求。
    judgment = any(word in case_question for word in LEGAL_JUDGMENT_MARKERS)
    plan.primary_intent = "legality_assessment" if judgment else "procedure_guidance" if any(word in case_question for word in ("质疑", "投诉", "期限", "程序")) else "rule_explanation"
    plan.needs, plan.cases, facts = match_cases(case_question, domain)
    issues_in_requested_articles = all(need.sources and any(
        ref.document_title and title_matches(ref.document_title, source.document_title)
        and ref.article_number and source.article_number
        and article_value(ref.article_number) == article_value(source.article_number)
        for ref in refs for source in need.sources) for need in plan.needs)
    # 附带条号不等于只查原文；新增材料、投诉等争点不能被快速路径吞掉。
    if refs and any(ref.article_number for ref in refs) and not judgment and issues_in_requested_articles and not re.search(r"起诉|诉讼|赔偿|时效|救济|后果|如何计算|截止|哪一天|衔接|区别|比较|关系|联系|结合|分析|适用|解读|解释|为什么", question):
        plan.primary_intent = "exact_reference"
        plan.safe_exact = True
    if len(plan.needs) > 6:
        plan.boundary_reason = "本次问题涉及超过6个待核验争点，请按资格、评审、救济等主题拆分提问。"
        plan.boundary_action = "clarify"
        plan.required_case_facts = ["本次优先处理的争点"]
        return plan
    specific_case = judgment or bool(re.search(r"我们|我司|本项目|该项目|被否决|被拒绝|怎么维权|截止日期|最后一天|哪一天", question))
    if specific_case:
        plan.required_case_facts = facts
        if domain == "uncertain":
            plan.required_case_facts = list(dict.fromkeys(["项目属于政府采购还是工程建设招标", *facts]))[:6]
        if re.search(r"截止日期|最后一天|哪一天", question):
            plan.required_case_facts = list(dict.fromkeys([*plan.required_case_facts, "相关文书收到日期与适用的官方工作日历"]))[:6]
    if not plan.needs and not plan.safe_exact:
        from agent.schemas import EvidenceNeed
        plan.needs = [EvidenceNeed(issue="问题所需的一般规则", query=question)]
    return plan


def deterministic_intent(question: str) -> QueryPlan:
    return plan_question(question)


def is_safe_exact_reference_question(question: str, analysis: QueryPlan) -> bool:
    return analysis.safe_exact and not analysis.boundary_reason
