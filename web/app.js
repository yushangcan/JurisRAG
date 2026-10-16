"use strict";

// 先集中获取页面元素，后续函数直接复用，避免每次交互都重新查找 DOM。
const elements = {
  form: document.querySelector("#question-form"),
  question: document.querySelector("#question-input"),
  mode: document.querySelector("#query-mode"),
  topK: document.querySelector("#top-k"),
  submit: document.querySelector("#submit-button"),
  clear: document.querySelector("#clear-button"),
  healthBadge: document.querySelector("#health-badge"),
  healthText: document.querySelector("#health-text"),
  feedback: document.querySelector("#feedback-section"),
  feedbackTitle: document.querySelector("#feedback-title"),
  feedbackMessage: document.querySelector("#feedback-message"),
  loading: document.querySelector("#loading-section"),
  loadingText: document.querySelector("#loading-text"),
  result: document.querySelector("#result-section"),
  resultMeta: document.querySelector("#result-meta"),
  answerAlert: document.querySelector("#answer-alert"),
  answer: document.querySelector("#answer-content"),
  disclaimer: document.querySelector("#disclaimer"),
  citationCount: document.querySelector("#citation-count"),
  citations: document.querySelector("#citation-list"),
  trace: document.querySelector("#trace-content"),
};

// 页面只允许一个正在执行的请求。保存控制器后，用户清空页面时可以取消旧请求。
let activeRequest = null;
let history = [];

/**
 * 创建带文本的 DOM 元素。所有接口文本都通过 textContent 写入，浏览器不会把
 * 法规原文当作 HTML 执行，这比拼接 innerHTML 更安全，也更容易理解。
 */
function makeElement(tagName, className = "", text = "") {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text !== "") node.textContent = text;
  return node;
}

/** 只接受 http/https 官方链接，阻止异常协议出现在可点击链接中。 */
function safeHttpUrl(value) {
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

/** 将不同的后端生成模式翻译成用户能直接看懂的中文。 */
function generationModeLabel(mode) {
  const labels = {
    agent_llm: "Agent 规划与模型归纳",
    agent_evidence_fallback: "Agent 规划与原文回答",
    agent_abstention: "Agent 证据不足拒答",
    agent_exact_reference: "指定法条原文",
    rag_llm: "基础检索与模型归纳",
    rag_evidence_fallback: "基础检索与原文回答",
    rag_abstention: "基础检索证据不足拒答",
    rag_exact_reference: "指定法条原文",
  };
  return labels[mode] || mode || "未知模式";
}

/**
 * 请求健康接口并显示索引规模。失败只改变状态提示，不阻止用户继续尝试提问，
 * 因为短暂的网络抖动可能在下一次请求时已经恢复。
 */
async function checkHealth() {
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data.status !== "ok") {
      elements.healthBadge.className = "health-badge is-offline";
      elements.healthText.textContent = "知识库组件未就绪";
      return;
    }
    elements.healthBadge.className = "health-badge is-online";
    elements.healthText.textContent = `知识库在线 · ${data.indexed_children.toLocaleString()} 个检索块`;
  } catch {
    elements.healthBadge.className = "health-badge is-offline";
    elements.healthText.textContent = "知识库连接失败";
  }
}

/** 统一控制加载状态，保证按钮、提示条和表单状态保持一致。 */
function setLoading(isLoading) {
  elements.submit.disabled = isLoading;
  elements.submit.classList.toggle("is-loading", isLoading);
  elements.loading.hidden = !isLoading;
}

/** 展示接口错误或输入错误；错误信息来自后端时也按纯文本处理。 */
function showFeedback(title, message) {
  elements.feedbackTitle.textContent = title;
  elements.feedbackMessage.textContent = message;
  elements.feedback.hidden = false;
}

/** 隐藏上一次错误，通常在发起新请求或清空页面时调用。 */
function hideFeedback() {
  elements.feedback.hidden = true;
}

/** 创建回答顶部的模式、来源数等摘要标签。 */
function renderResultMeta(data) {
  elements.resultMeta.replaceChildren();
  elements.resultMeta.append(
    makeElement("span", "meta-chip", generationModeLabel(data.generation_mode)),
    makeElement("span", "meta-chip", `${data.source_count || 0} 部法规来源`),
    makeElement("span", "meta-chip", `${(data.citations || []).length} 条证据`),
  );
}

/** 根据 citation 的分页字段生成一个不重复的页码说明。 */
function citationPageLabel(citation) {
  if (citation.page_numbers) return `第 ${citation.page_numbers} 页`;
  if (citation.page_start && citation.page_end && citation.page_start !== citation.page_end) {
    return `第 ${citation.page_start}-${citation.page_end} 页`;
  }
  if (citation.page_number || citation.page_start) {
    return `第 ${citation.page_number || citation.page_start} 页`;
  }
  return "";
}

/**
 * 将一条后端 citation 渲染成可展开证据。摘要用于快速扫视，展开区保留法条
 * 原文、命中通道、精确匹配类型和官方来源，满足 RAG 的可溯源要求。
 */
function renderCitation(citation, index) {
  const card = makeElement("details", "citation-card");
  if (index === 0) card.open = true;

  const summary = makeElement("summary", "citation-summary");
  summary.append(makeElement("span", "citation-number", String(index + 1)));

  const title = makeElement("span", "citation-title");
  title.append(makeElement("strong", "", citation.document_title || "未命名法规"));
  const positionParts = [citation.category, citation.article_or_position, citationPageLabel(citation)].filter(Boolean);
  title.append(makeElement("span", "", positionParts.join(" · ") || "正文位置未标注"));
  summary.append(title, makeElement("span", "expand-label", "展开原文"));

  const body = makeElement("div", "citation-body");
  const metadata = makeElement("div", "citation-metadata");
  (citation.retrieval_channels || []).forEach((channel) => {
    metadata.append(makeElement("span", "channel-chip", `通道：${channel}`));
  });
  (citation.exact_matches || []).forEach((match) => {
    metadata.append(makeElement("span", "exact-chip", `精确匹配：${match}`));
  });
  if (citation.reranker_score !== null && citation.reranker_score !== "" && Number.isFinite(Number(citation.reranker_score))) {
    metadata.append(makeElement("span", "meta-chip", `重排分数 ${Number(citation.reranker_score).toFixed(2)}`));
  }

  const quote = makeElement("blockquote", "quote-block", citation.quote || "暂无原文片段");
  const actions = makeElement("div", "source-actions");
  const sourceInfo = [citation.source_portal, citation.section_path].filter(Boolean).join(" · ");
  actions.append(makeElement("span", "", sourceInfo || "来源信息未标注"));

  const sourceUrl = safeHttpUrl(citation.source_url || citation.attachment_url);
  if (sourceUrl) {
    const link = makeElement("a", "", "访问官方来源 ↗");
    link.href = sourceUrl;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    actions.append(link);
  }

  body.append(metadata, quote, actions);
  card.append(summary, body);
  return card;
}

/** 渲染全部证据；拒答时没有正式 citation，会显示明确的空状态。 */
function renderCitations(citations) {
  elements.citations.replaceChildren();
  elements.citationCount.textContent = citations.length ? `共 ${citations.length} 条` : "未形成有效引用";

  if (!citations.length) {
    elements.citations.append(
      makeElement("p", "quote-block", "本次检索未通过证据充分性判断，因此没有将候选片段作为正式引用。"),
    );
    return;
  }
  citations.forEach((citation, index) => elements.citations.append(renderCitation(citation, index)));
}

/** 创建检索过程中的单个“字段名 + 值”项目。 */
function traceItem(label, value) {
  const wrapper = makeElement("dl", "trace-item");
  wrapper.append(makeElement("dt", "", label), makeElement("dd", "", String(value ?? "未提供")));
  return wrapper;
}

/**
 * 只展示面试和调试最有价值的检索字段，避免直接倾倒整段 JSON 影响阅读。
 * 这些字段能说明是否精确查询、是否触发多路召回以及 Reranker 是否生效。
 */
function renderTrace(agentTrace = {}, data = {}) {
    const actions = (agentTrace.actions || [])
      .map((action) => `${action.tool_name}(${action.status || "unknown"})`)
      .join(" → ");
    elements.trace.replaceChildren(
      traceItem("执行模式", agentTrace.mode === "rag" ? "基础检索" : "法律检索助手"),
      traceItem("意图", data.intent || "未提供"),
      traceItem("适用领域", data.domain || "uncertain"),
      traceItem("标准化问题", data.normalized_question || data.question),
      traceItem("法律争点", (data.legal_issues || []).join("；") || "未提供"),
      traceItem("检索轮数", agentTrace.iterations ?? 0),
      traceItem("检索操作数", agentTrace.tool_call_count ?? 0),
      traceItem("模型调用数", agentTrace.llm_call_count ?? 0),
      traceItem("工具轨迹", actions || "未调用搜索工具"),
      traceItem("停止原因", data.stop_reason || "未提供"),
      traceItem("可回答性", data.answerability || "未提供"),
      traceItem("待补充事实", (data.missing_user_facts || []).join("；") || "无"),
    );
}

/** 把一次完整 API 响应更新到结果区域。 */
function renderResult(data) {
  renderResultMeta(data);
  elements.answer.textContent = data.answer || "接口没有返回回答内容。";
  elements.disclaimer.textContent = data.disclaimer || "";
  renderCitations(data.citations || []);
  renderTrace(data.agent_trace || {}, data);

  const alertText = data.warning || data.refusal_reason || "";
  elements.answerAlert.hidden = !alertText && !data.insufficient_evidence;
  elements.answerAlert.classList.toggle("is-refusal", Boolean(data.insufficient_evidence));
  elements.answerAlert.textContent = alertText || "当前证据不足，系统已停止生成法律结论。";
  if (data.status === "needs_clarification") {
    elements.answerAlert.hidden = false;
    elements.answerAlert.classList.remove("is-refusal");
    elements.answerAlert.textContent = "需要补充信息：" + (data.clarifying_questions || []).join("；") + "。当前仅展示一般规则。";
  }
  elements.result.hidden = false;
}

/** 从 FastAPI 的 detail 字段中提取易读错误，不暴露原始响应对象。 */
async function responseError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === "string") return data.detail;
  } catch {
    // 非 JSON 错误响应会使用下方的通用提示。
  }
  return `服务返回 HTTP ${response.status}，请检查后台日志。`;
}

/**
 * 提交问答请求。浏览器发送的字段与 QuestionRequest 完全对应：question 是问题，
 * top_k 是最终希望返回的父法条数量。
 */
async function askQuestion(question, topK, mode) {
  if (activeRequest) return;
  const controller = new AbortController();
  activeRequest = controller;
  setLoading(true);
  hideFeedback();
  elements.result.hidden = true;

  try {
    const endpoint = mode === "agent" ? "/api/v1/agent/questions" : "/api/v1/questions";
    elements.loadingText.textContent = mode === "agent"
      ? "正在识别场景、查找适用法条并核对证据，请稍候..."
      : "正在查找与问题相关的法条，请稍候...";
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(mode === "agent" ? { question, top_k: topK, debug_trace: true, history } : { question, top_k: topK }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await responseError(response));
    const data = await response.json();
    if (controller.signal.aborted || activeRequest !== controller) return;
    renderResult(data);
    if (mode === "agent") history = [...history, { role: "user", content: question }].slice(-4);
    elements.result.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    if (error.name !== "AbortError") {
      showFeedback("问答请求失败", error.message || "无法连接后端服务，请稍后重试。");
    }
  } finally {
    if (activeRequest === controller) {
      setLoading(false);
      activeRequest = null;
    }
  }
}

/** 校验表单并开始提问；空字符串不会进入后端。 */
function handleSubmit(event) {
  event.preventDefault();
  const question = elements.question.value.trim();
  if (!question) {
    showFeedback("请输入问题", "问题不能为空，可以先点击下方的示例问题。 ");
    elements.question.focus();
    return;
  }
  askQuestion(question, Number(elements.topK.value), elements.mode.value);
}

/** 清空当前问题和回答，并取消可能仍在运行的网络请求。 */
function clearWorkspace() {
  if (activeRequest) activeRequest.abort();
  activeRequest = null;
  elements.question.value = "";
  history = [];
  elements.result.hidden = true;
  hideFeedback();
  setLoading(false);
  elements.question.focus();
}

/** 示例问题点击后直接填入并提交，让首次使用者快速看到完整检索链路。 */
function useExample(event) {
  elements.question.value = event.currentTarget.textContent.trim();
  elements.form.requestSubmit();
}

// 绑定页面交互。Ctrl+Enter 是文本输入区常用的提交方式，同时保留按钮操作。
elements.form.addEventListener("submit", handleSubmit);
elements.clear.addEventListener("click", clearWorkspace);
elements.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
document.querySelectorAll(".example-button").forEach((button) => {
  button.addEventListener("click", useExample);
});

checkHealth();
