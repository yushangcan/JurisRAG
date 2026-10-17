# JurisRAG 法律检索 Agent

面向中国政府采购与招投标的本地法规检索助手。配置模型后，大模型在循环中选择检索、读取法条、补查、追问和回答；程序执行只读工具并核验引用。主服务仍是单个 Python 进程，使用 SQLite FTS5，无需向量权重或独立工具服务。

没有模型配置或显式关闭模型时，使用离线规则检索与原文展示，并在 `agent_trace.controller` 中明确标为 `rule_workflow`，不计作模型 Agent 验证。模型决策调用失败时标为 `rule_fallback`。架构与验证方式见 [模型循环说明](docs/模型驱动Agent循环.md)。

当前包含 35 类常见业务场景。这里的“场景”是待查争点与检索规则，不是裁判案例库。系统提供可溯源的检索辅助，不能确认实时法规效力或替代专业法律判断。

## 快速运行

使用 Python 3.12，在项目根目录执行：

~~~powershell
python -m venv .venv-lite
.\.venv-lite\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
$env:RETRIEVAL_MODE='lexical'
.\.venv-lite\Scripts\python.exe -m offline_ingestion.build_index
.\.venv-lite\Scripts\python.exe run_api.py
~~~

打开 [本地页面](http://127.0.0.1:8000) 或 [API 文档](http://127.0.0.1:8000/docs)。已有 .venv-lite 时无需再次创建。在本机 `.env` 中设置 `ENABLE_LLM=true`、`LLM_PROVIDER` 与对应 Key、模型名。新 `.env.example` 已开启模型开关，但未填写 Key；未配置时仍能使用规则模式。旧 `.env` 中的预算可按下文更新。需要强制离线时，在启动前设置 `$env:ENABLE_LLM='false'`。

索引缺失时，接口会明确提示先建库。建库读取已提交并冻结摘要的 Markdown，不需要安装 PDF、Word 解析库。当前 80 份来源记录去重后得到 **66 份文档、1,565 个父法条/章节、2,299 个检索子块**。本机默认 SQLite 文件约 18.34 MiB。

## 这次如何减重

| 部分 | 原实现 | 当前默认实现 |
| --- | --- | --- |
| 检索存储 | Chroma 与向量模型 | 单个 SQLite 文件与 FTS5 |
| Agent 控制 | LangGraph 状态图 | 模型选择动作的异步循环 |
| 工具调用 | 强制独立 MCP 服务 | 进程内业务调用，MCP 可选 |
| 模型使用 | 多节点规划、生成、审查 | 同一个决策接口，观察结果后选择下一步 |
| 原文查询 | 复杂流程中的特例 | 确定性定位与父条回填，零模型调用 |
| 主依赖清单 | 15 项直接依赖 | 4 项运行直接依赖 |
| 解析、向量、评测 | 与主服务耦合 | 按用途单独安装 |

4 项直接依赖是 FastAPI、Pydantic、python-dotenv、Uvicorn；它们仍有正常的传递依赖。没有用另一套重型框架替代原框架。原来的官方语料、来源元数据、父子结构与人工可读历史报告保留。

## Agent 的闭环

~~~text
问题与最近用户追问
        ↓
规则边界与最低争点提示
        ↓
模型选择动作 ← 工具观察、历史动作、回答校验反馈
        ├─ search → 精确引用 / BM25 / 多查询 → 返回候选
        ├─ read_evidence → 读取实际父法条 → 返回正文
        ├─ answer → 核验ID、引文、数字、争点与最低法源
        │                  ├─ 通过 → 展示回答与依据
        │                  └─ 失败 → 把具体问题交回模型
        ├─ clarify → 请求补充事实
        └─ refuse → 说明依据或能力不足
~~~

默认最多 3 轮搜索、8 次模型决策、2 个并发请求。单次模型等待默认 20 秒，总请求等待默认 120 秒；配置上限分别为 6 轮、12 次、8 并发、60 秒、300 秒。`AGENT_MAX_ITERATIONS` 限制搜索次数，读取与回答也会消耗模型调用预算。重复动作、连续无新证据或预算耗尽均会停止。旧 `.env` 中的值仍会覆盖默认值，需要同步更新。基础 RAG 继续只检索一轮并可作一次模型归纳。

模型自行选择查询、精确查找参数和父证据ID，可以增加待查争点；已有争点与用户指定引用不能静默删除。程序检查正式引用和已有场景法源要求，未知场景的语义相关性由模型判断。候选必须读取父证据后才允许引用，最终最多 `top_k` 条；校验失败不会把候选变成正式引用。模型判断和程序规则都不能保证法律正确性。

## 常见场景与边界

| 场景 | 行为 |
| --- | --- |
| 明确法规名与条号、多法规多条号 | 逐条核验；不存在的条号不能用相似条文替代 |
| 资格与证明材料、联合体、关联供应商 | 按争点组织证据，保留适用制度差异 |
| 单一来源、询价报价、采购方式 | 查对应程序与条件原文 |
| 同品牌、评标方法、评标委员会 | 定位评审规则；具体项目需要相应事实 |
| 页数、装订、格式要求导致否决 | 区分设置条件与否决依据；事实不足时追问 |
| 质疑、答复、投诉 | 分开检索各环节；具体日期需要文书日期与官方工作日历 |
| 中小企业政策 | 同时检索原管理办法与库内后续政策；不保证截至今天完整 |
| 地域限制、虚假材料、责任 | 分开处理行为规则与责任 |
| 境外门槛、税率、地方年度限额 | 说明缺少对应法源，不猜测数额 |
| 实时效力、最新版本、完整废止清单 | 说明没有实时核验能力 |
| 胜诉预测、保证中标 | 不作结果保证 |
| 仅条号、问题过多 | 请求法规名称或拆分问题 |
| 伪造证据 ID、改写引文、无依据数字 | 将校验反馈交回模型修正；预算耗尽时拒答 |
| 模型接口失败或连续两次输出无效动作格式 | 在剩余搜索预算内使用规则与原文回退，并明确标记 |
| 总体超时 | 清空正式引用并停止等待 |

默认使用最近用户问题承接“那第十三条呢”“那需要什么材料”等追问。旧助手回答不作为法律证据；前端历史只保留在当前页面内存中，清空页面会清除历史。

模型归纳必须为每条陈述提供实际证据 ID 和连续原文引文。代码检查真实引文、数字、法规标题/条号和部分明显的越界结论，并展示原文。内容覆盖阈值与规则映射属于启发式检查，**不等于语义正确性的证明**；原文来源可信也不意味着具体项目当然适用。

## API 契约

| 接口 | 用途 |
| --- | --- |
| GET /live | 仅检查进程能响应 |
| GET /health | 索引与启用的可选模型文件检查 |
| GET /ready | 检索依赖缺失时返回 503 |
| POST /api/v1/questions | 一轮基础 RAG |
| POST /api/v1/agent/questions | 完整检索 Agent，支持 history |
| GET /api/v1/agent/health | Agent 配置、预算和容量 |
| GET /api/v1/cases | 当前场景目录 |
| GET /api/v1/evidence/{evidence_id} | 父证据完整原文及来源 |

问答请求：

~~~json
{"question":"供应商资格条件与证明材料分别有哪些？","top_k":5}
~~~

Agent 请求可增加 history（最多 6 条，每条最多 2,000 字）和 debug_trace。问题最多 2,000 字，top_k 为 1～10。返回 status：

- answered：通过当前证据门，可以展示原文或归纳；不代表具体法律结论已由专家确认。
- needs_clarification：需要补充法规名称或项目事实；有充分一般规则时仍可展示引用。
- refused：范围外、证据不足或预算终止；citations 为空。

同时返回 answerability、missing_user_facts、clarifying_questions、citations、generation_mode、stop_reason 与执行计数。Token 缺失时为 null，不估算费用。并发达到上限返回 503 与 Retry-After；参数错误返回 422，未知证据返回 404。请求 ID 在响应体与响应头一致；日志不写用户问题、原文、模型回复或密钥。

## 可选能力

依赖按用途拆分：

| 清单 | 用途 |
| --- | --- |
| requirements.txt | 默认轻量服务与从冻结 Markdown 建库 |
| requirements-dev.txt | 测试与离线回归 |
| requirements-ingest.txt | 新增/重新转换 HTML、PDF、Word 原文 |
| requirements-semantic.txt | 可选 BGE 向量与 ONNX 重排 |
| requirements-mcp.txt | 可选 MCP 对外工具出口 |

模型循环在本机 .env 中设置 ENABLE_LLM=true、LLM_PROVIDER 与对应 Key/模型名。支持 DeepSeek、DashScope、OpenAI 兼容接口；主依赖无需增加模型 SDK。不要提交或打印密钥。模型配置不完整或请求失败时仍可使用规则原文回退；明确法条原文查询始终不调用模型。

需要语义检索时安装 requirements-semantic.txt，使用 scripts.download_models 准备模型，设置 RETRIEVAL_MODE=hybrid、正确的 EMBEDDING_MODEL_PATH，再重建索引。可选 RERANKER_BACKEND=bge_reranker_base_onnx；默认 none。向量存在同一 SQLite，使用标准库余弦扫描，适合当前小型语料，不面向百万级数据。hash 向量仅用于确定性检查，不能称为语义模型。模型健康检查只检查文件与依赖，不代表推理已成功。

MCP 只在对接外部客户端时使用：

~~~powershell
.\.venv-lite\Scripts\python.exe -m pip install -r requirements-mcp.txt
.\.venv-lite\Scripts\python.exe run_mcp.py --transport stdio
~~~

默认暴露 6 个只读工具：精确、关键词、多查询、父证据、文档范围、争点覆盖；hybrid 才增加语义工具。主 Agent 不连接 MCP，不需要额外端口。

## 语料校验与更新

~~~powershell
.\.venv-lite\Scripts\python.exe -m offline_ingestion.verify_snapshots
~~~

所有来源必须与清单 SHA-256 一致，Markdown 必须与冻结摘要及原文哈希一致。建库先验证输入，再写临时库并原子替换；失败保留旧索引。Windows 下若文件被外部程序占用，请停止服务后重建。

新增原文需安装 requirements-ingest.txt，更新官方来源清单，分别运行 offline_ingestion.markdown_converter.convert_corpus 或 convert_legal_collection，检查其审计报告，再运行 offline_ingestion.verify_snapshots --refresh 更新摘要并重建。刷新不是法律专业审核，也不会查询今天的法规效力。第三方数据使用边界见 [DATA_NOTICE.md](DATA_NOTICE.md)。

## 验证与评测

~~~powershell
.\.venv-lite\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv-lite\Scripts\python.exe -m pip check
.\.venv-lite\Scripts\python.exe -m pytest -q tests
.\.venv-lite\Scripts\python.exe -m evaluation.run_lightweight_eval
.\.venv-lite\Scripts\python.exe -m evaluation.run_lightweight_eval --dataset evaluation/datasets/legal_rag_gold_v2.jsonl --output evaluation/results/lightweight_gold50_v2.json
~~~

默认评测显式关闭外部模型，统计最终正式引用的 Parent Recall、MRR、多法源全部命中、误拒、边界拒答、时延和调用数。--with-llm 必须明确配置模型，降级会单独标记；该选项也不会自动产生法律正确性评分。

规则版本历史报告见 [30 题回归](evaluation/results/lightweight_agent_v2.md) 与 [50 题回归](evaluation/results/lightweight_gold50_v2.md)。两者均为无模型开发回归，不是模型循环成绩；部分场景规则参考了开发反馈，不能当作独立测试集或线上准确率。历史 RAGAS 和旧框架报告仅作历史记录，不能拼接比较。

| 历史规则版本离线回归 | 基础 RAG | 规则流程 |
| --- | ---: | ---: |
| 30题 Parent Recall@5 | 0.8333 | 0.9762 |
| 30题边界正确拒答 | 9/9 | 9/9 |
| 50题 Parent Recall@5 | 0.9000 | 1.0000 |
| 50题有依据题误拒 | 5/50 | 0/50 |

以上是规则参与开发后的回归结果，检索全部命中不能表述为法律回答全部正确。实际验收与剩余缺口见 [轻量化验收记录](docs/轻量化验收报告.md)。

2026-10-17 的模型循环验收为 **101项测试通过、1项可选MCP协议测试跳过**；真实 qwen-plus 已完成普通路径和人为设置首轮证据遗漏的补查路径。补查用例执行2次搜索、5次模型决策，最后通过引用检查；这仅证明该用例的流程可用，不能作为模型准确率或成本均值。早期测试出现引用校验失败和JSON格式错误并回退，完整记录见 [模型循环验收记录](docs/模型循环验收记录.md)。

已在独立 Python 3.12 轻量环境验证建库、测试和本地 HTTP 请求；该环境没有 Chroma、LangChain、LangGraph、MCP、FastEmbed、Torch、NumPy。旧版本的可选MCP会话证据见历史验收。本轮未进行独立题集的模型质量评估、BGE实际推理、浏览器交互、容器运行或远端CI。

## 部署与回退

本机默认只监听 127.0.0.1:8000。服务没有账户隔离和鉴权；对外部署需要自行配置受控入口。

容器是可选方式：先 docker compose build，再 docker compose run --rm api python -m offline_ingestion.build_index，最后 docker compose up -d。SQLite 通过本地 data/index 挂载；Compose 配置已校验，但本次没有构建或运行容器。

模型循环分支为 feature/model-driven-agent-loop；改动前提交 bacf402 保存在 backup/pre-model-loop-20261017。更早的框架版本位于 backup/pre-lightweight-agent-20261016。切换前应先保存当前改动，旧 Chroma 文件未自动删除。

设计细节见 [项目设计与实现说明](docs/项目设计与实现说明.md)，项目讲解见 [Agent 项目整体介绍](docs/Agent项目整体介绍.md)。
