# JurisRAG 轻量法律检索 Agent

面向中国政府采购与招投标的本地法规检索助手。默认只运行一个 Python 进程，使用 SQLite FTS5 做中文全文检索；没有模型 Key、向量权重或独立工具服务，也能完成问题规划、检索、证据核验、原文回答与追问。

当前包含 35 类常见业务场景。这里的“场景”是待查争点与检索规则，不是裁判案例库。系统提供可溯源的检索辅助，不能确认实时法规效力或替代专业法律判断。

## 快速运行

使用 Python 3.12，在项目根目录执行：

~~~powershell
python -m venv .venv-lite
.\.venv-lite\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
$env:RETRIEVAL_MODE='lexical'
$env:ENABLE_LLM='false'
.\.venv-lite\Scripts\python.exe -m offline_ingestion.build_index
.\.venv-lite\Scripts\python.exe run_api.py
~~~

打开 [本地页面](http://127.0.0.1:8000) 或 [API 文档](http://127.0.0.1:8000/docs)。已有 .venv-lite 时无需再次创建。原来的 .env 可以保留；上述两个环境变量保证本次使用离线轻量模式，不会调用外部模型。

索引缺失时，接口会明确提示先建库。建库读取已提交并冻结摘要的 Markdown，不需要安装 PDF、Word 解析库。当前 80 份来源记录去重后得到 **66 份文档、1,565 个父法条/章节、2,299 个检索子块**。本机默认 SQLite 文件约 18.34 MiB。

## 这次如何减重

| 部分 | 原实现 | 当前默认实现 |
| --- | --- | --- |
| 检索存储 | Chroma 与向量模型 | 单个 SQLite 文件与 FTS5 |
| Agent 控制 | LangGraph 状态图 | 普通异步有限循环 |
| 工具调用 | 强制独立 MCP 服务 | 进程内业务调用，MCP 可选 |
| 模型使用 | 多节点规划、生成、审查 | 最多一次补检改写、一次归纳 |
| 原文查询 | 复杂流程中的特例 | 确定性定位与父条回填，零模型调用 |
| 主依赖清单 | 15 项直接依赖 | 4 项运行直接依赖 |
| 解析、向量、评测 | 与主服务耦合 | 按用途单独安装 |

4 项直接依赖是 FastAPI、Pydantic、python-dotenv、Uvicorn；它们仍有正常的传递依赖。没有用另一套重型框架替代原框架。原来的官方语料、来源元数据、父子结构与人工可读历史报告保留。

## Agent 的闭环

~~~text
问题与最近用户追问
        ↓
规则规划：领域、正式引用、业务场景、待补事实
        ↓
精确引用 + 中文 BM25，多查询融合
        ↓
逐引用、逐争点核验
        ├─ 缺口 → 有限补查 → 没有新查询/新证据或预算用尽则停止
        └─ 完整 → 重新选择证据 → 回填完整父条 → 再检查最终上下文
                                                    ↓
                                    原文回答，或可选的一次模型归纳
                                                    ↓
                                    核验引用；失败时回到原文
~~~

默认最多 2 轮检索、2 次模型调用、2 个并发请求。每次模型等待默认 20 秒，整体请求等待默认 50 秒。配置的上限分别是 2 轮、2 次、8 并发、25 秒与 60 秒。基础 RAG 使用同一存储、证据门和引用格式，只检索一轮，不使用场景法源补查。

规则只决定“查什么、去哪里查”。回答引用的正文必须从实际索引读取。累计候选重新排序，后续发现的证据可以替换早期结果。最终返回的证据数不足以覆盖全部争点时，系统拒答，不把未展示的候选算成有效引用。

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
| 模型超时、伪造证据 ID、改写引文、无依据数字 | 回到实际原文；总体超时则清空引用并停止等待 |

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

需要模型归纳时，在本机 .env 中设置 ENABLE_LLM=true、LLM_PROVIDER 与对应 Key/模型名。支持 DeepSeek、DashScope、OpenAI 兼容接口，默认不开启。不要提交或打印密钥。模型配置不完整或请求失败时仍可使用原文；明确法条原文查询始终不调用模型。

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

当前报告见 [30 题回归](evaluation/results/lightweight_agent_v2.md) 与 [50 题回归](evaluation/results/lightweight_gold50_v2.md)。两者均为开发回归，部分场景规则参考了开发反馈，不能当作独立测试集或线上准确率。历史 RAGAS 和旧框架报告仅作历史记录，不能拼接比较。

已在独立 Python 3.12 轻量环境验证建库、测试和真实本地 HTTP 请求；该环境没有 Chroma、LangChain、LangGraph、MCP、FastEmbed、Torch、NumPy。可选 MCP 的内存协议会话另行测试。当前未验证付费模型端到端质量、BGE 实际推理、浏览器交互、容器运行和远端 CI。

## 部署与回退

本机默认只监听 127.0.0.1:8000。服务没有账户隔离和鉴权；对外部署需要自行配置受控入口。

容器是可选方式：先 docker compose build，再 docker compose run --rm api python -m offline_ingestion.build_index，最后 docker compose up -d。SQLite 通过本地 data/index 挂载；Compose 配置已校验，但本次没有构建或运行容器。

重构分支为 refactor/lightweight-legal-agent；改动前提交保存在 backup/pre-lightweight-agent-20261016。切换前应先保存当前改动。旧版本依赖和运行方式可在备份分支查看，旧 Chroma 文件未自动删除。

设计细节见 [项目设计与实现说明](docs/项目设计与实现说明.md)，项目讲解见 [Agent 项目整体介绍](docs/Agent项目整体介绍.md)。
