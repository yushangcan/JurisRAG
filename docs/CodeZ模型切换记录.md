# CodeZ模型切换记录

日期：2026-10-17。分支：`feature/codez-responses`。切换前提交 `482363d0e35cb19e1f6564443a38e186c931d624` 保存在 `backup/pre-codez-responses-20261017`。

## 生效范围

Agent的动作决策、补查后的再次决策、最终回答、固定RAG归纳，以及 `evaluation.run_lightweight_eval --with-llm` 的答案生成，共用同一模型工厂，统一使用以下配置：

| 项目 | 配置 |
| --- | --- |
| 供应商 | `codez` |
| 模型 | `gpt-6.1-sol` |
| 基础地址 | `https://api-sss.zwenooo.link/v1` |
| 请求地址 | `POST /v1/responses` |
| 认证 | 用户提供的Key，通过Bearer认证发送 |
| 推理强度 | `reasoning.effort=max` |
| 响应存储 | `store=false`，对应 `DISABLE_RESPONSE_STORAGE=true` |
| 输出格式 | `text.format.type=json_object`，继续执行动作Schema与引文校验 |

真实密钥仅保存在本机已忽略的 `.env`，示例配置不包含密钥。旧供应商支持保留，但当前配置不会调用它们；模型故障时回退规则与原文。未修改历史评测报告中的实际模型名称。

本次没有新增RAGAS评分流程；现有 `--with-llm` 是答案生成和检索回归，不是RAGAS裁判。旧版RAGAS脚本位于历史分支，未在本次切换中运行。

用户提供的TOML是Codex客户端格式，本项目使用 `.env` 映射相关字段。`notify` 和 `[windows]` 不参与项目模型请求，没有执行桌面通知程序或修改全局Codex配置。`requires_openai_auth=true` 对应请求的Bearer认证，不需要为本项目引入Codex登录流程。

## 协议与运行限制

Responses请求不发送Chat专用的messages、temperature、max_tokens或response_format。输出预算为8,192 Token，包含推理Token；单次未完成响应不能作为有效动作执行。只提取完成响应中assistant消息的output_text，不提取推理内容。未完成或无正文的失败若返回用量，保留供应商Token记录。

CodeZ的本机配置与 `.env.example` 使用单次等待60秒、总体300秒；3次搜索和8次模型调用限制保持不变。代码在没有环境覆盖时仍使用20秒/120秒默认值。

`store=false`是请求给网关的接口参数，不保证第三方网关的日志保留策略。模型标识由用户指定，最小真实请求返回的模型名为 `gpt-6.1-sol`；客户端没有验证网关底层模型身份。

## 验证结果

Python 3.12.14，`.venv-lite`。全量测试为 **107 passed, 1 skipped**，两项测试依赖弃用提示；跳过的是未安装可选MCP包的协议测试。`pip check`通过。

新增或扩展的本地HTTP测试覆盖Responses请求路径与参数、仅提取正文、实际字段格式的用量读取，以及未完成、执行失败、拒绝输出和HTTP错误时的规则回退。保留旧Chat协议回归。测试中的Token数来自本地模拟服务，不能当作供应商消费。

真实CodeZ调用结果：

| 验证 | 结果 | 模型调用 | 供应商输入/输出/总Token | 本次耗时 |
| --- | --- | ---: | --- | ---: |
| 最小JSON请求 | HTTP 200、completed，返回合法JSON及同名模型标识 | 1 | 5,125 / 23 / 5,148 | 3.16秒 |
| Agent：哪些情形可以采用单一来源采购？ | search→read_evidence→answer；`agent_llm_loop`，无警告，引用政府采购法第三十一条 | 3 | 28,450 / 2,450 / 30,900 | 60.08秒 |
| 固定RAG：同一问题 | Responses调用成功，归纳未通过陈述与引文的内容对应检查，返回 `rag_evidence_fallback` | 1 | 5,503 / 1,562 / 7,065 | 28.88秒 |

RAG调用确实使用新模型，但本次归纳没有通过全部证据检查，不能称为归纳成功；保留检查与原文回退，没有为切换模型放宽验证规则。上述结果为单题流程验证，不是质量或速度对比，不推导平均成本。

Agent和RAG的安全摘要保存在本机 `tmp/codez-responses-smoke-20261017.json`，只包含模式、动作、引用位置、供应商用量和错误元数据，不保存Key、完整提示或模型推理内容。

被测源码与索引摘要：

```text
implementation_sha256=5a7aa4278b4e9d37133999e71d29d8a79972262f2c79502c01a22a73a3d325e6
index_sha256=4ab869fd4428bf1a04d29e181bf9f6f8c0ff07cdad1a13e5137ff223b494daa0
```

实现摘要使用 `evaluation.run_lightweight_eval.implementation_hash()`。切换过程中只读取现有索引。没有进行RAGAS评分、独立题集质量比较、浏览器交互、容器或远端CI验证。
