# 评测结果说明

当前实现使用 evaluation.run_lightweight_eval。默认 model=None，不连接模型供应商；同时运行一轮基础 RAG 和有限检索 Agent，使用相同语料、证据门和返回数。

2026-10-17已新增模型选择动作的循环；默认评测仍是离线规则回归，其结果不能当作模型循环成绩。真实qwen-plus普通路径、受控遗漏补查、早期失败和供应商用量见 [模型循环验收记录](../../docs/模型循环验收记录.md)。

## 当前报告

- [current_codez_ragas_20261017.md](current_codez_ragas_20261017.md)：当前 CodeZ 模型版的30题实测，固定 RAG 与 Agent 并列；21道有依据题计算五项真实 RAGAS 指标，9道边界题检查拒答。报告包含有效评分数、回退、Token和配对差值，以报告状态为准。
- [lightweight_agent_v2.md](lightweight_agent_v2.md)：既有30题开发回归，其中21题有库内依据、9题是范围或能力边界。
- [lightweight_gold50_v2.md](lightweight_gold50_v2.md)：既有50题检索开发回归。
- 两份报告都保存逐题状态、最终引用 Recall、停止原因、题集/实现/索引 SHA-256 与运行时间。
- 对应完整 JSON 位于本机同目录，按仓库规则不提交；重新运行可生成。

## 复现

在项目根目录、已建库的 Python 3.12 环境执行：

~~~powershell
python -m evaluation.audit_agent_rag_comparison
python -m evaluation.run_lightweight_eval
python -m evaluation.run_lightweight_eval --dataset evaluation/datasets/legal_rag_gold_v2.jsonl --output evaluation/results/lightweight_gold50_v2.json
~~~

--with-llm 会明确调用已配置模型。没有模型时直接报错，不假装完成；调用失败或归纳校验失败会记录 model_run_degraded。该选项不自动提供回答质量裁判。

当前版本 RAGAS 使用单独的评测环境，不给主服务增加依赖。先安装 `evaluation/requirements-eval.txt`，并准备本地 `BAAI/bge-small-zh-v1.5` ONNX 权重；`--embedding-path` 指定含 `model_optimized.onnx` 的目录。问题相似度使用真实中文语义向量，不使用字符 hash。

~~~powershell
.\.venv-lite\Scripts\python.exe -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r evaluation/requirements-eval.txt
.\.venv-lite\Scripts\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase generate
.\.venv-eval\Scripts\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase score
.\.venv-lite\Scripts\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase report
~~~

生成和裁判都会调用本机已配置模型并产生供应商用量。答案按题保存，评分按子调用保存；相同签名重跑复用缓存，`--retry-failed` 仅重试失败评分。要做独立重复实验需指定新的 `--output`，不能覆盖既有业务答案。无上下文与失败的指标不补零，报告同时显示配对有效题数；同模型裁判和开发集偏差必须一并解释。

## 指标边界

Parent Recall@K 为前K个正式引用覆盖的 gold 父证据比例，按题取平均；MRR@5 是第一个 gold 命中的倒数排名。多法源全部命中要求一题的所有 gold 都进入前5条。多条证据不一定来自不同文件，因此这个指标更准确地理解为“多 gold 全部命中”。

拒答的正式引用为空，因此最终召回指标包含保守拒答的代价。正确边界拒答要求应拒答且没有引用。needs_clarification 可能仍展示充分的一般规则，不能据此认定具体项目问题已解决。耗时是本机顺序测试结果，不代表并发吞吐。

这些题集用于开发与回归，部分业务规则参考了开发失败样本。没有独立盲测、法律专家评分或当前法效核验，不能把 Recall 或原文返回称为法律回答准确率。

## 历史报告

其余既有 Markdown 是重构前结果，保留用于审计。旧 Chroma、LangGraph、RAGAS 运行脚本已从主流程移除，需要复现时查看 backup/pre-lightweight-agent-20261016。

历史 RAGAS 对比包含配置、余额或评分有效覆盖不足的问题，不能与当前离线检索指标直接比较，不能补算为零，也不能表述为本版本质量结论。原有 smoke_gpt54 等个人未跟踪文件不属于本次交付。
