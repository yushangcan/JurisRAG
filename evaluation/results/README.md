# 评测结果目录

这里保留适合阅读和版本审查的 Markdown 报告。评测脚本运行时生成的 JSON 结果、逐题模型答案和评分缓存会被 `.gitignore` 忽略，因为它们包含运行时间、绝对路径、模型响应和可重新生成的临时数据。

## 当前报告

- `bge_multi_recall_refusal_v2_top5.md`：50 题父法条召回、MRR 和 10 题拒答结果。
- `ragas_full_final_v2.md`：RAGAS 五项指标的部分完成报告，明确记录 42/50 题完成评分和余额不足边界。

评测 JSON 输出目录已在根 `.gitignore` 中排除；README 中的 `retrieval_latest.json`、`abstention_latest.json` 和 RAGAS 缓存都不会自动进入提交。

## 固定 RAG 与 Agent 同题对比

`evaluation/datasets/legal_agent_rag_comparison_v1.jsonl` 是冻结的 30 题开发集：21
道有答案题、7 道多文件融合题和 9 道边界/拒答题。运行前先审计来源：

```powershell
\.venv-eval\Scripts\python.exe -m evaluation.audit_agent_rag_comparison
```

完整对比要求回答模型和 RAGAS 裁判模型都通过环境变量配置。下面的变量只在当前
PowerShell 进程中生效，密钥不会写入文件：

```powershell
$env:ENABLE_LLM='true'
$env:LLM_PROVIDER='dashscope'       # 使用 OpenAI-compatible ChatOpenAI 客户端
$env:DASHSCOPE_API_KEY='<answer-provider-key>'
$env:DASHSCOPE_BASE_URL='https://api-sss.zwenooo.link/v1'
$env:DASHSCOPE_MODEL='deepseek-chat'
$env:RAGAS_EVALUATOR_API_KEY='<evaluator-key>'
$env:RAGAS_EVALUATOR_BASE_URL='https://api-sss.zwenooo.link/v1'
$env:RAGAS_EVALUATOR_MODEL='deepseek-chat'
\.venv-eval\Scripts\python.exe -m evaluation.run_agent_rag_ragas_comparison
```

脚本会先做一次最小模型预检；预检失败时不会把降级答案当作正式结果。若只想先
验证两套系统的执行、检索和边界指标，可在有效回答模型配置下使用
`--skip-ragas`，但报告状态会明确标记为 `generation_complete_ragas_skipped`，不
会冒充 RAGAS 已完成。

## 重新生成

```powershell
cd 'D:\面试面经\项目\JurisRAG'
\.venv\Scripts\python.exe -m evaluation.run_retrieval_eval --dataset evaluation/datasets/legal_rag_gold_v2.jsonl --output evaluation/results/retrieval_latest.json --top-k 5
\.venv-eval\Scripts\python.exe -m evaluation.run_ragas_full_eval --concurrency 3
```

生成文件仅用于本地分析，不应直接提交。若需要共享一次实验，请把运行配置、Git commit、数据集版本和摘要指标整理到新的 Markdown 报告中。
