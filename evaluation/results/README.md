# 评测结果目录

这里保留适合阅读和版本审查的 Markdown 报告。评测脚本运行时生成的 JSON 结果、逐题模型答案和评分缓存会被 `.gitignore` 忽略，因为它们包含运行时间、绝对路径、模型响应和可重新生成的临时数据。

## 当前报告

- `bge_multi_recall_refusal_v2_top5.md`：50 题父法条召回、MRR 和 10 题拒答结果。
- `ragas_full_final_v2.md`：RAGAS 五项指标的部分完成报告，明确记录 42/50 题完成评分和余额不足边界。

评测 JSON 输出目录已在根 `.gitignore` 中排除；README 中的 `retrieval_latest.json`、`abstention_latest.json` 和 RAGAS 缓存都不会自动进入提交。

## 重新生成

```powershell
cd 'D:\面试面经\项目\JurisRAG'
\.venv\Scripts\python.exe -m evaluation.run_retrieval_eval --dataset evaluation/datasets/legal_rag_gold_v2.jsonl --output evaluation/results/retrieval_latest.json --top-k 5
\.venv-eval\Scripts\python.exe -m evaluation.run_ragas_full_eval --concurrency 3
```

生成文件仅用于本地分析，不应直接提交。若需要共享一次实验，请把运行配置、Git commit、数据集版本和摘要指标整理到新的 Markdown 报告中。
