# 当前 CodeZ 版本 RAGAS 实测

- 运行状态：`complete_with_missing_scores`；生成 60/60；评分已结束 210/210。
- 生产版本：`dbde46cee3bc78de666694842210336ccad59233`；数据集：`legal_agent_rag_comparison_v1.jsonl`（开发回归集）。
- 模型：`codez/gpt-6.1-sol`，Responses，reasoning effort=max，store=false。回答与裁判使用同一模型。
- 30 道题 × 2 条链路；21 道有依据题 × 5 个 RAGAS 指标；9 道边界题只计算拒答与引用检查。参考答案和 gold 不进入生成链路。
- RAGAS 0.4.3 collections 原始指标；仅补充 JSON Schema 与中文输出要求。AnswerRelevancy strictness=3，使用本地 BAAI/bge-small-zh-v1.5 真实语义向量（512维）；两种问题均 query_embed。FactualCorrectness 为 F1、beta=1、atomicity=low、coverage=low。
- 上下文为最终正式引用的完整父证据，保持返回顺序；不是所有检索候选，也不等于模型曾看到的截断片段。所有交付答案原样评分，包括原文回退。无上下文的证据指标记为不适用，调用失败或非有限值留空。

| RAGAS 指标 | 固定 RAG 均值（有效/有依据题） | Agent 均值（有效/有依据题） | 配对有效题 | 配对 Agent−RAG |
| --- | ---: | ---: | ---: | ---: |
| context_precision | 1.0000 (18/21) | 0.9547 (19/21) | 17 | 0.0000 |
| context_recall | 0.9907 (18/21) | 0.9895 (19/21) | 17 | -0.0020 |
| faithfulness | 0.9902 (16/21) | 1.0000 (15/21) | 14 | 0.0084 |
| answer_relevancy | 0.7237 (21/21) | 0.7056 (21/21) | 21 | -0.0181 |
| factual_correctness | 0.2690 (21/21) | 0.1795 (20/21) | 20 | -0.0930 |

| 系统交付与开销 | 固定 RAG | Agent |
| --- | ---: | ---: |
| 有依据题返回证据数 / 21 | 18 | 19 |
| 有依据题误拒答数 / 21 | 3 | 2 |
| 边界题正确拒答数 / 9 | 9 | 9 |
| 引用ID或原文不匹配数 | 0 | 0 |
| 含告警的请求数 | 8 | 17 |
| 平均请求耗时 ms（并发最多2，非吞吐测试） | 17508.3353 | 74004.7943 |
| 最终引用 parent_recall_at_5 | 0.8333 | 0.8810 |
| 最终引用 mrr_at_5 | 0.7857 | 0.7857 |
| 最终引用 all_gold_hit_at_5 | 0.8095 | 0.8571 |
| 生成模型调用数（含失败） | 16 | 80 |
| 生成失败调用数 | 3 | 17 |
| 生成已知 total tokens（报告调用数/全部调用数） | 90785 (13/16) | 691614 (63/80) |
| 裁判模型调用数（含失败） | 271 | 305 |
| 裁判失败调用数 | 38 | 69 |
| 裁判已知 total tokens（报告调用数/全部调用数） | 1564685 (233/271) | 1611737 (236/305) |

## 解释与限制

- Context Precision/Recall 衡量最终交付证据对参考答案的支持；Faithfulness 衡量答案能否由这些证据支持；Answer Relevancy 衡量相关性；Factual Correctness 衡量与参考答案的事实一致性。它们不能替代法律专家或现行法效核验。
- 每项同时报告有效评分数与相同题目的配对差值；不同分母的均值不可直接作为优劣结论。拒答并未被删除：证据指标缺失，相关性和事实一致性继续评分。有效评分均值不能脱离返回率理解。
- 同模型作为回答者与裁判可能存在偏好；题集参与过开发，且样本较小。这次不是独立泛化测试，也没有统计显著性或法律正确性保证。参考答案的覆盖范围仍会影响 F1。
- 精确原文快路径保留在系统结果中，可能不调用模型。Agent 的校验失败、预算停止、超时和规则回退均保留；没有调整生产行为来追求分数。
- Token 只来自供应商 usage；未知用量留空，已知 token 可能只是下界。没有核验价格，不换算人民币成本。裁判开销单独列出，不算作线上业务成本。

## 控制器与输出类型

- rag controllers：`{"fixed_rag": 27, "exact_reference": 3}`
- rag generation_modes：`{"rag_llm": 8, "rag_exact_reference": 2, "rag_evidence_fallback": 8, "rag_abstention": 12}`
- agent controllers：`{"rule_fallback": 17, "exact_reference": 3, "llm_loop": 3, "rule_workflow": 7}`
- agent generation_modes：`{"agent_evidence_fallback": 14, "agent_exact_reference": 2, "agent_llm_loop": 3, "agent_abstention": 11}`

## 逐题结果

CP/CR/F/AR/FC 分别为上面五项指标；`—` 表示不适用、未完成或失败，详见完整 JSON。

| case_id | 模式 | 交付状态 | 输出类型 | CP | CR | F | AR | FC |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| case_outcome_009 | agent | refused | agent_abstention | — | — | — | — | — |
| case_outcome_009 | rag | refused | rag_abstention | — | — | — | — | — |
| collusion_rule_and_deeming_053 | agent | answered | agent_evidence_fallback | 0.6389 | 1.0000 | — | 0.7532 | 0.0000 |
| collusion_rule_and_deeming_053 | rag | refused | rag_abstention | — | — | — | 0.0000 | 0.0000 |
| deemed_collusive_bid_045 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.6425 | 0.0000 |
| deemed_collusive_bid_045 | rag | answered | rag_llm | 1.0000 | 1.0000 | 0.8824 | 0.9001 | 0.0000 |
| demand_investigation_005 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.8830 | — |
| demand_investigation_005 | rag | answered | rag_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.8793 | 0.2000 |
| engineering_regime_and_scope_050 | agent | refused | agent_abstention | — | — | — | 0.0000 | 0.0000 |
| engineering_regime_and_scope_050 | rag | refused | rag_abstention | — | — | — | 0.0000 | 0.0000 |
| eu_threshold_004 | agent | refused | agent_abstention | — | — | — | — | — |
| eu_threshold_004 | rag | refused | rag_abstention | — | — | — | — | — |
| evaluation_committee_size_046 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.8668 | 0.3300 |
| evaluation_committee_size_046 | rag | answered | rag_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.8564 | 0.4000 |
| evaluation_method_and_committee_054 | agent | answered | agent_evidence_fallback | 1.0000 | 0.8000 | 1.0000 | 0.7979 | 0.3300 |
| evaluation_method_and_committee_054 | rag | answered | rag_llm | 1.0000 | 0.8333 | 1.0000 | 0.7593 | 0.6700 |
| government_engineering_law_025 | agent | answered | agent_llm_loop | 1.0000 | 1.0000 | 1.0000 | 0.8759 | 0.4000 |
| government_engineering_law_025 | rag | answered | rag_llm | 1.0000 | 1.0000 | 1.0000 | 0.8736 | 1.0000 |
| minimum_registered_capital_007 | agent | refused | agent_abstention | — | — | — | — | — |
| minimum_registered_capital_007 | rag | refused | rag_abstention | — | — | — | — | — |
| missing_gpl_article_001 | agent | refused | agent_abstention | — | — | — | — | — |
| missing_gpl_article_001 | rag | refused | rag_abstention | — | — | — | — | — |
| missing_tender_article_002 | agent | refused | agent_abstention | — | — | — | — | — |
| missing_tender_article_002 | rag | refused | rag_abstention | — | — | — | — | — |
| procurement_definition_011 | agent | answered | agent_exact_reference | 1.0000 | 1.0000 | 1.0000 | 0.8367 | 0.3300 |
| procurement_definition_011 | rag | answered | rag_exact_reference | 1.0000 | 1.0000 | 1.0000 | 0.8339 | 0.5000 |
| procurement_files_retention_019 | agent | answered | agent_llm_loop | 1.0000 | 1.0000 | 1.0000 | 0.9004 | 0.0000 |
| procurement_files_retention_019 | rag | answered | rag_llm | 1.0000 | 1.0000 | 1.0000 | 0.9555 | 0.0000 |
| procurement_methods_002 | agent | refused | agent_abstention | — | — | — | 0.0000 | 0.0000 |
| procurement_methods_002 | rag | answered | rag_llm | 1.0000 | 1.0000 | 1.0000 | 0.9005 | 0.6700 |
| procurement_principles_012 | agent | answered | agent_exact_reference | 1.0000 | 1.0000 | 1.0000 | 0.9663 | 0.0000 |
| procurement_principles_012 | rag | answered | rag_exact_reference | 1.0000 | 1.0000 | 1.0000 | 0.9615 | 0.0000 |
| procurement_recusal_026 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.7751 | 0.0000 |
| procurement_recusal_026 | rag | answered | rag_evidence_fallback | 1.0000 | 1.0000 | — | 0.7718 | 0.0000 |
| public_service_opinion_and_investigation_055 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | — | 0.7952 | 0.1100 |
| public_service_opinion_and_investigation_055 | rag | answered | rag_evidence_fallback | 1.0000 | 1.0000 | — | 0.8138 | 0.1200 |
| question_and_reply_period_049 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.7513 | 0.0000 |
| question_and_reply_period_049 | rag | answered | rag_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.7493 | 0.4000 |
| regional_discrimination_and_liability_052 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | — | 0.7287 | 0.0000 |
| regional_discrimination_and_liability_052 | rag | answered | rag_evidence_fallback | 1.0000 | 1.0000 | 0.9615 | 0.7221 | 0.0000 |
| regional_performance_discrimination_030 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.5942 | 0.0000 |
| regional_performance_discrimination_030 | rag | answered | rag_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.5975 | 0.0000 |
| same_brand_bidders_044 | agent | answered | agent_llm_loop | 1.0000 | 1.0000 | 1.0000 | 0.9156 | 0.4000 |
| same_brand_bidders_044 | rag | answered | rag_llm | 1.0000 | 1.0000 | 1.0000 | 0.9354 | 0.2900 |
| shanghai_future_threshold_005 | agent | refused | agent_abstention | — | — | — | — | — |
| shanghai_future_threshold_005 | rag | refused | rag_abstention | — | — | — | — | — |
| supplier_conditions_and_materials_051 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.9082 | 0.4000 |
| supplier_conditions_and_materials_051 | rag | answered | rag_llm | 1.0000 | 1.0000 | 1.0000 | 0.8785 | 0.4000 |
| supplier_false_material_penalty_023 | agent | needs_clarification | agent_evidence_fallback | 0.5000 | 1.0000 | — | 0.0000 | 0.2900 |
| supplier_false_material_penalty_023 | rag | refused | rag_abstention | — | — | — | 0.0000 | 0.0000 |
| supplier_qualification_001 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.9297 | 1.0000 |
| supplier_qualification_001 | rag | answered | rag_llm | 1.0000 | 1.0000 | 1.0000 | 0.9359 | 1.0000 |
| supplier_question_period_009 | agent | answered | agent_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.8960 | 0.0000 |
| supplier_question_period_009 | rag | answered | rag_evidence_fallback | 1.0000 | 1.0000 | 1.0000 | 0.8728 | 0.0000 |
| tax_rate_008 | agent | refused | agent_abstention | — | — | — | — | — |
| tax_rate_008 | rag | refused | rag_abstention | — | — | — | — | — |
| us_federal_rule_003 | agent | refused | agent_abstention | — | — | — | — | — |
| us_federal_rule_003 | rag | refused | rag_abstention | — | — | — | — | — |
| validity_inventory_010 | agent | refused | agent_abstention | — | — | — | — | — |
| validity_inventory_010 | rag | refused | rag_abstention | — | — | — | — | — |

## 缺失与失败

- procurement_methods_002:agent / context_precision：not_applicable，empty_final_contexts。
- procurement_methods_002:agent / context_recall：not_applicable，empty_final_contexts。
- procurement_methods_002:agent / faithfulness：not_applicable，empty_final_contexts。
- procurement_recusal_026:rag / faithfulness：failed，ModelUnavailable。
- supplier_false_material_penalty_023:rag / context_precision：not_applicable，empty_final_contexts。
- supplier_false_material_penalty_023:rag / context_recall：not_applicable，empty_final_contexts。
- supplier_false_material_penalty_023:rag / faithfulness：not_applicable，empty_final_contexts。
- supplier_false_material_penalty_023:agent / faithfulness：failed，ModelUnavailable。
- demand_investigation_005:agent / factual_correctness：failed，RemoteDisconnected。
- engineering_regime_and_scope_050:rag / context_precision：not_applicable，empty_final_contexts。
- engineering_regime_and_scope_050:rag / context_recall：not_applicable，empty_final_contexts。
- engineering_regime_and_scope_050:rag / faithfulness：not_applicable，empty_final_contexts。
- engineering_regime_and_scope_050:agent / context_precision：not_applicable，empty_final_contexts。
- engineering_regime_and_scope_050:agent / context_recall：not_applicable，empty_final_contexts。
- engineering_regime_and_scope_050:agent / faithfulness：not_applicable，empty_final_contexts。
- collusion_rule_and_deeming_053:rag / context_precision：not_applicable，empty_final_contexts。
- collusion_rule_and_deeming_053:rag / context_recall：not_applicable，empty_final_contexts。
- collusion_rule_and_deeming_053:rag / faithfulness：not_applicable，empty_final_contexts。
- regional_discrimination_and_liability_052:agent / faithfulness：failed，RemoteDisconnected。
- collusion_rule_and_deeming_053:agent / faithfulness：failed，ModelUnavailable。
- public_service_opinion_and_investigation_055:rag / faithfulness：failed，ModelUnavailable。
- public_service_opinion_and_investigation_055:agent / faithfulness：failed，ModelUnavailable。

## 复现与冻结信息

- 开始 UTC：2026-10-17T13:23:59.756547+00:00；最后保存 UTC：2026-10-17T15:25:24.570319+00:00。
- 完整答案、证据、子调用缓存与 usage：`current_codez_ragas_20261017.json`（本机保存，遵循仓库规则不提交 JSON）。
- 同一路径重跑会检查数据/索引/生产代码/模型/评分参数签名，已有答案不会重新生成；成功评分子调用会复用，三个反问采样使用不同编号。`--retry-failed` 只重试失败评分，不重抽业务答案。

```powershell
.\.venv-lite\Scripts\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase generate
.\.venv-eval\Scripts\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase score
.\.venv-lite\Scripts\python.exe -u -X utf8 -m evaluation.run_current_ragas --phase report
```

```json
{
  "signature": {
    "dataset_sha256": "5169593c3be942fde8719f340e1737c5bfaeca25d199cd5acc5cd0ea032456cc",
    "index_sha256": "4ab869fd4428bf1a04d29e181bf9f6f8c0ff07cdad1a13e5137ff223b494daa0",
    "runtime_sha256": "b9c6240deb120d7541d248116102d0b4995f6ca64538db5204f94fc079aa892b",
    "model": {
      "provider": "codez",
      "base_url": "https://api-sss.zwenooo.link/v1",
      "model": "gpt-6.1-sol",
      "wire_api": "responses",
      "reasoning_effort": "max",
      "disable_response_storage": true
    },
    "agent_settings": {
      "enabled": true,
      "max_iterations": 3,
      "max_llm_calls": 8,
      "llm_timeout_seconds": 60,
      "request_timeout_seconds": 300,
      "max_question_chars": 2000,
      "max_concurrent_requests": 2
    },
    "retrieval_mode": "lexical",
    "top_k": 5,
    "ragas_version": "0.4.3",
    "answer_relevancy_strictness": 3,
    "factual_correctness": {
      "mode": "f1",
      "beta": 1.0,
      "atomicity": "low",
      "coverage": "low"
    },
    "embedding": {
      "model": "BAAI/bge-small-zh-v1.5",
      "role": "query_embed_for_both_questions",
      "files": {
        "model_optimized.onnx": "1294ea4b6331115a353d81f96b85e8c8d7fdcc284453d5b2fab5b016230aad38",
        "config.json": "9088751d39abbf86ec3d19ffca92ad62ad19075f7e59712e6c71217fa125d1d3",
        "tokenizer.json": "48cea5d44424912a6fd1ea647bf4fe50b55ab8b1e5879c3275f80e339e8fae26",
        "tokenizer_config.json": "e6f3b96db926a37d4039995fbf5ad17de158dfb8f6343d607e4dbaad18d75f5a"
      }
    },
    "context_policy": "full_final_citation_quotes_in_returned_order",
    "judge_prompt_adapter": "stock_ragas_plus_chinese_output_and_json_schema_v1",
    "judge_timeout_seconds": 120,
    "judge_attempts_per_call": 2
  },
  "generation_environment": {
    "python": "3.12.14",
    "packages": {
      "pydantic": "2.13.5"
    }
  },
  "scoring_environment": {
    "python": "3.12.14",
    "packages": {
      "ragas": "0.4.3",
      "pydantic": "2.13.5",
      "openai": "2.54.0",
      "instructor": "1.16.0",
      "fastembed": "0.7.4",
      "onnxruntime": "1.29.0",
      "numpy": "2.5.2",
      "langchain-community": "0.3.31"
    }
  },
  "generation_concurrency": 2,
  "scoring_concurrency": 8,
  "generation_runner_sha256": "be67e672ec2d517235614637cf0bd1b16e0a08b17f197cf79d1862c7296b244e",
  "scoring_source_sha256": "e61c76d3a4317674ee66fcfb53ca30cb7bea9b2f09eb07297b0c34f44095c3a7"
}
```
