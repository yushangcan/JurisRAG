# 轻量法律检索 Agent 开发集回归

- 状态：offline_complete；题集：legal_agent_rag_comparison_v1.jsonl；Top-K=5；检索模式：lexical
- Git：bacf402f45053422466534587d798dfa2aa8373c；存在未提交文件：True
- 数据集 SHA-256：5169593c3be942fde8719f340e1737c5bfaeca25d199cd5acc5cd0ea032456cc
- 实现 SHA-256：e02570f7320a11236f1e72d95c743887d2a7785b276152fb868db8bdb044f713
- 索引 SHA-256：4ab869fd4428bf1a04d29e181bf9f6f8c0ff07cdad1a13e5137ff223b494daa0；时间 UTC：2026-10-17T06:35:28.058746+00:00
- 显式请求外部模型：False；没有评判模型回答的法律正确性。耗时是本机顺序运行测量，不代表并发能力。

| 指标 | 基础 RAG | 法律检索 Agent |
| --- | ---: | ---: |
| Parent Recall@1 | 0.6429 | 0.6667 |
| Parent Recall@3 | 0.8333 | 0.9762 |
| Parent Recall@5 | 0.8333 | 0.9762 |
| MRR@5 | 0.7857 | 0.8810 |
| 有答案题误拒答数 | 3 | 0 |
| 边界题正确拒答数 | 9 | 9 |
| 多法源全部命中@5 | 0.5714 | 0.8571 |
| 平均耗时 ms | 60.1800 | 98.3723 |
| 外部模型调用数 | 0 | 0 |

指标基于最终正式引用的父法条ID。证据不足时清空引用，因此召回结果也体现保守停止的代价。

| case_id | 模式 | 状态 | Recall@5 | 停止原因 |
| --- | --- | --- | ---: | --- |
| procurement_definition_011 | rag | answered | 1.0000 | evidence_complete |
| procurement_definition_011 | agent | answered | 1.0000 | evidence_complete |
| procurement_principles_012 | rag | answered | 1.0000 | evidence_complete |
| procurement_principles_012 | agent | answered | 1.0000 | evidence_complete |
| supplier_qualification_001 | rag | answered | 1.0000 | evidence_complete |
| supplier_qualification_001 | agent | answered | 1.0000 | evidence_complete |
| procurement_methods_002 | rag | answered | 1.0000 | evidence_complete |
| procurement_methods_002 | agent | answered | 1.0000 | evidence_complete |
| supplier_question_period_009 | rag | answered | 1.0000 | evidence_complete |
| supplier_question_period_009 | agent | answered | 1.0000 | evidence_complete |
| procurement_files_retention_019 | rag | answered | 1.0000 | evidence_complete |
| procurement_files_retention_019 | agent | answered | 1.0000 | evidence_complete |
| procurement_recusal_026 | rag | answered | 1.0000 | evidence_complete |
| procurement_recusal_026 | agent | answered | 1.0000 | evidence_complete |
| regional_performance_discrimination_030 | rag | answered | 1.0000 | evidence_complete |
| regional_performance_discrimination_030 | agent | answered | 1.0000 | evidence_complete |
| supplier_false_material_penalty_023 | rag | refused | 0.0000 | iteration_budget_exhausted |
| supplier_false_material_penalty_023 | agent | needs_clarification | 1.0000 | evidence_complete |
| government_engineering_law_025 | rag | answered | 1.0000 | evidence_complete |
| government_engineering_law_025 | agent | answered | 1.0000 | evidence_complete |
| same_brand_bidders_044 | rag | answered | 1.0000 | evidence_complete |
| same_brand_bidders_044 | agent | answered | 1.0000 | evidence_complete |
| deemed_collusive_bid_045 | rag | answered | 1.0000 | evidence_complete |
| deemed_collusive_bid_045 | agent | answered | 1.0000 | evidence_complete |
| evaluation_committee_size_046 | rag | answered | 1.0000 | evidence_complete |
| evaluation_committee_size_046 | agent | answered | 1.0000 | evidence_complete |
| demand_investigation_005 | rag | answered | 1.0000 | evidence_complete |
| demand_investigation_005 | agent | answered | 1.0000 | evidence_complete |
| question_and_reply_period_049 | rag | answered | 1.0000 | evidence_complete |
| question_and_reply_period_049 | agent | answered | 1.0000 | evidence_complete |
| engineering_regime_and_scope_050 | rag | refused | 0.0000 | iteration_budget_exhausted |
| engineering_regime_and_scope_050 | agent | answered | 1.0000 | evidence_complete |
| supplier_conditions_and_materials_051 | rag | answered | 1.0000 | evidence_complete |
| supplier_conditions_and_materials_051 | agent | answered | 1.0000 | evidence_complete |
| regional_discrimination_and_liability_052 | rag | answered | 1.0000 | evidence_complete |
| regional_discrimination_and_liability_052 | agent | answered | 1.0000 | evidence_complete |
| collusion_rule_and_deeming_053 | rag | refused | 0.0000 | iteration_budget_exhausted |
| collusion_rule_and_deeming_053 | agent | answered | 1.0000 | evidence_complete |
| evaluation_method_and_committee_054 | rag | answered | 0.5000 | evidence_complete |
| evaluation_method_and_committee_054 | agent | answered | 0.5000 | evidence_complete |
| public_service_opinion_and_investigation_055 | rag | answered | 1.0000 | evidence_complete |
| public_service_opinion_and_investigation_055 | agent | answered | 1.0000 | evidence_complete |
| missing_gpl_article_001 | rag | refused | - | formal_reference_missing |
| missing_gpl_article_001 | agent | refused | - | formal_reference_missing |
| missing_tender_article_002 | rag | refused | - | formal_reference_missing |
| missing_tender_article_002 | agent | refused | - | formal_reference_missing |
| us_federal_rule_003 | rag | refused | - | scope_or_capability_boundary |
| us_federal_rule_003 | agent | refused | - | scope_or_capability_boundary |
| eu_threshold_004 | rag | refused | - | scope_or_capability_boundary |
| eu_threshold_004 | agent | refused | - | scope_or_capability_boundary |
| shanghai_future_threshold_005 | rag | refused | - | scope_or_capability_boundary |
| shanghai_future_threshold_005 | agent | refused | - | scope_or_capability_boundary |
| minimum_registered_capital_007 | rag | refused | - | scope_or_capability_boundary |
| minimum_registered_capital_007 | agent | refused | - | scope_or_capability_boundary |
| tax_rate_008 | rag | refused | - | scope_or_capability_boundary |
| tax_rate_008 | agent | refused | - | scope_or_capability_boundary |
| case_outcome_009 | rag | refused | - | scope_or_capability_boundary |
| case_outcome_009 | agent | refused | - | scope_or_capability_boundary |
| validity_inventory_010 | rag | refused | - | scope_or_capability_boundary |
| validity_inventory_010 | agent | refused | - | scope_or_capability_boundary |
