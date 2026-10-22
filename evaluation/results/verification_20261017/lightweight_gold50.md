# 轻量法律检索 Agent 开发集回归

- 状态：offline_complete；题集：legal_rag_gold_v2.jsonl；Top-K=5；检索模式：lexical
- Git：bacf402f45053422466534587d798dfa2aa8373c；存在未提交文件：True
- 数据集 SHA-256：457454ddb6f1049064e7601e1dab7933c28a05b88478a69da6d7c1971ac3ed2a
- 实现 SHA-256：e02570f7320a11236f1e72d95c743887d2a7785b276152fb868db8bdb044f713
- 索引 SHA-256：4ab869fd4428bf1a04d29e181bf9f6f8c0ff07cdad1a13e5137ff223b494daa0；时间 UTC：2026-10-17T06:35:56.819773+00:00
- 显式请求外部模型：False；没有评判模型回答的法律正确性。耗时是本机顺序运行测量，不代表并发能力。

| 指标 | 基础 RAG | 法律检索 Agent |
| --- | ---: | ---: |
| Parent Recall@1 | 0.6600 | 0.6900 |
| Parent Recall@3 | 0.8800 | 0.9800 |
| Parent Recall@5 | 0.9000 | 1.0000 |
| MRR@5 | 0.7617 | 0.8317 |
| 有答案题误拒答数 | 5 | 0 |
| 边界题正确拒答数 | 0 | 0 |
| 多法源全部命中@5 | 0.5000 | 1.0000 |
| 平均耗时 ms | 45.8318 | 92.1918 |
| 外部模型调用数 | 0 | 0 |

指标基于最终正式引用的父法条ID。证据不足时清空引用，因此召回结果也体现保守停止的代价。

| case_id | 模式 | 状态 | Recall@5 | 停止原因 |
| --- | --- | --- | ---: | --- |
| supplier_qualification_001 | rag | answered | 1.0000 | evidence_complete |
| supplier_qualification_001 | agent | answered | 1.0000 | evidence_complete |
| procurement_methods_002 | rag | answered | 1.0000 | evidence_complete |
| procurement_methods_002 | agent | answered | 1.0000 | evidence_complete |
| mandatory_tender_projects_003 | rag | refused | 0.0000 | iteration_budget_exhausted |
| mandatory_tender_projects_003 | agent | answered | 1.0000 | evidence_complete |
| bid_collusion_004 | rag | answered | 1.0000 | evidence_complete |
| bid_collusion_004 | agent | answered | 1.0000 | evidence_complete |
| demand_investigation_005 | rag | answered | 1.0000 | evidence_complete |
| demand_investigation_005 | agent | answered | 1.0000 | evidence_complete |
| sme_reserved_share_006 | rag | answered | 1.0000 | evidence_complete |
| sme_reserved_share_006 | agent | answered | 1.0000 | evidence_complete |
| infrastructure_scope_007 | rag | answered | 1.0000 | evidence_complete |
| infrastructure_scope_007 | agent | answered | 1.0000 | evidence_complete |
| framework_purchase_scope_008 | rag | answered | 1.0000 | evidence_complete |
| framework_purchase_scope_008 | agent | answered | 1.0000 | evidence_complete |
| supplier_question_period_009 | rag | answered | 1.0000 | evidence_complete |
| supplier_question_period_009 | agent | answered | 1.0000 | evidence_complete |
| competitive_negotiation_010 | rag | answered | 1.0000 | evidence_complete |
| competitive_negotiation_010 | agent | answered | 1.0000 | evidence_complete |
| procurement_definition_011 | rag | answered | 1.0000 | evidence_complete |
| procurement_definition_011 | agent | answered | 1.0000 | evidence_complete |
| procurement_principles_012 | rag | answered | 1.0000 | evidence_complete |
| procurement_principles_012 | agent | answered | 1.0000 | evidence_complete |
| centralized_catalog_013 | rag | answered | 1.0000 | evidence_complete |
| centralized_catalog_013 | agent | answered | 1.0000 | evidence_complete |
| centralized_agency_014 | rag | answered | 1.0000 | evidence_complete |
| centralized_agency_014 | agent | answered | 1.0000 | evidence_complete |
| agency_agreement_015 | rag | answered | 1.0000 | evidence_complete |
| agency_agreement_015 | agent | answered | 1.0000 | evidence_complete |
| alternative_method_approval_016 | rag | answered | 1.0000 | evidence_complete |
| alternative_method_approval_016 | agent | answered | 1.0000 | evidence_complete |
| bid_cancellation_017 | rag | needs_clarification | 1.0000 | evidence_complete |
| bid_cancellation_017 | agent | needs_clarification | 1.0000 | evidence_complete |
| inquiry_one_price_018 | rag | answered | 1.0000 | evidence_complete |
| inquiry_one_price_018 | agent | answered | 1.0000 | evidence_complete |
| procurement_files_retention_019 | rag | answered | 1.0000 | evidence_complete |
| procurement_files_retention_019 | agent | answered | 1.0000 | evidence_complete |
| supplier_inquiry_secret_020 | rag | answered | 1.0000 | evidence_complete |
| supplier_inquiry_secret_020 | agent | answered | 1.0000 | evidence_complete |
| report_illegal_procurement_021 | rag | answered | 1.0000 | evidence_complete |
| report_illegal_procurement_021 | agent | answered | 1.0000 | evidence_complete |
| purchaser_discrimination_liability_022 | rag | answered | 1.0000 | evidence_complete |
| purchaser_discrimination_liability_022 | agent | answered | 1.0000 | evidence_complete |
| supplier_false_material_penalty_023 | rag | refused | 0.0000 | iteration_budget_exhausted |
| supplier_false_material_penalty_023 | agent | needs_clarification | 1.0000 | evidence_complete |
| policy_measures_024 | rag | refused | 0.0000 | iteration_budget_exhausted |
| policy_measures_024 | agent | answered | 1.0000 | evidence_complete |
| government_engineering_law_025 | rag | answered | 1.0000 | evidence_complete |
| government_engineering_law_025 | agent | answered | 1.0000 | evidence_complete |
| procurement_recusal_026 | rag | answered | 1.0000 | evidence_complete |
| procurement_recusal_026 | agent | answered | 1.0000 | evidence_complete |
| public_service_demand_027 | rag | answered | 1.0000 | evidence_complete |
| public_service_demand_027 | agent | answered | 1.0000 | evidence_complete |
| supplier_materials_028 | rag | answered | 1.0000 | evidence_complete |
| supplier_materials_028 | agent | answered | 1.0000 | evidence_complete |
| related_suppliers_029 | rag | needs_clarification | 1.0000 | evidence_complete |
| related_suppliers_029 | agent | needs_clarification | 1.0000 | evidence_complete |
| regional_performance_discrimination_030 | rag | answered | 1.0000 | evidence_complete |
| regional_performance_discrimination_030 | agent | answered | 1.0000 | evidence_complete |
| prequalification_period_031 | rag | answered | 1.0000 | evidence_complete |
| prequalification_period_031 | agent | answered | 1.0000 | evidence_complete |
| consortium_grade_032 | rag | answered | 1.0000 | evidence_complete |
| consortium_grade_032 | agent | answered | 1.0000 | evidence_complete |
| above_threshold_nonopen_033 | rag | needs_clarification | 1.0000 | evidence_complete |
| above_threshold_nonopen_033 | agent | needs_clarification | 1.0000 | evidence_complete |
| batch_centralized_034 | rag | answered | 1.0000 | evidence_complete |
| batch_centralized_034 | agent | answered | 1.0000 | evidence_complete |
| split_to_avoid_open_bid_035 | rag | answered | 1.0000 | evidence_complete |
| split_to_avoid_open_bid_035 | agent | answered | 1.0000 | evidence_complete |
| bid_clarification_deadline_036 | rag | needs_clarification | 1.0000 | evidence_complete |
| bid_clarification_deadline_036 | agent | needs_clarification | 1.0000 | evidence_complete |
| evaluation_methods_037 | rag | answered | 1.0000 | evidence_complete |
| evaluation_methods_037 | agent | answered | 1.0000 | evidence_complete |
| award_timeline_038 | rag | answered | 1.0000 | evidence_complete |
| award_timeline_038 | agent | answered | 1.0000 | evidence_complete |
| winner_refuses_contract_039 | rag | answered | 1.0000 | evidence_complete |
| winner_refuses_contract_039 | agent | answered | 1.0000 | evidence_complete |
| known_harm_date_040 | rag | answered | 1.0000 | evidence_complete |
| known_harm_date_040 | agent | answered | 1.0000 | evidence_complete |
| contract_addition_limit_041 | rag | answered | 1.0000 | evidence_complete |
| contract_addition_limit_041 | agent | answered | 1.0000 | evidence_complete |
| supplier_refuses_contract_042 | rag | refused | 0.0000 | iteration_budget_exhausted |
| supplier_refuses_contract_042 | agent | needs_clarification | 1.0000 | evidence_complete |
| expert_not_independent_043 | rag | needs_clarification | 1.0000 | evidence_complete |
| expert_not_independent_043 | agent | needs_clarification | 1.0000 | evidence_complete |
| same_brand_bidders_044 | rag | answered | 1.0000 | evidence_complete |
| same_brand_bidders_044 | agent | answered | 1.0000 | evidence_complete |
| deemed_collusive_bid_045 | rag | answered | 1.0000 | evidence_complete |
| deemed_collusive_bid_045 | agent | answered | 1.0000 | evidence_complete |
| evaluation_committee_size_046 | rag | answered | 1.0000 | evidence_complete |
| evaluation_committee_size_046 | agent | answered | 1.0000 | evidence_complete |
| award_announcement_period_047 | rag | answered | 1.0000 | evidence_complete |
| award_announcement_period_047 | agent | answered | 1.0000 | evidence_complete |
| small_business_price_deduction_048 | rag | answered | 1.0000 | evidence_complete |
| small_business_price_deduction_048 | agent | answered | 1.0000 | evidence_complete |
| question_and_reply_period_049 | rag | answered | 1.0000 | evidence_complete |
| question_and_reply_period_049 | agent | answered | 1.0000 | evidence_complete |
| engineering_regime_and_scope_050 | rag | refused | 0.0000 | iteration_budget_exhausted |
| engineering_regime_and_scope_050 | agent | answered | 1.0000 | evidence_complete |
