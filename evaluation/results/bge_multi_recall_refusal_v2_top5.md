# 法规 RAG 检索评测报告

- 运行时间：2026-08-20T17:10:04+08:00
- Git commit：`1a6274e99c3caf661eeebe21ea538b2ba6f2b872`
- Git 工作树：有未提交实验改动
- 数据集：`legal_rag_gold_v2`，50 条人工 gold
- 语料：66 份唯一文档
- 检索配置：parent_child_bm25_exact_vector_rrf_optional_reranker，Top-K=5
- Embedding：`bge_small_zh_v1_5` / `BAAI/bge-small-zh-v1.5`
- Reranker：`bge_reranker_base_onnx`，候选数=20
- HyDE：未启用
- RAGAS：0.4.3，仅运行 ID-based 非 LLM 指标

## 汇总

| 指标 | 分数 | 满分题数 |
|---|---:|---:|
| parent_recall_at_1 | 0.6200 | 31/50 |
| parent_recall_at_3 | 0.8500 | 42/50 |
| parent_recall_at_5 | 0.9500 | 47/50 |
| reciprocal_rank_at_5 | 0.7530 | 31/50 |
| document_hit_at_5 | 0.9800 | 49/50 |
| article_hit_at_5 | 0.9600 | 48/50 |
| ragas_id_context_precision_at_5 | 0.1980 | 0/50 |
| ragas_id_context_recall_at_5 | 0.9500 | 47/50 |

检索延迟：mean=8429.10 ms，P50=8065.20 ms，P95=12833.62 ms。
冷启动预热：18600.32 ms（不计入上述分位数）。
有答案题放行率：1.0000，错误拒答 0/50。
二次多路召回触发率：0.0000。

## 逐题结果

| case_id | gold 法条 | Top-1 | Recall@5 | MRR@5 |
|---|---|---|---:|---:|
| supplier_qualification_001 | 第二十二条 | 《中华人民共和国政府采购法》第二十二条 | 1.0 | 1.000 |
| procurement_methods_002 | 第二十六条 | 《中华人民共和国政府采购法》第二十六条 | 1.0 | 1.000 |
| mandatory_tender_projects_003 | 第三条 | 《中华人民共和国招标投标法》第三条 | 1.0 | 1.000 |
| bid_collusion_004 | 第三十二条 | 《中华人民共和国招标投标法实施条例》第六十七条 | 1.0 | 0.250 |
| demand_investigation_005 | 第十一条 | 《政府采购需求管理办法》第十一条 | 1.0 | 1.000 |
| sme_reserved_share_006 | 第八条 | 《政府采购促进中小企业发展管理办法》第八条 | 1.0 | 1.000 |
| infrastructure_scope_007 | 第二条 | 《必须招标的基础设施和公用事业项目范围规定》第二条 | 1.0 | 1.000 |
| framework_purchase_scope_008 | 第三条 | 《政府采购框架协议采购方式管理暂行办法》第三条 | 1.0 | 1.000 |
| supplier_question_period_009 | 第十条 | 《中华人民共和国政府采购法》第五十二条 | 1.0 | 0.500 |
| competitive_negotiation_010 | 第二十七条 | 《政府采购竞争性磋商采购方式管理暂行办法》第三条 | 1.0 | 0.500 |
| procurement_definition_011 | 第二条 | 《中华人民共和国政府采购法》第二条 | 1.0 | 1.000 |
| procurement_principles_012 | 第三条 | 《中华人民共和国政府采购法》第三条 | 1.0 | 1.000 |
| centralized_catalog_013 | 第七条 | 《中华人民共和国政府采购法》第十八条 | 0.0 | 0.000 |
| centralized_agency_014 | 第十八条 | 《中华人民共和国政府采购法》第十八条 | 1.0 | 1.000 |
| agency_agreement_015 | 第二十条 | 《政府采购代理机构管理暂行办法》第十三条 | 1.0 | 0.500 |
| alternative_method_approval_016 | 第二十七条 | 《政府采购需求管理办法》第十九条 | 1.0 | 0.200 |
| bid_cancellation_017 | 第三十六条 | 《中华人民共和国政府采购法》第三十六条 | 1.0 | 1.000 |
| inquiry_one_price_018 | 第四十条 | 《政府采购非招标采购方式管理办法》第四十七条 | 1.0 | 0.200 |
| procurement_files_retention_019 | 第四十二条 | 《中华人民共和国政府采购法》第四十二条 | 1.0 | 1.000 |
| supplier_inquiry_secret_020 | 第五十一条 | 《中华人民共和国政府采购法》第五十一条 | 1.0 | 1.000 |
| report_illegal_procurement_021 | 第七十条 | 《中华人民共和国政府采购法》第七十条 | 1.0 | 1.000 |
| purchaser_discrimination_liability_022 | 第七十一条 | 《中华人民共和国政府采购法实施条例》第二十条 | 1.0 | 0.250 |
| supplier_false_material_penalty_023 | 第七十七条 | 《中华人民共和国政府采购法》第七十七条 | 1.0 | 1.000 |
| policy_measures_024 | 第六条 | 《政府采购促进中小企业发展管理办法》第八条 | 1.0 | 0.500 |
| government_engineering_law_025 | 第七条 | 《中华人民共和国政府采购法》第四条 | 0.0 | 0.000 |
| procurement_recusal_026 | 第九条 | 《中华人民共和国政府采购法实施条例》第九条 | 1.0 | 1.000 |
| public_service_demand_027 | 第十五条 | 《财政部关于进一步加强政府采购需求和履约验收管理的指导意见》 | 1.0 | 0.500 |
| supplier_materials_028 | 第十七条 | 《中华人民共和国政府采购法实施条例》第十七条 | 1.0 | 1.000 |
| related_suppliers_029 | 第十八条 | 《中华人民共和国政府采购法实施条例》第十八条 | 1.0 | 1.000 |
| regional_performance_discrimination_030 | 第二十条 | 《中华人民共和国政府采购法实施条例》第二十条 | 1.0 | 1.000 |
| prequalification_period_031 | 第二十一条 | 《中华人民共和国政府采购法实施条例》第二十一条 | 1.0 | 1.000 |
| consortium_grade_032 | 第二十二条 | 《中华人民共和国政府采购法实施条例》第二十二条 | 1.0 | 1.000 |
| above_threshold_nonopen_033 | 第二十三条 | 《政府采购需求管理办法》第十九条 | 1.0 | 0.500 |
| batch_centralized_034 | 第二十四条 | 《中华人民共和国政府采购法实施条例》第二十四条 | 1.0 | 1.000 |
| split_to_avoid_open_bid_035 | 第二十八条 | 《中华人民共和国政府采购法实施条例》第二十八条 | 1.0 | 1.000 |
| bid_clarification_deadline_036 | 第三十一条 | 《政府采购货物和服务招标投标管理办法》第二十七条 | 1.0 | 0.500 |
| evaluation_methods_037 | 第三十四条 | 《政府采购货物和服务招标投标管理办法》第五十四条 | 1.0 | 0.500 |
| award_timeline_038 | 第四十三条 | 《政府采购非招标采购方式管理办法》第四十九条 | 1.0 | 0.333 |
| winner_refuses_contract_039 | 第四十九条 | 《中华人民共和国政府采购法实施条例》第四十九条 | 1.0 | 1.000 |
| known_harm_date_040 | 第五十三条 | 《中华人民共和国政府采购法实施条例》第五十三条 | 1.0 | 1.000 |
| contract_addition_limit_041 | 第六十七条 | 《中华人民共和国政府采购法》第四十九条 | 1.0 | 0.333 |
| supplier_refuses_contract_042 | 第七十二条 | 《中华人民共和国政府采购法实施条例》第七十二条 | 1.0 | 1.000 |
| expert_not_independent_043 | 第七十五条 | 《政府采购评审专家管理办法》第二十七条 | 1.0 | 0.500 |
| same_brand_bidders_044 | 第三十一条 | 《政府采购货物和服务招标投标管理办法》第三十一条 | 1.0 | 1.000 |
| deemed_collusive_bid_045 | 第三十七条 | 《政府采购货物和服务招标投标管理办法》第三十七条 | 1.0 | 1.000 |
| evaluation_committee_size_046 | 第四十七条 | 《政府采购货物和服务招标投标管理办法》第四十七条 | 1.0 | 1.000 |
| award_announcement_period_047 | 第六十九条 | 《政府采购货物和服务招标投标管理办法》第六十九条 | 1.0 | 1.000 |
| small_business_price_deduction_048 | 第九条 | 《政府采购促进中小企业发展管理办法》第九条 | 1.0 | 1.000 |
| question_and_reply_period_049 | 第五十二条、第十三条 | 《政府采购质疑和投诉办法》第十四条 | 0.5 | 0.250 |
| engineering_regime_and_scope_050 | 第七条、第三条 | 《中华人民共和国政府采购法》第四条 | 1.0 | 0.333 |

## 评测边界

- v2 共50题，gold 的法条定位和原文摘录已由 `evaluation.audit_gold_dataset` 逐条核验；它仍是开发集，不等于生产准确率。
- 本报告只评检索，没有运行答案生成，也没有运行 Faithfulness、Factual Correctness 等 LLM 裁判指标。
- RAGAS ID Context Precision 衡量 Top-5 中 gold 父法条占比；它不是语义相关性的 LLM 判断。
- “满分题数”表示该题此项指标等于 1.0；单个 gold 法条配合 5 个不同检索结果时，ID Precision 命中一次通常只有 0.2。
- 低分案例应回查检索结果和源法条，不应修改 gold 迎合模型预测。
