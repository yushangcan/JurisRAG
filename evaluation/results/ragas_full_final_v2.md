# JurisRAG 最终版本 RAGAS 评测（部分完成）

- RAGAS版本：`0.4.3`
- 评测模型：`deepseek-chat`
- 数据集：`legal_rag_gold_v2`
- 总题数：50
- 答案生成成功：50/50
- 五项指标全部成功：42/50
- **状态：未完成。评测接口余额不足（HTTP 402），缺失指标未计入均值。**
- 下表每个均值只使用该指标成功返回的题目，失败题不是0分。

## RAGAS 汇总

| 指标 | 平均分 | 成功题数 | 失败题数 | 中文含义 |
| --- | ---: | ---: | ---: | --- |
| context_precision | 0.9112 | 43 | 7 | 召回上下文中与问题和参考答案相关的比例 |
| context_recall | 0.9545 | 44 | 6 | 参考答案所需信息被召回上下文覆盖的程度 |
| faithfulness | 0.9756 | 43 | 7 | 生成答案中的陈述能否由检索证据支持 |
| answer_relevancy | 0.9330 | 43 | 7 | 生成答案是否直接回答用户问题 |
| factual_correctness | 0.2638 | 42 | 8 | 生成答案与人工参考答案的事实一致程度 |

## 确定性检索指标

| 指标 | 结果 |
| --- | ---: |
| Parent Recall@1 | 0.6200 |
| Parent Recall@3 | 0.8500 |
| Parent Recall@5 | 0.9500 |
| MRR@5 | 0.7530 |
| RAGAS ID Context Precision@5 | 0.1980 |
| RAGAS ID Context Recall@5 | 0.9500 |

## 逐题结果

| case_id | CP | CR | 忠实度 | 回答相关性 | 事实正确性 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| supplier_qualification_001 | 1.0000 | 1.0000 | 1.0000 | 0.9953 | 0.0000 | 成功 |
| procurement_methods_002 | 0.7000 | 1.0000 | 0.9565 | 0.9778 | 0.1700 | 成功 |
| mandatory_tender_projects_003 | 1.0000 | 1.0000 | 1.0000 | 0.8984 | 0.1500 | 成功 |
| bid_collusion_004 | 1.0000 | 1.0000 | 1.0000 | 0.9833 | 0.1800 | 成功 |
| demand_investigation_005 | 1.0000 | 1.0000 | 1.0000 | 0.8934 | 0.0000 | 成功 |
| sme_reserved_share_006 | 1.0000 | 1.0000 | 0.8000 | 0.8307 | 0.4000 | 成功 |
| infrastructure_scope_007 | 1.0000 | 1.0000 | 0.9000 | 0.8438 | 0.0000 | 成功 |
| framework_purchase_scope_008 | 1.0000 | 0.0000 | 1.0000 | 0.8819 | 0.0000 | 成功 |
| supplier_question_period_009 | 1.0000 | 1.0000 | 1.0000 | 0.8052 | 0.2500 | 成功 |
| competitive_negotiation_010 | 0.5000 | 1.0000 | 1.0000 | 0.9583 | 0.0000 | 成功 |
| procurement_definition_011 | 1.0000 | 1.0000 | 1.0000 | 0.9804 | 0.2000 | 成功 |
| procurement_principles_012 | 1.0000 | 1.0000 | 1.0000 | 0.9316 | 1.0000 | 成功 |
| centralized_catalog_013 | 1.0000 | 1.0000 | 0.8182 | 0.9855 | 0.0000 | 成功 |
| centralized_agency_014 | 0.7000 | 1.0000 | 1.0000 | 0.9866 | 0.0000 | 成功 |
| agency_agreement_015 | 1.0000 | 1.0000 | 1.0000 | 0.9891 | 0.4300 | 成功 |
| alternative_method_approval_016 | 0.6792 | 1.0000 | 1.0000 | 0.9148 | 0.2200 | 成功 |
| bid_cancellation_017 | 0.7500 | 1.0000 | 1.0000 | 0.7289 | 0.4000 | 成功 |
| inquiry_one_price_018 | 0.8667 | 1.0000 | 1.0000 | 0.9822 | 0.5000 | 成功 |
| procurement_files_retention_019 | 1.0000 | 1.0000 | 1.0000 | 0.8979 | 0.0000 | 成功 |
| supplier_inquiry_secret_020 | 1.0000 | 1.0000 | 1.0000 | 0.9926 | 0.0000 | 成功 |
| report_illegal_procurement_021 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 成功 |
| purchaser_discrimination_liability_022 | 0.5000 | 1.0000 | 1.0000 | 0.9884 | 0.3600 | 成功 |
| supplier_false_material_penalty_023 | 0.7556 | 1.0000 | 1.0000 | 0.9927 | 0.0000 | 成功 |
| policy_measures_024 | 1.0000 | 1.0000 | 1.0000 | 0.9753 | 0.4200 | 成功 |
| government_engineering_law_025 | 1.0000 | 0.0000 | 1.0000 | 0.9218 | 0.4000 | 成功 |
| procurement_recusal_026 | 0.8333 | 1.0000 | 0.8182 | 0.9759 | 0.6000 | 成功 |
| public_service_demand_027 | 0.8667 | 1.0000 | 1.0000 | 0.9426 | 0.4000 | 成功 |
| supplier_materials_028 | 1.0000 | 1.0000 | 1.0000 | 0.9106 | 0.0000 | 成功 |
| related_suppliers_029 | 1.0000 | 1.0000 | 0.8571 | 0.9052 | 0.4400 | 成功 |
| regional_performance_discrimination_030 | 0.8333 | 1.0000 | 1.0000 | 0.9517 | 0.2900 | 成功 |
| prequalification_period_031 | 0.8056 | 1.0000 | 1.0000 | 0.9126 | 0.0000 | 成功 |
| consortium_grade_032 | 1.0000 | 1.0000 | 1.0000 | 0.9187 | 0.6700 | 成功 |
| above_threshold_nonopen_033 | 1.0000 | 1.0000 | 0.8000 | 0.9376 | 0.4000 | 成功 |
| batch_centralized_034 | 1.0000 | 1.0000 | 1.0000 | 0.8187 | 0.8600 | 成功 |
| split_to_avoid_open_bid_035 | 1.0000 | 1.0000 | 1.0000 | 0.9375 | 0.0000 | 成功 |
| bid_clarification_deadline_036 | 1.0000 | 1.0000 | 1.0000 | 0.8933 | 0.0000 | 成功 |
| evaluation_methods_037 | 0.9167 | 1.0000 | 1.0000 | 0.9745 | 1.0000 | 成功 |
| award_timeline_038 | 1.0000 | 1.0000 | 1.0000 | 0.9404 | 0.0000 | 成功 |
| winner_refuses_contract_039 | 1.0000 | 1.0000 | 1.0000 | 0.9819 | 0.4000 | 成功 |
| known_harm_date_040 | 0.8333 | 1.0000 | 1.0000 | 0.8686 | 0.0000 | 成功 |
| contract_addition_limit_041 | 0.7556 | 1.0000 | 1.0000 | 0.9890 | 0.2900 | 成功 |
| supplier_refuses_contract_042 | 0.8875 | 1.0000 | 1.0000 | 0.9405 | 0.1500 | 成功 |
| expert_not_independent_043 | 1.0000 | 1.0000 | 1.0000 | 0.9838 | - | 失败: factual_correctness |
| same_brand_bidders_044 | - | 1.0000 | - | - | - | 失败: context_precision,faithfulness,answer_relevancy,factual_correctness |
| deemed_collusive_bid_045 | - | - | - | - | - | 失败: context_precision,context_recall,faithfulness,answer_relevancy,factual_correctness |
| evaluation_committee_size_046 | - | - | - | - | - | 失败: context_precision,context_recall,faithfulness,answer_relevancy,factual_correctness |
| award_announcement_period_047 | - | - | - | - | - | 失败: context_precision,context_recall,faithfulness,answer_relevancy,factual_correctness |
| small_business_price_deduction_048 | - | - | - | - | - | 失败: context_precision,context_recall,faithfulness,answer_relevancy,factual_correctness |
| question_and_reply_period_049 | - | - | - | - | - | 失败: context_precision,context_recall,faithfulness,answer_relevancy,factual_correctness |
| engineering_regime_and_scope_050 | - | - | - | - | - | 失败: context_precision,context_recall,faithfulness,answer_relevancy,factual_correctness |

## 解释边界

- Recall@K、MRR和ID指标基于人工gold ID，是确定性检索指标。
- 五项RAGAS语义指标由LLM裁判给出，不是司法机关或法律专家认证。
- Faithfulness高只表示答案有上下文依据，不自动代表依据是现行有效法规。
- Factual Correctness依赖当前人工参考答案的完整性。
- 本数据集仍是开发集，不能将结果外推为生产环境法律准确率。
