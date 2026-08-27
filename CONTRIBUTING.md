# 参与 JurisRAG 开发

本项目把“可复现”和“法律证据边界”放在功能数量之前。每次修改应从 `main` 的可运行提交创建短期分支，保持一个提交只完成一类可解释变更。

## 本地开发

1. 使用 Python 3.12 或项目已验证的兼容版本创建独立虚拟环境。
2. 复制 `.env.example` 为 `.env`，真实 Key 只保存在本机。
3. 安装依赖：`python -m pip install -r requirements.txt`。
4. 若没有本地模型，可先设置 `EMBEDDING_BACKEND=hash`、`RERANKER_BACKEND=none`，再运行 `python -m offline_ingestion.build_index`。
5. 运行 `python -m pytest -q` 和 `python -m pip check`。

## 分支与提交

- 分支示例：`feat/exact-search`、`fix/citation-contract`、`docs/evaluation-boundary`。
- 提交使用 `feat:`、`fix:`、`test:`、`docs:`、`ci:`、`build:`、`chore:` 前缀。
- 不重写他人历史，不提交无关格式化，不把多个无关功能塞进一个 commit。

## 法律问答改动的额外要求

- 任何生成答案都必须能映射到真实证据 ID；证据不足时失败关闭。
- 法规名、条号、页码、来源 URL 与内容哈希不能在清洗和切块中丢失。
- 合法性评价、法律后果和原文查询要分开测试，不能为了降低延迟绕过高风险验证。
- 评测必须记录数据集、配置、模型、代码 commit 和失败样本；开发集指标不能称为线上准确率。

## 禁止提交

`.env`、API Key、虚拟环境、模型权重、`data/index/`、`tmp/`、日志、包含隐私的提问或未脱敏响应均不得进入 Git。提交前运行 `git status` 与 `git diff --cached` 人工复核。
