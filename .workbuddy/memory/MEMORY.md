# 全真模拟与专项刷题系统 - 项目记忆

## 项目概述
为 2026 年北京市朝阳区社区工作者考试开发的 Streamlit 全自动备考平台。
**核心特点**：内置 5 大板块考纲逻辑，点击即自动生成题目，无需用户上传任何资料。

## 技术栈
- Streamlit 1.58.0 + Plotly 6.8.0（雷达图/趋势图）
- SQLite3（本地持久化）
- DeepSeek API（兼容 OpenAI 格式）

## 项目结构（重构后）
- app.py：主应用，4 大模块（专项刷题、全真模拟、错题本、学习报告）
- llm_service.py：LLM 服务层，硬编码 5 大板块专家级 System Prompt
- database.py：SQLite 数据库层（wrong_questions, exam_history, daily_stats）
- requirements.txt：依赖（streamlit, requests, pandas, plotly）
- .streamlit/secrets.toml：API 密钥配置

## 5 大板块
1. 公共管理：行政理论、基层自治、接诉即办
2. 法律基础：民法典、居委会组织法、劳动法
3. 社会建设与社会工作：个案/小组/社区工作、社会保障
4. 朝阳区区情：区划地理、CBD、"两区"建设
5. 基本能力：逻辑推理、应急处理、群众沟通

## 关键设计决策
- 专项刷题每个板块 5 题（3单选+1多选+1判断）
- 全真模拟卷 20 题（5 板块各 4 题），含倒计时和雷达图
- 错题重做模式：答对自动标记 mastered
- 学习报告含趋势图和板块分布饼图
