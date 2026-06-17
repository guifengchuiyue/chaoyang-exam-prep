# -*- coding: utf-8 -*-
"""
全真模拟与专项刷题系统 - 主应用
================================
北京市朝阳区社区工作者全真模拟与专项刷题系统
基于 Streamlit，内置 5 大板块考纲逻辑，点击即自动生成题目。
"""

import streamlit as st
import json
import time
import random
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import defaultdict

from database import Database
from llm_service import (
    generate_category_questions,
    generate_full_exam,
    check_answer,
    format_options,
    SYSTEM_PROMPTS,
)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="朝阳社工全真模拟刷题系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 移动端检测：注入 viewport meta 标签
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
""", unsafe_allow_html=True)

# ============================================================
# 初始化
# ============================================================

DB_PATH = "data/exam_system.db"
db = Database(DB_PATH)

CATEGORIES = ["公共管理", "法律基础", "社会建设与社会工作", "朝阳区区情", "基本能力"]
CATEGORY_ICONS = {
    "公共管理": "🏛️",
    "法律基础": "⚖️",
    "社会建设与社会工作": "🤝",
    "朝阳区区情": "🏙️",
    "基本能力": "🧠",
}
CATEGORY_COLORS = {
    "公共管理": "#5470C6",
    "法律基础": "#91CC75",
    "社会建设与社会工作": "#FAC858",
    "朝阳区区情": "#EE6666",
    "基本能力": "#73C0DE",
}

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    /* === 基础样式 === */
    .main-header { font-size: clamp(1.3em, 4vw, 2em); font-weight: bold; color: #C41E3A; margin-bottom: 0; }
    .sub-header { color: #666; font-size: 0.95em; margin-top: 0; }
    .question-card {
        background: white; border-radius: 10px; padding: 14px 16px;
        margin: 8px 0; border: 1px solid #E8E8E8;
        border-left: 4px solid #1976D2;
    }
    .question-card.correct { border-left-color: #2E7D32; background: #F1F8E9; }
    .question-card.wrong { border-left-color: #C62828; background: #FFF3F3; }
    .score-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-weight: bold; font-size: 0.85em;
    }
    .score-high { background: #E8F5E9; color: #2E7D32; }
    .score-mid { background: #FFF8E1; color: #F57F00; }
    .score-low { background: #FFEBEE; color: #C62828; }
    .stat-box { text-align: center; padding: 12px; border-radius: 10px; background: #F5F5F5; }
    .stat-box .big { font-size: clamp(1.4em, 5vw, 2.2em); font-weight: bold; color: #C41E3A; }
    .stat-box .label { font-size: 0.8em; color: #888; margin-top: 4px; }
    .timer-warning { color: #F57F00; font-weight: bold; }
    .timer-danger { color: #C62828; font-weight: bold; animation: blink 1s infinite; }
    @keyframes blink { 50% { opacity: 0.5; } }
    .footer { text-align: center; color: #AAA; font-size: 0.8em; margin-top: 30px; padding: 15px; }

    /* === 移动端适配 === */
    @media (max-width: 768px) {
        /* 侧边栏在手机上自动折叠 */
        section[data-testid="stSidebar"] { display: none; }

        /* 按钮更大、更容易点击 */
        button[kind="primary"], .stButton > button {
            min-height: 44px !important;
            font-size: 16px !important;
            padding: 10px 20px !important;
        }

        /* radio 和 checkbox 增大点击区域 */
        .stRadio label, .stCheckbox label {
            min-height: 40px;
            line-height: 40px;
            font-size: 15px !important;
        }

        /* 题目卡片内边距缩小 */
        .question-card { padding: 10px 12px; margin: 6px 0; }

        /* 表格/图表可横向滚动 */
        .js-plotly-plot { max-width: 100% !important; overflow-x: auto; }

        /* 标题字号缩小 */
        h1 { font-size: 1.5em !important; }
        h2 { font-size: 1.3em !important; }
        h3 { font-size: 1.1em !important; }
    }

    /* === 超大屏幕 === */
    @media (min-width: 1200px) {
        .question-card { padding: 18px 24px; }
    }

    /* === 触屏友好：增大所有交互元素间距 === */
    @media (hover: none) and (pointer: coarse) {
        button, .stRadio div[role="radiogroup"] label, .stCheckbox label {
            padding: 12px 8px !important;
            margin: 4px 0 !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 工具函数
# ============================================================

def get_config_from_secrets():
    """从 Streamlit secrets 获取 API 配置"""
    try:
        return {
            "api_base_url": st.secrets.get("LLM_API_BASE_URL", "https://api.deepseek.com/v1"),
            "api_key": st.secrets.get("LLM_API_KEY", ""),
            "model_name": st.secrets.get("LLM_MODEL_NAME", "deepseek-chat"),
        }
    except Exception:
        return {"api_base_url": "https://api.deepseek.com/v1", "api_key": "", "model_name": "deepseek-chat"}


def check_api_ready(api_key: str) -> bool:
    if not api_key or api_key == "your-api-key-here":
        st.warning("⚠️ 请先在侧边栏填入 API Key")
        return False
    return True


def render_question_input(q: dict, idx: int, prefix: str = ""):
    """渲染单道题的输入控件，返回用户答案（不重复显示选项文本，选项只在控件中显示）"""
    key = f"{prefix}_{q['type']}_{idx}"
    q_type = q.get("type", "单选")
    options = q.get("options", {})

    if q_type == "单选":
        choices = [f"{k}. {v}" for k, v in options.items()]
        ans = st.radio(
            "请选择一个选项：",
            choices,
            key=key,
            index=None,
        )
        return ans[0] if ans else ""
    elif q_type == "多选":
        st.caption("请勾选所有正确的选项（可多选）：")
        selected = []
        cols = st.columns(len(options))
        for i, (k, v) in enumerate(options.items()):
            with cols[i]:
                if st.checkbox(f"{k}. {v}", key=f"{key}_{k}"):
                    selected.append(k)
        return "".join(sorted(selected)) if selected else ""
    elif q_type == "判断":
        ans = st.radio(
            "请判断对错：",
            ["对（正确）", "错（错误）"],
            key=key,
            index=None,
        )
        return "对" if ans and "对" in ans else ("错" if ans else "")
    return ""


def render_result_card(q: dict, idx: int, user_ans: str, is_correct: bool):
    """渲染答题结果卡片"""
    card_class = "correct" if is_correct else "wrong"
    st.markdown(f'<div class="question-card {card_class}">', unsafe_allow_html=True)

    status = "✅ 正确" if is_correct else "❌ 错误"
    st.markdown(f"**{status}** | {CATEGORY_ICONS.get(q.get('category',''),'')} {q.get('category','')} | {q.get('knowledge_point','')}")

    st.markdown(q["question"])
    if q.get("options"):
        for k, v in q["options"].items():
            st.markdown(f"- {k}. {v}")

    if not is_correct:
        st.info(f"**正确答案**：{q['answer']}")
        if user_ans:
            st.error(f"**你的答案**：{user_ans}")

    with st.expander("📖 查看解析"):
        st.markdown(q.get("explanation", "暂无解析"))

    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 侧边栏
# ============================================================

with st.sidebar:
    st.markdown("## 🎯 朝阳社工")
    st.markdown("### 全真模拟刷题系统")
    st.caption("北京市朝阳区社区工作者考试专用")

    st.divider()

    # API 配置
    with st.expander("⚙️ API 配置", expanded=False):
        cfg = get_config_from_secrets()
        api_base_url = st.text_input("Base URL", value=cfg["api_base_url"])
        api_key = st.text_input("API Key", value=cfg["api_key"], type="password", placeholder="sk-...")
        model_name = st.text_input("模型", value=cfg["model_name"])
        if st.button("💾 保存（本次会话）", use_container_width=True):
            st.success("已保存")

    st.divider()

    # 导航
    st.subheader("📋 功能导航")
    page = st.radio("", ["📝 专项刷题", "📋 全真模拟", "📓 错题本", "📊 学习报告"],
                    label_visibility="collapsed")

    st.divider()

    # 统计概览
    wrong_stats = db.get_wrong_stats()
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("待攻克", wrong_stats["total"])
    with col_b:
        st.metric("已掌握", wrong_stats["mastered"])

    # 历史考试
    exams = db.get_exam_history(5)
    if exams:
        st.divider()
        st.caption("📜 最近考试")
        for ex in exams[:3]:
            score_badge = "score-high" if ex["score"] >= 70 else ("score-mid" if ex["score"] >= 50 else "score-low")
            st.markdown(
                f"<span class='score-badge {score_badge}'>{ex['score']:.0f}分</span> "
                f"{ex['exam_type']} · {ex['created_at'][:10]}",
                unsafe_allow_html=True
            )

    st.divider()
    st.caption(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# 移动端底部导航（仅在手机宽度时通过 JS 显示）
st.markdown("""
<div id="mobile-nav" style="display:none; position:fixed; bottom:0; left:0; right:0;
    background:white; border-top:2px solid #E0E0E0; z-index:9999;
    padding:8px 0; box-shadow:0 -2px 8px rgba(0,0,0,0.08);">
    <div style="display:flex; justify-content:space-around;">
        <a href="?nav=practice" style="text-decoration:none; color:#333; text-align:center; flex:1;">
            <div style="font-size:1.2em;">📝</div>
            <div style="font-size:0.7em;">专项刷题</div>
        </a>
        <a href="?nav=exam" style="text-decoration:none; color:#333; text-align:center; flex:1;">
            <div style="font-size:1.2em;">📋</div>
            <div style="font-size:0.7em;">全真模拟</div>
        </a>
        <a href="?nav=wrong" style="text-decoration:none; color:#333; text-align:center; flex:1;">
            <div style="font-size:1.2em;">📓</div>
            <div style="font-size:0.7em;">错题本</div>
        </a>
        <a href="?nav=report" style="text-decoration:none; color:#333; text-align:center; flex:1;">
            <div style="font-size:1.2em;">📊</div>
            <div style="font-size:0.7em;">学习报告</div>
        </a>
    </div>
</div>
<script>
(function(){
    var nav = document.getElementById('mobile-nav');
    if(window.innerWidth <= 768) { nav.style.display = 'block'; }
    window.addEventListener('resize', function(){
        nav.style.display = (window.innerWidth <= 768) ? 'block' : 'none';
    });
})();
</script>
""", unsafe_allow_html=True)


# ============================================================
# 模块一：专项刷题
# ============================================================

def module_practice():
    st.markdown('<p class="main-header">📝 五大专项训练</p>', unsafe_allow_html=True)
    st.caption("点击任意板块，AI 自动生成 5 道紧扣考纲的专属题目。无需上传任何资料。")

    # 5 个板块 Tab
    tabs = st.tabs([f"{CATEGORY_ICONS[c]} {c}" for c in CATEGORIES])

    for i, category in enumerate(CATEGORIES):
        with tabs[i]:
            st.markdown(f"### {CATEGORY_ICONS[category]} {category}专项训练")
            st.caption(SYSTEM_PROMPTS[category].split("## 考查范围")[1].split("## 出题要求")[0].strip()
                       if "## 考查范围" in SYSTEM_PROMPTS[category] else "")

            # Session state per category
            q_key = f"q_{category}"
            ans_key = f"ans_{category}"
            sub_key = f"sub_{category}"

            if q_key not in st.session_state:
                st.session_state[q_key] = None
            if ans_key not in st.session_state:
                st.session_state[ans_key] = {}
            if sub_key not in st.session_state:
                st.session_state[sub_key] = False

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button(f"🚀 生成{category}题目", key=f"gen_{category}",
                             type="primary", use_container_width=True):
                    if not check_api_ready(api_key):
                        st.stop()
                    with st.spinner(f"🤔 AI 正在生成{category}专项题目..."):
                        try:
                            data = generate_category_questions(
                                api_base_url, api_key, model_name, category
                            )
                            st.session_state[q_key] = data
                            st.session_state[ans_key] = {}
                            st.session_state[sub_key] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"生成失败：{e}")

            # 显示题目
            questions_data = st.session_state[q_key]
            if questions_data:
                questions = questions_data.get("questions", [])
                st.markdown(f"---\n#### 📝 共 {len(questions)} 道题，请作答：")

                user_answers = {}
                for j, q in enumerate(questions):
                    q_type = q.get("type", "单选")
                    type_label = {"单选": "单选题", "多选": "多选题", "判断": "判断题"}.get(q_type, q_type)
                    st.markdown(f"**第 {j+1} 题** [{type_label}] {q['question']}")
                    # 选项在 render_question_input 控件中展示，不重复显示
                    ans = render_question_input(q, j, f"cat_{category}")
                    user_answers[str(j)] = ans
                    st.markdown("---")

                col_s1, col_s2 = st.columns([1, 3])
                with col_s1:
                    if st.button("📤 提交答案", key=f"submit_{category}",
                                 type="primary", use_container_width=True):
                        st.session_state[ans_key] = user_answers
                        st.session_state[sub_key] = True

                        # 统计并更新数据库
                        correct = 0
                        for j, q in enumerate(questions):
                            ua = user_answers.get(str(j), "")
                            if check_answer(ua, q["answer"], q["type"]):
                                correct += 1
                            else:
                                db.add_wrong_question(
                                    category=category,
                                    question_type=q["type"],
                                    question_text=q["question"],
                                    options=json.dumps(q.get("options", {}), ensure_ascii=False),
                                    correct_answer=q["answer"],
                                    user_answer=ua,
                                    explanation=q.get("explanation", ""),
                                    knowledge_point=q.get("knowledge_point", ""),
                                )
                        db.update_daily_stats(category, len(questions), correct)
                        st.rerun()

            # 显示结果
            if st.session_state[sub_key] and st.session_state[ans_key]:
                st.markdown("---")
                st.markdown("### 📊 答题结果")
                correct = sum(
                    1 for j, q in enumerate(questions)
                    if check_answer(st.session_state[ans_key].get(str(j), ""),
                                    q["answer"], q["type"])
                )
                total = len(questions)
                col_r1, col_r2, col_r3 = st.columns(3)
                with col_r1:
                    st.metric("正确", correct)
                with col_r2:
                    st.metric("错误", total - correct)
                with col_r3:
                    st.metric("正确率", f"{correct/total*100:.0f}%")
                st.progress(correct / total)

                st.markdown("---")
                for j, q in enumerate(questions):
                    ua = st.session_state[ans_key].get(str(j), "")
                    is_c = check_answer(ua, q["answer"], q["type"])
                    render_result_card(q, j, ua, is_c)

                if st.button("🔄 重新生成", key=f"regen_{category}", use_container_width=True):
                    st.session_state[q_key] = None
                    st.session_state[sub_key] = False
                    st.rerun()


# ============================================================
# 模块二：全真模拟卷
# ============================================================

def module_full_exam():
    st.markdown('<p class="main-header">📋 全真模拟考试</p>', unsafe_allow_html=True)
    st.caption("一键生成 20 题综合模拟卷，涵盖 5 大板块，倒计时交卷，智能评分。")

    # Session state
    if "exam_data" not in st.session_state:
        st.session_state.exam_data = None
    if "exam_started" not in st.session_state:
        st.session_state.exam_started = False
    if "exam_answers" not in st.session_state:
        st.session_state.exam_answers = {}
    if "exam_submitted" not in st.session_state:
        st.session_state.exam_submitted = False
    if "exam_start_time" not in st.session_state:
        st.session_state.exam_start_time = None
    if "exam_results" not in st.session_state:
        st.session_state.exam_results = None

    # ---- 未开始：显示生成按钮 ----
    if not st.session_state.exam_started:
        st.markdown("""
        <div style="background:#FFF3E0; border-radius:12px; padding:24px; margin:20px 0; text-align:center;">
            <h3>📋 全真模拟考试说明</h3>
            <p style="color:#666;">
            系统将从 <b>公共管理、法律基础、社会建设与社会工作、朝阳区区情、基本能力</b><br>
            5 大板块中各抽取题目，拼装成一套 <b>20 题综合模拟试卷</b>。
            </p>
            <p style="color:#888; font-size:0.9em;">
            ⏱ 建议用时：40 分钟 | 📝 题型：单选 + 多选 + 判断 | 🔍 交卷后逐题解析
            </p>
        </div>
        """, unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            if st.button("🚀 一键生成全真模拟卷", type="primary", use_container_width=True):
                if not check_api_ready(api_key):
                    st.stop()
                with st.spinner("🤔 AI 正在从 5 大板块生成全真模拟卷（预计 30-60 秒）..."):
                    try:
                        data = generate_full_exam(api_base_url, api_key, model_name)
                        st.session_state.exam_data = data
                        st.session_state.exam_started = True
                        st.session_state.exam_answers = {}
                        st.session_state.exam_submitted = False
                        st.session_state.exam_start_time = time.time()
                        st.session_state.exam_results = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"生成失败：{e}")

    # ---- 考试进行中 ----
    else:
        exam_data = st.session_state.exam_data
        if exam_data is None:
            st.error("试卷数据异常，请刷新重试")
            return

        questions = exam_data.get("questions", [])
        time_limit = exam_data.get("time_limit_minutes", 40)

        # 倒计时
        elapsed = time.time() - (st.session_state.exam_start_time or time.time())
        remaining = max(0, time_limit * 60 - elapsed)
        mins = int(remaining // 60)
        secs = int(remaining % 60)

        timer_class = ""
        if remaining < 300:
            timer_class = "timer-danger"
        elif remaining < 600:
            timer_class = "timer-warning"

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center;
                    background:#F5F5F5; border-radius:10px; padding:12px 20px; margin-bottom:16px;">
            <span style="font-weight:bold; font-size:1.1em;">📋 {exam_data.get('exam_title','全真模拟卷')}</span>
            <span class="{timer_class}" style="font-size:1.3em;">⏱ {mins:02d}:{secs:02d}</span>
            <span>📝 {len(questions)} 题</span>
        </div>
        """, unsafe_allow_html=True)

        # 题目
        user_answers = {}
        for j, q in enumerate(questions):
            with st.container():
                q_type = q.get("type", "单选")
                type_label = {"单选": "单选题", "多选": "多选题", "判断": "判断题"}.get(q_type, q_type)
                st.markdown(f"**第 {j+1} 题** [{CATEGORY_ICONS.get(q.get('category',''),'')} {q.get('category','')}] "
                            f"({type_label})")
                st.markdown(q["question"])
                # 选项在 render_question_input 控件中展示，不重复显示
                ans = render_question_input(q, j, "exam")
                user_answers[str(j)] = ans
                st.markdown("---")

        col_s1, col_s2, col_s3 = st.columns([1, 1, 2])
        with col_s1:
            if st.button("📤 交卷", type="primary", use_container_width=True):
                st.session_state.exam_answers = user_answers
                st.session_state.exam_submitted = True
                elapsed_sec = int(time.time() - (st.session_state.exam_start_time or time.time()))

                # 批改
                correct_total = 0
                category_stats = defaultdict(lambda: {"total": 0, "correct": 0})
                results = []

                for j, q in enumerate(questions):
                    ua = user_answers.get(str(j), "")
                    is_c = check_answer(ua, q["answer"], q["type"])
                    cat = q.get("category", "未知")
                    category_stats[cat]["total"] += 1
                    if is_c:
                        correct_total += 1
                        category_stats[cat]["correct"] += 1
                    else:
                        db.add_wrong_question(
                            category=cat,
                            question_type=q["type"],
                            question_text=q["question"],
                            options=json.dumps(q.get("options", {}), ensure_ascii=False),
                            correct_answer=q["answer"],
                            user_answer=ua,
                            explanation=q.get("explanation", ""),
                            knowledge_point=q.get("knowledge_point", ""),
                        )
                    results.append({"q": q, "ua": ua, "correct": is_c})

                total = len(questions)
                score = correct_total / total * 100

                # 保存考试历史
                cat_scores = {c: (s["correct"] / s["total"] * 100) if s["total"] else 0
                              for c, s in category_stats.items()}
                db.save_exam_result("全真模拟", total, correct_total, score, elapsed_sec, cat_scores)
                db.update_daily_stats("全真模拟", total, correct_total)

                st.session_state.exam_results = {
                    "results": results, "correct": correct_total, "total": total,
                    "score": score, "category_stats": dict(category_stats),
                    "elapsed": elapsed_sec,
                }
                st.rerun()

        with col_s2:
            if st.button("🔄 重新生成", use_container_width=True):
                for k in ["exam_data", "exam_started", "exam_answers",
                          "exam_submitted", "exam_start_time", "exam_results"]:
                    if k in st.session_state:
                        st.session_state[k] = None if k != "exam_answers" else {}
                st.rerun()

    # ---- 交卷后显示结果 ----
    if st.session_state.exam_submitted and st.session_state.exam_results:
        er = st.session_state.exam_results
        results = er["results"]

        st.markdown("---")
        st.markdown("## 📊 考试成绩报告")

        # 总分卡片
        score = er["score"]
        score_class = "score-high" if score >= 70 else ("score-mid" if score >= 50 else "score-low")
        elapsed_str = f"{er['elapsed']//60}分{er['elapsed']%60}秒"

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.markdown(f'<div class="stat-box"><div class="big">{score:.0f}</div><div class="label">总分（百分制）</div></div>', unsafe_allow_html=True)
        with col_s2:
            st.markdown(f'<div class="stat-box"><div class="big">{er["correct"]}/{er["total"]}</div><div class="label">正确/总题数</div></div>', unsafe_allow_html=True)
        with col_s3:
            st.markdown(f'<div class="stat-box"><div class="big">{elapsed_str}</div><div class="label">用时</div></div>', unsafe_allow_html=True)
        with col_s4:
            grade = "优秀" if score >= 85 else ("良好" if score >= 70 else ("一般" if score >= 50 else "需努力"))
            st.markdown(f'<div class="stat-box"><div class="big">{grade}</div><div class="label">评级</div></div>', unsafe_allow_html=True)

        st.progress(score / 100, text=f"得分率 {score:.1f}%")

        # 各板块雷达图
        st.markdown("### 🎯 各板块得分分析")
        cs = er["category_stats"]
        radar_categories = list(cs.keys())
        radar_scores = [cs[c]["correct"] / cs[c]["total"] * 100 if cs[c]["total"] else 0 for c in radar_categories]

        col_radar, col_bar = st.columns([1, 1])
        with col_radar:
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=radar_scores + [radar_scores[0]],
                theta=radar_categories + [radar_categories[0]],
                fill='toself',
                fillcolor='rgba(196, 30, 58, 0.2)',
                line=dict(color='#C41E3A', width=2),
                name='得分率(%)'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(range=[0, 100], ticksuffix='%')),
                height=400, margin=dict(l=40, r=40, t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_bar:
            bar_data = pd.DataFrame({
                "板块": radar_categories,
                "得分率": radar_scores,
                "颜色": [CATEGORY_COLORS.get(c, "#888") for c in radar_categories],
            })
            fig_bar = px.bar(bar_data, x="板块", y="得分率", color="板块",
                             color_discrete_map={c: CATEGORY_COLORS.get(c, "#888") for c in radar_categories},
                             text="得分率")
            fig_bar.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
            fig_bar.update_layout(yaxis_range=[0, 110], height=400, showlegend=False,
                                  margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        # 各板块详情
        st.markdown("### 📋 各板块详情")
        for cat in radar_categories:
            cat_data = cs[cat]
            cat_score = cat_data["correct"] / cat_data["total"] * 100 if cat_data["total"] else 0
            st.markdown(
                f"{CATEGORY_ICONS.get(cat,'')} **{cat}**：{cat_data['correct']}/{cat_data['total']} "
                f"({cat_score:.0f}%)"
            )
            st.progress(cat_score / 100)

        # 逐题解析
        st.markdown("---")
        st.markdown("### 🔍 逐题解析")
        for j, r in enumerate(results):
            render_result_card(r["q"], j, r["ua"], r["correct"])

        if st.button("🔄 再来一套", use_container_width=True, type="primary"):
            for k in ["exam_data", "exam_started", "exam_answers",
                      "exam_submitted", "exam_start_time", "exam_results"]:
                if k in st.session_state:
                    st.session_state[k] = None if k != "exam_answers" else {}
            st.rerun()


# ============================================================
# 模块三：错题本
# ============================================================

def module_wrong_book():
    st.markdown('<p class="main-header">📓 错题本</p>', unsafe_allow_html=True)
    st.caption("错题重做，答对自动移除。攻克弱项，精准提分。")

    stats = db.get_wrong_stats()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📌 待攻克", stats["total"])
    with col2:
        st.metric("✅ 已掌握", stats["mastered"])
    with col3:
        total_all = stats["total"] + stats["mastered"]
        rate = stats["mastered"] / total_all * 100 if total_all else 0
        st.metric("🎯 掌握率", f"{rate:.1f}%")

    if stats["by_category"]:
        st.markdown("**按板块分布：**")
        cols = st.columns(len(stats["by_category"]))
        for i, (cat, cnt) in enumerate(stats["by_category"].items()):
            with cols[i]:
                st.markdown(f"{CATEGORY_ICONS.get(cat,'')} {cat}: **{cnt}**")

    st.divider()

    filter_cat = st.selectbox("按板块筛选", ["全部"] + CATEGORIES, key="wrong_filter")
    cat_param = filter_cat if filter_cat != "全部" else None
    wrong_list = db.get_wrong_questions(category=cat_param, only_unmastered=True)

    if not wrong_list:
        st.success("🎉 太棒了！当前没有待攻克的错题。")
        return

    st.markdown(f"共 **{len(wrong_list)}** 道待攻克错题")

    # 错题重做
    st.divider()
    st.subheader("🔄 错题重做")

    if "retry_answers" not in st.session_state:
        st.session_state.retry_answers = {}
    if "retry_done" not in st.session_state:
        st.session_state.retry_done = False
    if "retry_results" not in st.session_state:
        st.session_state.retry_results = {}

    with st.form("retry_form"):
        retry_answers = {}
        for i, wq in enumerate(wrong_list):
            # 规范化题型：数据库可能存英文（兼容旧数据），统一转为中文
            raw_q_type = str(wq["question_type"]).strip()
            q_type_map = {
                "single": "单选", "single_choice": "单选",
                "multi": "多选", "multiple": "多选", "multiple_choice": "多选",
                "tf": "判断", "true_false": "判断",
                "单选": "单选", "多选": "多选", "判断": "判断",
            }
            q_type = q_type_map.get(raw_q_type, "单选")
            type_label = {"单选": "单选题", "多选": "多选题", "判断": "判断题"}[q_type]
            st.markdown(f"**第 {i+1} 题** [{CATEGORY_ICONS.get(wq['category'],'')} {wq['category']}] "
                        f"({type_label}) {wq['question_text']}")
            st.caption(f"上次你的答案：{wq['user_answer']} | 重做次数：{wq['retry_count']}")

            options = json.loads(wq["options"]) if wq["options"] else {}
            # 如果 options 为空，补全默认选项
            if not options:
                if q_type == "单选":
                    options = {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"}
                elif q_type == "多选":
                    options = {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D", "E": "选项E"}

            if q_type == "单选":
                choices = [f"{k}. {v}" for k, v in options.items()]
                ans = st.radio("请选择一个选项：", choices, key=f"retry_r_{wq['id']}", index=None)
                retry_answers[str(wq["id"])] = ans[0] if ans else ""
            elif q_type == "多选":
                st.caption("请勾选所有正确的选项（可多选）：")
                selected = []
                cols = st.columns(len(options))
                for j, (k, v) in enumerate(options.items()):
                    with cols[j]:
                        if st.checkbox(f"{k}. {v}", key=f"retry_cb_{wq['id']}_{k}"):
                            selected.append(k)
                retry_answers[str(wq["id"])] = "".join(sorted(selected))
            elif q_type == "判断":
                ans = st.radio("请判断对错：", ["对（正确）", "错（错误）"], key=f"retry_tf_{wq['id']}", index=None)
                retry_answers[str(wq["id"])] = "对" if ans and "对" in ans else ("错" if ans else "")
            st.markdown("---")

        if st.form_submit_button("📤 提交重做答案", type="primary", use_container_width=True):
            st.session_state.retry_answers = retry_answers
            st.session_state.retry_done = True

            results = {}
            correct_count = 0
            for wq in wrong_list:
                wid = str(wq["id"])
                ua = retry_answers.get(wid, "")
                is_c = check_answer(ua, wq["correct_answer"], wq["question_type"])
                results[wid] = is_c
                if is_c:
                    db.mark_mastered(wq["id"])
                    correct_count += 1
                else:
                    db.increment_retry(wq["id"])
            st.session_state.retry_results = results
            db.update_daily_stats("错题重做", len(wrong_list), correct_count)
            st.rerun()

    if st.session_state.retry_done and st.session_state.retry_results:
        rr = st.session_state.retry_results
        correct_retry = sum(1 for v in rr.values() if v)
        st.markdown(f"### 重做结果：✅ {correct_retry} / {len(rr)}")
        if correct_retry > 0:
            st.success(f"🎉 {correct_retry} 道错题已移出错题本！")
        if st.button("🔄 继续重做剩余错题", use_container_width=True):
            st.session_state.retry_done = False
            st.session_state.retry_results = {}
            st.rerun()


# ============================================================
# 模块四：学习报告
# ============================================================

def module_report():
    st.markdown('<p class="main-header">📊 学习报告</p>', unsafe_allow_html=True)
    st.caption("追踪学习轨迹，洞察薄弱环节。")

    # 历史考试趋势
    exams = db.get_exam_history(20)
    if exams:
        st.subheader("📈 考试成绩趋势")
        df_exam = pd.DataFrame(exams)
        df_exam["date"] = pd.to_datetime(df_exam["created_at"]).dt.strftime("%m/%d %H:%M")
        fig_line = px.line(df_exam, x="date", y="score", markers=True,
                           title="历次考试得分趋势",
                           labels={"score": "得分", "date": "日期"})
        fig_line.add_hline(y=60, line_dash="dash", line_color="orange",
                           annotation_text="及格线")
        fig_line.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_line, use_container_width=True)

        # 统计
        avg_score = df_exam["score"].mean()
        best_score = df_exam["score"].max()
        total_exams = len(df_exam)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总考试次数", total_exams)
        with col2:
            st.metric("平均分", f"{avg_score:.1f}")
        with col3:
            st.metric("最高分", f"{best_score:.0f}")

    # 每日刷题统计
    daily = db.get_daily_stats(14)
    if daily:
        st.subheader("📅 近两周刷题统计")
        df_daily = pd.DataFrame(daily)
        df_daily["正确率"] = (df_daily["c"] / df_daily["t"] * 100).round(1)
        fig_bar2 = px.bar(df_daily, x="session_date", y="t", color="正确率",
                          labels={"session_date": "日期", "t": "刷题数"},
                          color_continuous_scale="RdYlGn")
        fig_bar2.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bar2, use_container_width=True)

    # 错题分布
    wrong_stats = db.get_wrong_stats()
    if wrong_stats["by_category"]:
        st.subheader("🎯 错题板块分布")
        df_wrong = pd.DataFrame({
            "板块": list(wrong_stats["by_category"].keys()),
            "错题数": list(wrong_stats["by_category"].values()),
        })
        fig_pie = px.pie(df_wrong, names="板块", values="错题数",
                         color="板块",
                         color_discrete_map={c: CATEGORY_COLORS.get(c, "#888") for c in df_wrong["板块"]})
        fig_pie.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    if not exams and not daily:
        st.info("还没有学习记录，快去刷题吧！")


# ============================================================
# 主路由
# ============================================================

def main():
    if page == "📝 专项刷题":
        module_practice()
    elif page == "📋 全真模拟":
        module_full_exam()
    elif page == "📓 错题本":
        module_wrong_book()
    elif page == "📊 学习报告":
        module_report()

    st.divider()
    st.markdown("""
    <div class="footer">
        <p>🎯 北京市朝阳区社区工作者全真模拟与专项刷题系统</p>
        <p>祝考试顺利，成功上岸！💪</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
