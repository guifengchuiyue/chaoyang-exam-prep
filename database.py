# -*- coding: utf-8 -*-
"""
全真模拟与专项刷题系统 - 数据库模块
====================================
使用 SQLite3 存储错题、考试历史和面试记录。
"""

import sqlite3
import os
import json
from datetime import datetime
from contextlib import contextmanager


class Database:
    """数据库管理类，封装所有 SQLite 操作"""

    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_tables()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        """初始化所有数据表"""
        with self._get_conn() as conn:
            c = conn.cursor()

            # ---- 错题本 ----
            c.execute("""
                CREATE TABLE IF NOT EXISTS wrong_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    options TEXT,
                    correct_answer TEXT NOT NULL,
                    user_answer TEXT,
                    explanation TEXT,
                    knowledge_point TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    is_mastered INTEGER DEFAULT 0
                )
            """)

            # ---- 考试历史 ----
            c.execute("""
                CREATE TABLE IF NOT EXISTS exam_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    exam_type TEXT NOT NULL,
                    total_questions INTEGER NOT NULL,
                    correct_count INTEGER NOT NULL,
                    score REAL NOT NULL,
                    duration_seconds INTEGER DEFAULT 0,
                    category_scores TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # ---- 每日刷题统计 ----
            c.execute("""
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date DATE DEFAULT (date('now')),
                    category TEXT NOT NULL,
                    total INTEGER DEFAULT 0,
                    correct INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            c.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_stats_unique
                ON daily_stats(session_date, category)
            """)

    # ===================== 错题本操作 =====================

    def add_wrong_question(self, category: str, question_type: str,
                           question_text: str, options: str,
                           correct_answer: str, user_answer: str,
                           explanation: str = "", knowledge_point: str = "") -> int:
        """添加错题，返回行ID"""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO wrong_questions
                    (category, question_type, question_text, options,
                     correct_answer, user_answer, explanation, knowledge_point)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (category, question_type, question_text, options,
                  correct_answer, user_answer, explanation, knowledge_point))
            return c.lastrowid

    def get_wrong_questions(self, category: str = None,
                            only_unmastered: bool = True,
                            limit: int = 100) -> list:
        """获取错题列表"""
        with self._get_conn() as conn:
            conditions = []
            params = []
            if only_unmastered:
                conditions.append("is_mastered = 0")
            if category:
                conditions.append("category = ?")
                params.append(category)
            where = " AND ".join(conditions) if conditions else "1=1"
            c = conn.cursor()
            c.execute(f"""
                SELECT * FROM wrong_questions
                WHERE {where}
                ORDER BY created_at DESC LIMIT ?
            """, params + [limit])
            return [dict(r) for r in c.fetchall()]

    def mark_mastered(self, question_id: int):
        """标记错题已掌握"""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE wrong_questions
                SET is_mastered = 1, retry_count = retry_count + 1
                WHERE id = ?
            """, (question_id,))

    def increment_retry(self, question_id: int):
        """增加重做次数"""
        with self._get_conn() as conn:
            conn.execute("""
                UPDATE wrong_questions SET retry_count = retry_count + 1
                WHERE id = ?
            """, (question_id,))

    def get_wrong_stats(self) -> dict:
        """错题统计"""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM wrong_questions WHERE is_mastered = 0")
            total = c.fetchone()[0]
            c.execute("SELECT COUNT(*) FROM wrong_questions WHERE is_mastered = 1")
            mastered = c.fetchone()[0]
            c.execute("""
                SELECT category, COUNT(*) FROM wrong_questions
                WHERE is_mastered = 0 GROUP BY category
            """)
            by_cat = {r[0]: r[1] for r in c.fetchall()}
            return {"total": total, "mastered": mastered, "by_category": by_cat}

    # ===================== 考试历史 =====================

    def save_exam_result(self, exam_type: str, total: int, correct: int,
                         score: float, duration: int = 0,
                         category_scores: dict = None) -> int:
        """保存考试成绩"""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO exam_history
                    (exam_type, total_questions, correct_count, score,
                     duration_seconds, category_scores)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (exam_type, total, correct, score, duration,
                  json.dumps(category_scores or {}, ensure_ascii=False)))
            return c.lastrowid

    def get_exam_history(self, limit: int = 20) -> list:
        """获取考试历史"""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT * FROM exam_history
                ORDER BY created_at DESC LIMIT ?
            """, (limit,))
            return [dict(r) for r in c.fetchall()]

    # ===================== 每日统计 =====================

    def update_daily_stats(self, category: str, total: int, correct: int):
        """更新每日刷题统计"""
        with self._get_conn() as conn:
            today = datetime.now().strftime("%Y-%m-%d")
            c = conn.cursor()
            c.execute("""
                INSERT INTO daily_stats (session_date, category, total, correct)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_date, category) DO UPDATE SET
                    total = total + excluded.total,
                    correct = correct + excluded.correct
            """, (today, category, total, correct))

    def get_daily_stats(self, days: int = 7) -> list:
        """获取最近N天统计"""
        with self._get_conn() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT session_date,
                       SUM(total) as t, SUM(correct) as c
                FROM daily_stats
                WHERE session_date >= date('now', ?)
                GROUP BY session_date ORDER BY session_date
            """, (f"-{days} days",))
            return [dict(r) for r in c.fetchall()]
