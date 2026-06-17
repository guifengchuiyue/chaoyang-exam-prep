# -*- coding: utf-8 -*-
"""
全真模拟与专项刷题系统 - LLM 服务层
====================================
硬编码 5 大板块专家级 Prompt，自动调用大模型 API 生成高仿真题目。
"""

import json
import re
import requests
from typing import Optional, Dict, Any, List


# ============================================================
# 通用 API 调用
# ============================================================

def call_llm(api_base_url: str, api_key: str, model_name: str,
             system_prompt: str, user_prompt: str,
             temperature: float = 0.5, max_tokens: int = 4096) -> Optional[str]:
    """调用大模型 API（兼容 OpenAI / DeepSeek 格式）"""
    url = api_base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url = f"{url}/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        raise Exception("API 请求超时，请检查网络或稍后重试。")
    except requests.exceptions.ConnectionError:
        raise Exception("无法连接 API 服务器，请检查 Base URL。")
    except requests.exceptions.HTTPError as e:
        detail = ""
        if e.response is not None:
            try:
                detail = e.response.json().get("error", {}).get("message", "")
            except Exception:
                detail = e.response.text[:200]
        raise Exception(f"API 错误 (HTTP {e.response.status_code if e.response else '?'}): {detail}")
    except Exception as e:
        raise Exception(f"API 调用异常: {str(e)}")


# ============================================================
# JSON 解析工具
# ============================================================

def _extract_json(text: str) -> Optional[Dict]:
    """从大模型返回文本中提取 JSON"""
    # 优先匹配 ```json ... ```
    m = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if m:
        text = m.group(1).strip()
    else:
        s = text.find('{')
        e = text.rfind('}')
        if s != -1 and e != -1:
            text = text[s:e + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ============================================================
# 题目生成公共 Prompt 模板
# ============================================================

QUESTION_FORMAT_INSTRUCTION = """
## 输出格式（必须是合法 JSON，不要输出任何其他内容）

```json
{
  "questions": [
    {
      "id": 1,
      "type": "单选",
      "category": "公共管理",
      "question": "题目正文（必须使用中文）",
      "options": {"A": "选项A内容", "B": "选项B内容", "C": "选项C内容", "D": "选项D内容"},
      "answer": "A",
      "explanation": "详细解析，指出考点和答题思路",
      "knowledge_point": "具体考点名称"
    }
  ]
}
```

**极其重要的题型与选项规范（必须严格遵守）：**
- type 字段必须使用中文：单选、多选、判断
- 单选题（type="单选"）：必须有且仅有 4 个选项（A/B/C/D），缺一不可！answer 为单个大写字母如 "A"
- 多选题（type="多选"）：必须有且仅有 5 个选项（A/B/C/D/E），缺一不可！answer 为大写字母连写如 "ABD"，正确答案至少 2 个
- 判断题（type="判断"）：answer 必须为"对"或"错"（字符串，不是布尔值 true/false）
- 每道题的 options 字段绝对不能为空、绝对不能缺失！
- 每道题必须有 knowledge_point 字段标明考点
- 所有文本内容必须使用中文，不允许出现英文单词

**极其重要的去重规则（必须严格遵守）：**
- 每次生成的 5 道题目必须覆盖不同的考点（knowledge_point 不得重复）
- 题目之间不能有相似的题干表述，每道题必须是全新的、独立的知识点考查
- 避免出现笼统模糊的题目，每道题要有明确的具体考查内容
- 如果发现自己可能生成重复题目，立即换一个角度或考点重新出题
"""


# ============================================================
# 5 大板块专家级 System Prompt（硬编码）
# ============================================================

SYSTEM_PROMPTS = {
    "公共管理": """你是一位资深公务员考试命题专家，长期参与北京市社区工作者招聘考试命题工作。
你的任务是根据以下考点范围，自动生成高质量的社区工作者考试题目。

## 考查范围
1. **行政管理基础理论**：公共管理基本概念、政府职能转变、服务型政府建设
2. **基层治理体系**：街道办事处与社区居委会的关系、网格化管理、"街乡吹哨、部门报到"机制
3. **基层群众自治**：《城市居民委员会组织法》核心条款、居委会选举与议事规则
4. **接诉即办机制**：北京市"接诉即办"工作条例、12345热线响应流程、考核评价体系
5. **社区服务与管理**：社区公共服务供给、社会组织培育、社区工作者职责定位

## 出题要求
- 生成 5 道题目，包含 3 道单选题、1 道多选题、1 道判断题
- 题目要有区分度，约40%基础记忆型、40%理解应用型、20%综合分析型
- 选项设计要有迷惑性，干扰项应来源于常见误解
- 解析必须指出考点和答题思路
""",

    "法律基础": """你是一位资深法律考试命题专家，精通社区治理相关法律法规。
你的任务是根据以下考点范围，自动生成高质量的社区工作者法律基础考题。

## 考查范围
1. **《民法典》社区相关**：相邻关系（通风采光噪音）、建筑物区分所有权、物业服务合同、业主大会与业委会、继承编核心条款
2. **《城市居民委员会组织法》**：居委会性质与任务、选举程序、居民会议制度
3. **《劳动法》与《劳动合同法》**：劳动关系认定、劳动合同解除、劳动争议处理
4. **北京市地方性法规**：《北京市物业管理条例》、《北京市生活垃圾管理条例》、《北京市文明行为促进条例》
5. **社区高频法律实务**：人民调解、法律援助、信访条例

## 出题要求
- 生成 5 道题目，包含 3 道单选题、1 道多选题、1 道判断题
- 题目以案例情境题为主（至少3道），考查法条在社区场景中的实际运用
- 解析必须引用具体法条名称和条款
""",

    "社会建设与社会工作": """你是一位资深社会工作教育专家，长期从事社区工作者职业资格考试培训。
你的任务是根据以下考点范围，自动生成高质量的社会工作知识考题。

## 考查范围
1. **社会工作专业方法**：个案工作（接案-预估-计划-介入-评估-结案）、小组工作（类型与阶段）、社区工作（地区发展/社会策划/社会行动模式）
2. **社会工作价值观与伦理**：助人自助、保密原则、案主自决、非评判态度
3. **社会保障体系**：社会救助（低保）、社会保险（养老/医疗）、社会福利、优抚安置
4. **社区治理理念**：共建共治共享、五社联动、协商民主、居民参与
5. **基层矛盾化解**：沟通技巧、调解方法、危机干预、情绪疏导

## 出题要求
- 生成 5 道题目，包含 3 道单选题、1 道多选题、1 道判断题
- 侧重实务应用，考查社工专业方法在具体情境中的选择
- 解析必须说明为何正确选项符合社工专业理念
""",

    "朝阳区区情": """你是一位北京市朝阳区区情研究专家，非常熟悉朝阳区的发展历史、现状和规划。
请基于你的知识储备，自动生成以下内容的考试题目。

## 考查范围（基于真实公开信息）
1. **基本区情**：朝阳区地理位置（北京东部，面积470.8平方公里）、行政区划（24个街道、19个乡/地区办事处）、人口规模（常住人口约345万）
2. **经济发展**：CBD（北京商务中心区）定位与发展、中关村朝阳园（电子城）产业布局、"两区"建设（国家服务业扩大开放综合示范区+自贸试验区）
3. **社会事业**：朝阳区教育（如北京中学等名校）、医疗资源、文化设施（798艺术区、中国电影博物馆等）
4. **基层治理特色**："朝阳群众"品牌、社区治理创新实践、党建引领基层治理
5. **发展规划**：朝阳区"十四五"规划重点、"五宜"朝阳建设（宜居、宜业、宜商、宜学、宜游）

## 出题要求
- 生成 5 道题目，包含 3 道单选题、1 道多选题、1 道判断题
- 以记忆型和理解型为主，考查考生对朝阳区基本情况的熟悉程度
- 解析应补充相关背景知识，帮助考生扩展了解
""",

    "基本能力": """你是一位资深行政职业能力测验命题专家，专门为社区工作者考试设计基本能力考题。
你的任务是根据以下考点范围，自动生成高质量的考题。

## 考查范围
1. **逻辑推理**：演绎推理、归纳推理、类比推理、逻辑判断
2. **言语理解与表达**：阅读理解、语句排序、逻辑填空
3. **数量关系与资料分析**：基础数学运算、图表数据解读
4. **应急处理能力**：突发事件研判、优先级排序、应急方案制定
5. **群众工作能力**：沟通协调、舆情应对、群众动员

## 出题要求
- 生成 5 道题目，包含 3 道单选题、1 道多选题、1 道判断题
- 题目需贴近社区工作实际场景，考查在具体情境中的问题解决能力
- 题干可设计简短案例，选项需体现不同的处理思路
""",
}


# ============================================================
# 专项题目生成
# ============================================================

def generate_category_questions(
    api_base_url: str, api_key: str, model_name: str, category: str
) -> Optional[Dict[str, Any]]:
    """
    根据板块类别自动生成 5 道题目

    参数:
        category: 五大板块之一（公共管理/法律基础/社会建设与社会工作/朝阳区区情/基本能力）

    返回:
        {"questions": [...]} 或 None
    """
    system_prompt = SYSTEM_PROMPTS.get(category)
    if not system_prompt:
        raise ValueError(f"未知板块: {category}")

    full_system = system_prompt + "\n" + QUESTION_FORMAT_INSTRUCTION

    user_prompt = f"请根据以上考点范围，为北京市朝阳区社区工作者考试生成 5 道{category}专项题目。\n\n"
    user_prompt += "题目题型分配（必须严格遵守）：\n"
    user_prompt += "第1题 = 单选\n第2题 = 单选\n第3题 = 单选\n第4题 = 多选\n第5题 = 判断\n\n"
    user_prompt += "每道题的 type 字段必须严格按照以上分配填写，不得自行改变题型。只输出 JSON。"

    response = call_llm(api_base_url, api_key, model_name, full_system, user_prompt,
                        temperature=0.7, max_tokens=4096)
    if response is None:
        return None

    data = _extract_json(response)
    if data is None:
        raise Exception(f"解析{category}题目 JSON 失败，原始响应: {response[:300]}...")
    if "questions" not in data:
        raise Exception(f"{category}题目缺少 questions 字段")
    # 数据清洗：规范化 type 字段和补全缺失的 options
    data["questions"] = _normalize_questions(data["questions"])
    return data


# ============================================================
# 全真模拟卷生成
# ============================================================

FULL_EXAM_SYSTEM_PROMPT = """你是一位资深社区工作者考试命题组组长，负责编制北京市朝阳区社区工作者招聘考试的全真模拟试卷。

你需要从以下 5 大知识板块中各抽取题目，拼装成一套 20 题的综合模拟试卷：

## 题目分配比例
- 公共管理：4 题（2道单选+1道多选+1道判断）
- 法律基础：4 题（2道单选+1道多选+1道判断）
- 社会建设与社会工作：4 题（2道单选+1道多选+1道判断）
- 朝阳区区情：4 题（2道单选+1道多选+1道判断）
- 基本能力：4 题（2道单选+1道多选+1道判断）

## 各板块考点范围

### 公共管理
行政管理基础理论、政府职能、基层群众自治、接诉即办响应机制、网格化管理、"街乡吹哨部门报到"

### 法律基础
《民法典》邻里/物业/继承、《城市居民委员会组织法》、《劳动法》、北京市物业管理条例等

### 社会建设与社会工作
个案/小组/社区工作方法、社会保障体系、社区治理理念、基层矛盾化解

### 朝阳区区情
朝阳区地理位置（470.8km²）、行政区划、CBD定位、中关村朝阳园、"两区"建设、"五宜"朝阳

### 基本能力
逻辑推理、言语理解、应急处理、群众沟通、资料分析

## 输出格式（严格遵守，缺一不可！）
```json
{
  "exam_title": "2026年北京市朝阳区社区工作者招聘考试·全真模拟卷",
  "total_questions": 20,
  "time_limit_minutes": 40,
  "questions": [
    {
      "id": 1,
      "type": "单选",
      "category": "公共管理",
      "question": "题目正文（必须中文）",
      "options": {"A": "选项A内容", "B": "选项B内容", "C": "选项C内容", "D": "选项D内容"},
      "answer": "A",
      "explanation": "详细解析",
      "knowledge_point": "具体考点"
    }
  ]
}
```

**极其重要的规范（每道题必须遵守，否则试卷无效）：**
- type 字段必须使用中文：单选、多选、判断
- 单选题（type="单选"）：options 必须有且仅有 4 个选项（A/B/C/D），缺一不可！answer 为单个大写字母
- 多选题（type="多选"）：options 必须有且仅有 5 个选项（A/B/C/D/E），缺一不可！answer 为大写字母连写如"ABD"
- 判断题（type="判断"）：options 可为空或省略，answer 必须为"对"或"错"
- 任何题目的 options 字段绝对不能为空对象{}（判断题除外）
- 每道题必须有 knowledge_point 字段
- 所有文本必须使用中文

## 质量要求
- 题目难度有梯度，基础题约40%、中等题约40%、难题约20%
- 只输出 JSON，不要输出任何其他内容
"""


def generate_full_exam(
    api_base_url: str, api_key: str, model_name: str
) -> Optional[Dict[str, Any]]:
    """一键生成 20 题全真模拟卷"""
    # 精确指定每道题的题型分配
    type_schedule = []
    for cat in ["公共管理", "法律基础", "社会建设与社会工作", "朝阳区区情", "基本能力"]:
        type_schedule.extend(["单选", "单选", "多选", "判断"])

    schedule_desc = "\n".join(f"第{i+1}题 = {t}（{cat}）" for i, (t, cat) in enumerate(
        zip(type_schedule, [c for c in ["公共管理", "法律基础", "社会建设与社会工作", "朝阳区区情", "基本能力"] for _ in range(4)])))

    user_prompt = "请生成一套完整的20题全真模拟试卷。\n\n"
    user_prompt += "每道题的题型必须严格按照以下分配（type字段必须与之一致）：\n"
    user_prompt += schedule_desc
    user_prompt += "\n\n每道题的 type 字段必须严格按照以上分配填写，不得自行改变题型。只输出 JSON。"

    response = call_llm(api_base_url, api_key, model_name,
                        FULL_EXAM_SYSTEM_PROMPT, user_prompt,
                        temperature=0.4, max_tokens=8192)
    if response is None:
        return None

    data = _extract_json(response)
    if data is None:
        raise Exception(f"解析模拟卷 JSON 失败，原始响应: {response[:300]}...")
    if "questions" not in data:
        raise Exception("模拟卷缺少 questions 字段")
    # 数据清洗
    data["questions"] = _normalize_questions(data["questions"])
    return data


# ============================================================
# 数据清洗：规范化 AI 返回的题目数据
# ============================================================

# type 字段英文→中文映射
TYPE_MAP = {
    "single": "单选", "single_choice": "单选", "radio": "单选",
    "multi": "多选", "multiple": "多选", "multiple_choice": "多选", "checkbox": "多选",
    "tf": "判断", "true_false": "判断", "judge": "判断", "bool": "判断",
}

# 默认选项模板
DEFAULT_SINGLE_OPTIONS = {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D"}
DEFAULT_MULTI_OPTIONS = {"A": "选项A", "B": "选项B", "C": "选项C", "D": "选项D", "E": "选项E"}


def _normalize_questions(questions: list) -> list:
    """规范化题目数据：强制按位置修正题型、统一 type 为中文、补全缺失的 options"""
    # 5 题模式：前3单选 + 第4多选 + 第5判断
    # 20 题模式：每4题一组（2单选+1多选+1判断）
    if len(questions) == 5:
        expected_types = ["单选", "单选", "单选", "多选", "判断"]
    elif len(questions) == 20:
        expected_types = []
        for _ in range(5):
            expected_types.extend(["单选", "单选", "多选", "判断"])
    else:
        expected_types = None

    for i, q in enumerate(questions):
        # 1. 无条件强制按位置修正题型（覆盖 AI 返回的任何 type 值）
        if expected_types and i < len(expected_types):
            q["type"] = expected_types[i]
        else:
            raw_type = str(q.get("type", "")).strip().lower()
            q["type"] = TYPE_MAP.get(raw_type, "单选")

        # 2. 补全/修正 options
        options = q.get("options")
        q_type = q["type"]
        if not options or not isinstance(options, dict) or len(options) == 0:
            if q_type == "多选":
                q["options"] = dict(DEFAULT_MULTI_OPTIONS)
            elif q_type == "单选":
                q["options"] = dict(DEFAULT_SINGLE_OPTIONS)
        # 如果多选只有4个选项，补到5个
        elif q_type == "多选" and len(options) < 5:
            keys = list("ABCDE")
            for k in keys:
                if k not in options:
                    options[k] = f"选项{k}"
            q["options"] = options

        # 3. 确保 answer 是字符串
        answer = q.get("answer", "")
        if isinstance(answer, bool):
            q["answer"] = "对" if answer else "错"
        elif isinstance(answer, list):
            q["answer"] = "".join(str(a).strip().upper() for a in answer)
        else:
            q["answer"] = str(answer).strip()

    return questions


# ============================================================
# 答案校验
# ============================================================

def check_answer(user_answer, correct_answer: str, q_type: str) -> bool:
    """检查答案是否正确（兼容中英文 type，兼容判断题布尔值/字符串）"""
    # 统一 type 为中文
    raw_type = str(q_type).strip().lower()
    normalized_type = TYPE_MAP.get(raw_type, q_type)

    if normalized_type == "判断":
        user_str = str(user_answer).strip()
        correct_str = str(correct_answer).strip()

        # 用户答案标准化：对/错
        user_is_true = user_str in ["对", "正确", "true", "True", "True", "是", "√"]
        # 正确答案标准化：对/错（兼容布尔值和字符串）
        if correct_str.lower() in ["true", "对", "正确", "是"]:
            correct_is_true = True
        elif correct_str.lower() in ["false", "错", "错误", "否"]:
            correct_is_true = False
        else:
            # 尝试解析为布尔值
            try:
                correct_is_true = bool(int(correct_str)) if correct_str.isdigit() else False
            except:
                correct_is_true = False

        return user_is_true == correct_is_true

    # 单选/多选：集合比较（忽略大小写和顺序）
    user_set = set(c.strip().upper() for c in str(user_answer) if c.strip().isalpha())
    correct_set = set(c.strip().upper() for c in str(correct_answer) if c.strip().isalpha())
    return user_set == correct_set and len(user_set) > 0


def format_options(options: dict) -> str:
    """格式化选项为显示文本"""
    return "\n".join(f"{k}. {v}" for k, v in options.items())
