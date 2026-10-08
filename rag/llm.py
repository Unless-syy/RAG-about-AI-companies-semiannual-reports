# -*- coding: utf-8 -*-
"""DeepSeek 问答。密钥只从环境变量 DEEPSEEK_API_KEY 读取，不下发浏览器。"""
import json

import requests

from . import config

SYSTEM_PROMPT = """你是一名严谨的A股上市公司财报分析助手。你只能依据用户提供的「资料片段」回答问题。

必须遵守：
1. 每一个论断后面紧跟引用标记，格式为 [公司·报告期·章节·页码]，直接复制资料片段标题里给出的引用串，例如 [寒武纪·2026年半年度·第三节 管理层讨论与分析·p12]。
2. 引用只能来自资料片段中真实出现的标注，不得编造公司、章节或页码。
3. 资料片段里没有的信息，回答"资料中未提及"，绝不猜测或使用你自己的先验知识补充。
4. 涉及数字时给出准确数值与单位，并注明同比/环比变化（资料里有的话）。
5. 跨公司比较题：先逐公司列出对应数据与引用，再给出结论。
6. 使用简体中文，分点作答，简洁准确。"""

USER_TEMPLATE = """以下是从各公司半年度报告中检索到的资料片段，每段前面是它的引用串：

{context}

---

问题：{question}

请依据上述资料作答，每个论断后附引用标记。"""


def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        blocks.append(f"【片段{i}】{h['citation']}\n{h['text']}")
    return "\n\n".join(blocks)


def _payload(question, hits, stream):
    return {
        "model": config.DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                context=build_context(hits), question=question)},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
        "stream": stream,
    }


def _headers():
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError("环境变量 DEEPSEEK_API_KEY 未设置")
    return {"Authorization": f"Bearer {config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"}


def answer(question, hits, timeout=180):
    """非流式，返回完整答案文本。"""
    r = requests.post(f"{config.DEEPSEEK_BASE}/chat/completions",
                      headers=_headers(),
                      json=_payload(question, hits, False),
                      timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def answer_stream(question, hits, timeout=180):
    """流式，逐段 yield 文本增量。"""
    with requests.post(f"{config.DEEPSEEK_BASE}/chat/completions",
                       headers=_headers(),
                       json=_payload(question, hits, True),
                       stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = obj.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if piece:
                yield piece
