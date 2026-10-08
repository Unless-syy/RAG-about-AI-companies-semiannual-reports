# -*- coding: utf-8 -*-
"""全局配置。密钥只从环境变量读取，绝不落盘、绝不下发浏览器。"""
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = DATA_DIR / "index"
PAGES_JSONL = DATA_DIR / "pages.jsonl"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
TABLE_CHECK = DATA_DIR / "table_check.md"      # 表格抽样核对结论
EVAL_TXT = ROOT / "eval_qa.txt"                # 评测输出（独立 txt）

# ---------------- 模型 / 密钥（全部走环境变量） ----------------
EMBED_MODEL_DIR = os.environ.get("EMBED_MODEL_DIR", "")
# Qwen3-Embedding 官方要求：query 加 instruct 前缀，document 不加
QUERY_INSTRUCTION = (
    "给定一个关于A股上市公司半年度报告的中文问题，"
    "检索能够回答该问题的财报原文段落"
)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = os.environ.get("DEEPSEEK_BASE", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

# ---------------- 切块参数 ----------------
CHUNK_TOKENS = 700          # 目标块大小（token）
CHUNK_OVERLAP = 0.10        # 10% 重叠
MIN_CHUNK_TOKENS = 60       # 过短的正文块并入前块

# ---------------- 检索参数（禁止重排 / 查询改写 / 其他多路召回） ----------------
TOP_PER_ROUTE = 20          # 向量 top20、BM25 top20
RRF_K = 60
FINAL_TOP = 10              # RRF 融合后取 top10

# ---------------- 服务 ----------------
HOST = "0.0.0.0"
PORT = int(os.environ.get("RAG_PORT", "8077"))
