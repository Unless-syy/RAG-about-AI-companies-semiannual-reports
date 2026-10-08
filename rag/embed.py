# -*- coding: utf-8 -*-
"""本地 Qwen3-Embedding-0.6B 向量编码。

- 模型路径读环境变量 EMBED_MODEL_DIR
- query 侧加 instruct 前缀，document 侧不加（Qwen3-Embedding 官方用法）
- 输出 L2 归一化向量，内积即余弦相似度
"""
import json
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from . import config

_MODEL = None
_BATCH = 32          # RTX 3060 Laptop 6GB：32 × 1024 token 约用 4GB 显存，留出余量


def pick_device(prefer=None):
    """优先 GPU；无 CUDA 时回退 CPU（此时建索引会明显变慢）。"""
    import torch
    if prefer:
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_model(device=None):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        dev = pick_device(device)
        _MODEL = SentenceTransformer(config.EMBED_MODEL_DIR, device=dev)
        if dev == "cuda":
            _MODEL.half()          # 0.6B 模型用 fp16，RTX 3060 上更快且精度足够
        _MODEL.max_seq_length = 1024
    return _MODEL


def format_query(q):
    """Qwen3-Embedding 的 instruct 模板：query 加指令，document 不加。"""
    return f"Instruct: {config.QUERY_INSTRUCTION}\nQuery: {q}"


def length_order(texts):
    """按文本长度排序的下标——长度相近的放进同一个 batch。

    报表块长度分布很散（270 ~ 4700 token），混着编码时整个 batch 都要
    padding 到最长那条，短块上是纯浪费；分桶后 padding 基本为零。
    """
    return sorted(range(len(texts)), key=lambda i: len(texts[i] or ""))


def encode_docs(texts, batch_size=None, show_progress=True, device=None,
                resume_path=None, on_batch=None):
    """按长度分桶编码并返回 L2 归一化向量矩阵 (n, dim)。

    resume_path 存在且行数匹配时，从上次中断处继续（配合 on_batch 落盘）。
    """
    import numpy as np

    m = get_model(device)
    bs = batch_size or _BATCH
    n = len(texts)
    order = length_order(texts)
    dim = m.get_sentence_embedding_dimension()

    emb = np.zeros((n, dim), dtype="float32")
    start = 0
    prog_path = f"{resume_path}.json" if resume_path else None
    if prog_path and os.path.exists(prog_path) and os.path.exists(resume_path):
        try:
            done = int(json.loads(open(prog_path, encoding="utf-8").read())["done"])
            prev = np.load(resume_path)
            if prev.shape == (n, dim) and 0 < done <= n:
                emb, start = prev.astype("float32"), done
                if show_progress:
                    print(f"      断点续跑：已完成 {start}/{n} 块", flush=True)
        except Exception:
            emb, start = np.zeros((n, dim), dtype="float32"), 0

    it = range(start, n, bs)
    if show_progress:
        try:
            from tqdm import tqdm
            it = tqdm(list(it), desc="      编码", unit="batch")
        except ImportError:
            it = list(it)

    for s in it:
        idx = order[s:s + bs]
        emb[idx] = m.encode([texts[i] for i in idx],
                            batch_size=bs,
                            normalize_embeddings=True,   # L2 归一化
                            show_progress_bar=False,
                            convert_to_numpy=True)
        done = min(s + bs, n)
        if resume_path and (done % (bs * 20) == 0 or done == n):
            np.save(resume_path, emb)                    # 断点落盘
            with open(prog_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"done": done, "total": n}))
        if on_batch is not None:
            on_batch(done, emb)

    for p in (prog_path, resume_path):                   # 编完了，清掉断点文件
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    return emb


def encode_queries(queries, device=None):
    m = get_model(device)
    texts = [format_query(q) for q in queries]
    return m.encode(texts, batch_size=len(texts) or 1,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True)


def embed_dim():
    return get_model().get_sentence_embedding_dimension()
