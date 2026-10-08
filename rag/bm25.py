# -*- coding: utf-8 -*-
"""BM25：jieba 分词 → 倒排索引。

自己建倒排（term → [(doc_id, tf)]），索引文件只存 posting，加载比权重库更快，
也便于把"分词方式"写清楚。参数取经典 k1=1.5, b=0.75。
"""
import math
import pickle
import re
from collections import defaultdict

import jieba

K1 = 1.5
B = 0.75

_NUM = re.compile(r"^\d[\d,.]*%?$")
_HAS_WORD = re.compile(r"[\w一-鿿]")

# 财报场景的高频虚词，去掉可减少噪音（保留数字与实词）
STOP = set("""
的 了 和 与 及 或 在 是 为 对 从 到 中 上 下 有 无 不 也 都 就 而 并 等 之 其 该 本 我 你 他
我们 公司 报告 年度 期间 情况 进行 以及 及其 这些 那些 一个 一种 由于 因此 所以 但是 然而
以及 包括 其中 根据 按照 关于 对于 通过 相关 有关 相应 分别 主要 重要 如下 如下 其他 以上
""".split())


def tokenize(text):
    """jieba 精确模式分词，保留数字与实词。查询与文档必须用同一函数。"""
    out = []
    for t in jieba.lcut((text or "").lower()):
        t = t.strip()
        if not t or t in STOP:
            continue
        if _NUM.match(t):
            if len(t) >= 2:
                out.append(t)
            continue
        if len(t) < 2 or not _HAS_WORD.search(t):
            continue
        out.append(t)
    return out


class BM25Index:
    def __init__(self, postings, doc_len, avgdl, n_docs):
        self.postings = postings          # term -> [(doc_id, tf), ...]
        self.doc_len = doc_len
        self.avgdl = avgdl or 1.0
        self.n_docs = n_docs
        self.idf = {}
        n = max(1, n_docs)
        for term, pl in postings.items():
            df = len(pl)
            self.idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    @classmethod
    def build(cls, docs):
        postings = defaultdict(list)
        doc_len = []
        for i, d in enumerate(docs):
            toks = tokenize(d)
            doc_len.append(len(toks))
            tf = defaultdict(int)
            for t in toks:
                tf[t] += 1
            for t, c in tf.items():
                postings[t].append((i, c))
        avgdl = (sum(doc_len) / len(doc_len)) if doc_len else 1.0
        return cls(dict(postings), doc_len, avgdl, len(docs))

    def search(self, query, topk=20):
        scores = defaultdict(float)
        for t in set(tokenize(query)):
            pl = self.postings.get(t)
            if not pl:
                continue
            idf = self.idf[t]
            for d, tf in pl:
                dl = self.doc_len[d]
                denom = tf + K1 * (1 - B + B * dl / self.avgdl)
                scores[d] += idf * tf * (K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:topk]

    # ---------------- 持久化 ----------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"postings": self.postings,
                         "doc_len": self.doc_len,
                         "avgdl": self.avgdl,
                         "n_docs": self.n_docs}, f, protocol=4)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["postings"], d["doc_len"], d["avgdl"], d["n_docs"])
