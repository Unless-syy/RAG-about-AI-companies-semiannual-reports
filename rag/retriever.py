# -*- coding: utf-8 -*-
"""混合检索：向量 top20 ∪ BM25 top20 → RRF 融合 → top10。

严格按要求，不做重排、不做查询改写、不引入第三路召回。
"""
import json
from collections import defaultdict

import numpy as np

from . import config, bm25 as bm25_mod, embed


def _load_chunks(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def rrf_fuse(vec_ranked, bm_ranked, k=None, topn=None):
    """Reciprocal Rank Fusion：score(d) = Σ_r 1/(k + rank_r(d))。

    vec_ranked / bm_ranked: [(doc_idx, score), ...] 已按分数降序。
    只对两路各自 top20 内出现的文档计分。
    """
    k = config.RRF_K if k is None else k
    topn = config.FINAL_TOP if topn is None else topn
    score = defaultdict(float)
    for rank, (idx, _) in enumerate(vec_ranked, start=1):
        score[idx] += 1.0 / (k + rank)
    for rank, (idx, _) in enumerate(bm_ranked, start=1):
        score[idx] += 1.0 / (k + rank)
    ranked = sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:topn]


class Retriever:
    """加载一次，常驻内存；查询时只做向量内积 + 倒排打分 + RRF。"""

    def __init__(self, chunks, emb, bm25):
        self.chunks = chunks
        self.emb = emb                       # (N, D) float32，已 L2 归一化
        self.bm25 = bm25

    @classmethod
    def load(cls, index_dir=None, warmup=True):
        index_dir = index_dir or config.INDEX_DIR
        chunks = _load_chunks(config.CHUNKS_JSONL)
        emb = np.load(index_dir / "emb.npy")
        bm = bm25_mod.BM25Index.load(index_dir / "bm25.pkl")
        bm25_mod.jieba.initialize()          # 预热分词器，避免首问卡顿
        if warmup:
            # 预热编码模型。模型是懒加载的，不预热就要等用户提第一个问题时
            # 才付这笔加载开销（本机实测约 20s），页面会卡在那里；预热后
            # 检索本身只要 0.1~0.3s。
            print("      预热编码模型（首次加载较慢，之后提问即时响应）…", flush=True)
            embed.encode_queries(["预热"])
        return cls(chunks, emb, bm)

    # -------------------------------------------------- 两路召回

    def search_vector(self, query, topk=None):
        topk = topk or config.TOP_PER_ROUTE
        q = embed.encode_queries([query])[0].astype(np.float32)
        sims = self.emb @ q                    # 均已归一化，内积即余弦
        n = len(sims)
        k = min(topk, n)
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]

    def search_bm25(self, query, topk=None):
        topk = topk or config.TOP_PER_ROUTE
        return self.bm25.search(query, topk)

    # -------------------------------------------------- 混合检索

    def retrieve(self, query, topn=None):
        topn = topn or config.FINAL_TOP
        vec = self.search_vector(query)
        bm = self.search_bm25(query)
        fused = rrf_fuse(vec, bm, topn=topn)

        vec_map = {i: (r, s) for r, (i, s) in enumerate(vec, start=1)}
        bm_map = {i: (r, s) for r, (i, s) in enumerate(bm, start=1)}

        results = []
        for rank, (idx, rrf) in enumerate(fused, start=1):
            c = dict(self.chunks[idx])
            vr, vs = vec_map.get(idx, (None, None))
            br, bs = bm_map.get(idx, (None, None))
            c.update({
                "rank": rank,
                "rrf_score": round(rrf, 6),
                "vec_rank": vr,
                "vec_score": round(vs, 4) if vs is not None else None,
                "bm25_rank": br,
                "bm25_score": round(bs, 4) if bs is not None else None,
                "in_vector": vr is not None,
                "in_bm25": br is not None,
            })
            results.append(c)
        return results
