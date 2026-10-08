# -*- coding: utf-8 -*-
"""生成可视化界面的数据文件（跑一次，与 build_index.py 平级）。

读建好的索引（`data/chunks.jsonl` + `data/index/emb.npy`）算出两套二维坐标，
连同语料信息一起导出到 `web/`：

    web/viewer_meta.js     语料信息 + 坐标 + 逐块轻量元数据（< 500KB，随页面一起加载）
    web/viewer_chunks.js   全部块原文（约 6MB，页面按需动态加载）

**为什么是两个文件**：`file://` 双击打开时，同目录的经典 `<script src>` 可以加载，
但 `type="module"` 与 `fetch('./x.json')` 都被浏览器拒（不是 http/https 的抓取）。
所以数据只能是「经典脚本 + 全局变量」，且拆成「小的一份必须加载，大的一份用到再加载」。
改动这里时别顺手换成 ES module 或 .json。

两套坐标：
    xyRaw  直接对原始向量降维——**这是检索真实使用的空间**，召回高亮只能画在它上面。
    xyRes  先减掉本公司质心再降维——去掉「元数据前缀」的影响后，才能看到跨公司的主题结构。
           实测：原始向量 1-近邻同公司率 99.1%，减质心后降到 31.4%（随机基线 7.3%），
           而同章节率反而从 0.887 升到 0.910。

    python build_viewer.py            # 约 30s（t-SNE 两次）
    python build_viewer.py --force    # 产物已存在时强制重算
    python build_viewer.py --method pca   # 跳过 t-SNE，只出 PCA 二维（约 1s）
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import config                              # noqa: E402
from rag.retriever import _load_chunks              # noqa: E402

WEB_DIR = config.ROOT / "web"
META_JS = WEB_DIR / "viewer_meta.js"
CHUNKS_JS = WEB_DIR / "viewer_chunks.js"

# 三组语义分色（散点图上超过 3 个系列色就过不了对比度/色觉校验，公司身份交给位置）
GROUPS = [
    ("财务报告", "#2a78d6"),
    ("管理层讨论与分析", "#eb6834"),
    ("其他", "#1baf7a"),
]


# ---------------------------------------------------------------- 输入

def load_inputs():
    """返回 (chunks, emb, manifest)。"""
    import numpy as np

    if not config.CHUNKS_JSONL.exists():
        sys.exit("找不到 %s，请先运行：python build_index.py" % config.CHUNKS_JSONL)
    emb_path = config.INDEX_DIR / "emb.npy"
    if not emb_path.exists():
        sys.exit("找不到 %s，请先运行：python build_index.py" % emb_path)

    chunks = _load_chunks(config.CHUNKS_JSONL)
    emb = np.load(emb_path)
    manifest_path = config.INDEX_DIR / "manifest.json"
    manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.exists() else {})

    # global_index 与 emb.npy 的行序一一对应，这是整套坐标的硬前提
    if len(chunks) != emb.shape[0]:
        sys.exit("块数(%d)与向量行数(%d)不一致，请重建索引"
                 % (len(chunks), emb.shape[0]))
    for i, c in enumerate(chunks):
        gi = c.get("global_index")
        if gi is not None and gi != i:
            sys.exit("第 %d 行的 global_index=%r 与行号不符，请重建索引" % (i, gi))
    return chunks, emb, manifest


# ---------------------------------------------------------------- 坐标

def _tsne(P, seed, perplexity, max_iter):
    from sklearn.manifold import TSNE
    args = dict(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", max_iter=max_iter, method="barnes_hut",
                angle=0.5, random_state=seed, verbose=0)
    try:
        return TSNE(metric="cosine", **args).fit_transform(P), "cosine"
    except Exception:                                  # 某些版本 BH 不支持 cosine
        return TSNE(metric="euclidean", **args).fit_transform(P), "euclidean"


def cluster_evr(X, labels):
    """某个标签（公司/章节）解释了多少嵌入方差：1 − 组内平方和 / 总平方和。

    这是「地图为什么是 14 个孤岛」的量化根据：公司身份解释约 42.8%、章节只有 8.8%，
    说明向量里主导的成分是「这是哪家公司」而非「这段在讲什么」。也正因为如此，
    xyRaw 只能当检索空间的忠实投影看，真正有主题信息的是去掉公司质心的 xyRes。
    """
    import numpy as np

    X = X.astype("float64")
    lab = np.asarray(labels)
    total = float(((X - X.mean(0)) ** 2).sum())
    within = 0.0
    for name in set(labels):
        sub = X[lab == name]
        within += float(((sub - sub.mean(0)) ** 2).sum())
    return round(1.0 - within / total, 4) if total else None


def embed_xy(emb, chunks, seed=42, method="tsne", perplexity=30.0, max_iter=1000):
    """返回 (xy_raw, xy_res, info)。两套坐标都是 (n, 2) float32。"""
    import numpy as np
    from sklearn.decomposition import PCA

    def reduce2(X):
        """PCA(50) 去噪 → 二维。必须给 random_state：3007×1024 会走 randomized SVD。"""
        if method == "pca":
            p = PCA(n_components=2, random_state=seed)
            return p.fit_transform(X), float(p.explained_variance_ratio_.sum()), "-"
        k = min(50, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=k, random_state=seed)
        P = pca.fit_transform(X)
        evr = float(pca.explained_variance_ratio_.sum())
        xy, metric = _tsne(P, seed, perplexity, max_iter)
        return xy, evr, metric

    t0 = time.time()
    xy_raw, evr_raw, metric = reduce2(emb.astype("float64"))
    t_raw = time.time() - t0

    # 残差视图：减掉本公司质心，再按行 L2 归一化（破坏归一化前提就不能用内积了）
    idx = {}
    for i, c in enumerate(chunks):
        idx.setdefault(c["company"], []).append(i)
    R = emb.astype("float64").copy()
    for comp, rows in idx.items():
        R[rows] -= R[rows].mean(axis=0, keepdims=True)
    norm = np.linalg.norm(R, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    R /= norm

    t0 = time.time()
    xy_res, _, _ = reduce2(R)
    t_res = time.time() - t0

    info = {"pca50_evr": round(evr_raw, 4), "tsne_metric": metric,
            "perplexity": perplexity, "max_iter": max_iter, "method": method,
            "n_same_company_1nn": _same_ratio(emb, chunks, 1),
            "res_same_company_1nn": _same_ratio(R, chunks, 1),
            "company_evr": cluster_evr(emb, [c["company"] for c in chunks]),
            "section_evr": cluster_evr(emb, [c["section"] for c in chunks])}
    # 耗时单独返回、不进产物：它是墙上时钟实测值，写进 viewer_meta.js 会让
    # 同一次构建不可复现（两次跑出来的 sha256 必然不同）。产物只放与时间无关的量，
    # 耗时打到控制台看。
    secs = {"tsne_seconds": round(t_raw, 1), "res_seconds": round(t_res, 1)}
    return xy_raw.astype("float32"), xy_res.astype("float32"), info, secs


def _same_ratio(X, chunks, k):
    """1-近邻落在同一家公司的比例——用来说明「地图其实是公司地图」。"""
    import numpy as np
    n = X.shape[0]
    if n > 6000:                       # 全量内积矩阵太大时抽样，够说明问题
        return None
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argmax(S, axis=1)
    comp = [c["company"] for c in chunks]
    return round(float(np.mean([comp[i] == comp[j] for i, j in enumerate(nn)])), 4)


# ---------------------------------------------------------------- 组装

def section_index(chunks):
    names, seen = [], {}
    for c in chunks:
        s = c["section"]
        if s not in seen:
            seen[s] = len(names)
            names.append(s)
    return names, [seen[c["section"]] for c in chunks]


def company_ranges(chunks):
    """公司 → 块区间。实测每家公司的块在 chunks.jsonl 里是连续的，
    所以只需存区间、前端二分即可，不必逐点存公司名。若不连续则返回 None。"""
    spans, i = [], 0
    while i < len(chunks):
        comp, j = chunks[i]["company"], i
        while j < len(chunks) and chunks[j]["company"] == comp:
            j += 1
        if any(c["company"] == comp for c in chunks[j:]):
            return None                # 后面又出现了 → 不连续，退回逐点存
        spans.append([i, j - i])
        i = j
    return spans


def classify_group(section):
    """章节名 → GROUPS 下标（前端按这个着色）。

    按**章节名**做子串判断，不按「第X节」的序号：各家报告的章节编号并不一致
    （「财务报告」在有的公司是第七节、有的是第八节）。「财务报告」先判，它是最
    稳的锚点；然后「管理层讨论与分析」；剩下的都归「其他」（重要事项、股份变动、
    公司治理、释义、封面等）。

    注意有一处解析遗留：某家公司有 1 个块落在名为
    「第三节 “管理层讨论与分析”中的“四、风险因素”部分内容。」的章节里——那是
    切块时把一句交叉引用误当成了章节标题。它含「管理层讨论与分析」，因此归入第 2 组，
    语义上也是对的（那 1 块正文确实是风险因素）。所以各组的块数由代码现算并显示，
    不在界面里写死。
    """
    if "财务报告" in section:
        return 0
    if "管理层讨论与分析" in section:
        return 1
    return 2


def snippet(text, n=44):
    """悬浮提示用的短预览：去掉 Markdown 表格符号与换行。"""
    t = text.replace("|", " ").replace("---", " ")
    t = " ".join(t.split())
    return t[:n] + ("…" if len(t) > n else "")


def build_meta(chunks, emb, manifest, xy_raw, xy_res, info):
    section_names, section_idx = section_index(chunks)
    ranges = company_ranges(chunks)
    group_idx = [classify_group(c["section"]) for c in chunks]
    group_counts = [group_idx.count(i) for i in range(len(GROUPS))]
    n = len(chunks)
    return {
        # 不放 generated_at：产物要可复现（同一次构建跑两次，sha256 必须一致）。
        # 构建时间打到控制台，页面里显示的是索引的 built_at（来自 manifest，稳定）。
        "apiBase": "http://127.0.0.1:%d" % config.PORT,
        "port": config.PORT,
        "corpus": {
            "n_chunks": n,
            "n_pages": manifest.get("n_pages"),
            "n_table_chunks": manifest.get("n_table_chunks"),
            "n_companies": len(manifest.get("companies") or []),
            "dim": int(emb.shape[1]),
            "embed_model": (manifest.get("embed_model") or "").replace("\\", "/").split("/")[-1],
            "device": manifest.get("device"),
            "built_at": manifest.get("built_at"),
            "chunk_tokens": manifest.get("chunk_tokens"),
            "chunk_overlap": manifest.get("chunk_overlap"),
            "rrf_k": manifest.get("rrf_k"),
            "top_per_route": manifest.get("top_per_route"),
            "final_top": manifest.get("final_top"),
            "parse_seconds": manifest.get("parse_seconds"),
            "embed_seconds": manifest.get("embed_seconds"),
        },
        "companies": manifest.get("companies") or [],
        "sectionNames": section_names,
        "groups": [{"name": g, "color": c} for g, c in GROUPS],
        "groupCounts": group_counts,        # 与 groups 同序，块数现算（见 classify_group）
        "groupIdx": group_idx,              # 逐块的组号，前端不再重写分类规则
        "sectionIdx": section_idx,
        "startPage": [c["start_page"] for c in chunks],
        "endPage": [c["end_page"] for c in chunks],
        "hasTable": [1 if c["has_table"] else 0 for c in chunks],
        "snippet": [snippet(c["text"]) for c in chunks],
        # 每项是 [起始下标, 块数]（不是 [起, 止]）；前端由 start 二分定位公司。
        # 实测 14 家各自连续、且首尾相接铺满 0..n-1，所以不必逐点存公司名。
        "compRanges": ranges,               # None → 前端退回逐点 company
        "company": [c["company"] for c in chunks] if ranges is None else None,
        "xyRaw": [round(v, 1) for v in xy_raw.reshape(-1).tolist()],
        "xyRes": [round(v, 1) for v in xy_res.reshape(-1).tolist()],
        "mapInfo": info,
    }


def write_meta_js(path, meta):
    """写成经典脚本的全局变量（file:// 下只能这样加载）。"""
    payload = json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    payload = _safe_js(payload)
    path.write_text("window.VIEWER_META=" + payload + ";\n", encoding="utf-8")
    return len(payload.encode("utf-8"))       # 按字节报大小（中文一字 3 字节）


def write_chunks_js(path, chunks):
    rows = [[c["text"], c["pages"]] for c in chunks]
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    payload = _safe_js(payload)
    path.write_text("window.VIEWER_CHUNKS=" + payload + ";\n", encoding="utf-8")
    return len(payload.encode("utf-8"))


def _safe_js(s):
    """JSON 里对 JS 字符串非法的行分隔符，以及将来若被内联时的 </script>。"""
    return (s.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
             .replace("</", "<\\/"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="产物已存在时也重算")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（决定地图可复现）")
    ap.add_argument("--method", choices=["tsne", "pca"], default="tsne")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    args = ap.parse_args()

    # 降维要用 sklearn；缺了就给一句能照做的提示，别甩 ImportError 堆栈
    try:
        import sklearn  # noqa: F401
    except ImportError:
        sys.exit("缺少 scikit-learn（语义地图降维要用到 PCA / t-SNE）。\n"
                 "  pip install -r requirements.txt      # 或 pip install scikit-learn")

    if (META_JS.exists() and CHUNKS_JS.exists() and not args.force
            and args.method == "tsne"):
        print("可视化数据已存在（%s / %s）。如需重算请加 --force"
              % (META_JS.name, CHUNKS_JS.name))
        return

    WEB_DIR.mkdir(exist_ok=True)
    print("[1/3] 读索引", flush=True)
    chunks, emb, manifest = load_inputs()
    print("      %d 块 × %d 维，%d 家公司"
          % (len(chunks), emb.shape[1], len(set(c["company"] for c in chunks))))

    print("[2/3] 降维（%s，seed=%d）" % (args.method, args.seed), flush=True)
    xy_raw, xy_res, info, secs = embed_xy(emb, chunks, seed=args.seed,
                                          method=args.method,
                                          perplexity=args.perplexity,
                                          max_iter=args.max_iter)
    print("      原始向量：%.1fs  PCA50 解释方差 %.1f%%  metric=%s"
          % (secs["tsne_seconds"], info["pca50_evr"] * 100, info["tsne_metric"]))
    print("      去公司均值：%.1fs" % secs["res_seconds"])
    print("      1-近邻同公司率：原始 %s → 去均值后 %s（随机基线取决于公司规模）"
          % (info["n_same_company_1nn"], info["res_same_company_1nn"]))

    print("[3/3] 写文件（%s）" % time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    meta = build_meta(chunks, emb, manifest, xy_raw, xy_res, info)
    n_meta = write_meta_js(META_JS, meta)
    n_chunks = write_chunks_js(CHUNKS_JS, chunks)
    print("      语义分组：%s" % "、".join(
        "%s %d 块" % (g, c) for (g, _), c in zip(GROUPS, meta["groupCounts"])))
    print("      %s  %.0f KB" % (META_JS.name, n_meta / 1024))
    print("      %s  %.1f MB" % (CHUNKS_JS.name, n_chunks / 1024 / 1024))
    print("      sha256(meta)   = %s" % sha256(META_JS)[:16])
    print("      sha256(chunks) = %s" % sha256(CHUNKS_JS)[:16])
    print("\n完成。打开 web/visual.html 即可（或起服务后访问 "
          "http://127.0.0.1:%d/visual.html）" % config.PORT)


if __name__ == "__main__":
    main()
