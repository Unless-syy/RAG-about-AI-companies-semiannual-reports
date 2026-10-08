# -*- coding: utf-8 -*-
"""建索引（只需跑一次）。

流程：PyMuPDF/pdfplumber 解析 → 章节切块 → Qwen3-Embedding 向量编码 → BM25 倒排 → 落盘。

    python build_index.py                 # 全流程
    python build_index.py --limit 2       # 只跑前 2 份，便于调试
    python build_index.py --workers 8     # 解析并行度（默认 min(8, PDF 数)）

也支持分阶段（解析完立即落盘，编码可以单独重跑，不必重新解析）：

    python build_index.py --stage parse
    python build_index.py --stage embed
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import config  # noqa: E402

COMPANIES_JSON = config.DATA_DIR / "companies.json"


def index_text(c):
    """真正送去编码/建倒排的文本 = 元数据前缀 + 块正文。

    为什么必须加这个前缀：解析阶段剔除了页眉页脚，而页眉页脚里正好有公司名，
    于是块的正文里几乎不再出现自己公司的名字。后果是「寒武纪2026年上半年营业
    收入是多少」这种点名公司的提问，两路检索都没法用公司名把范围限定住，
    只能靠各家共有的「增长/同期/上年」这类词打分，前排会混进别家公司。

    前缀让每个块都带上「公司/代码/报告期/章节」，点名提问就能命中。前缀放在
    最前面，长块被 max_seq_length 截断时它也一定保留。展示给用户的原文
    （下面 chunks.jsonl 里的 text 字段）不受影响。
    """
    return (f"{c['company']} {c['code']} {c['period']} {c['section']}\n{c['text']}")


def _work(pdf_str):
    """单个 PDF 的解析 + 切块（在子进程里跑，解析是 CPU 密集）。"""
    from pathlib import Path as P
    from rag import parse as _parse, chunker as _chunker

    p = P(pdf_str)
    meta, pages = _parse.parse_pdf(p)
    sections = _parse.detect_sections(pages)
    count = _chunker.load_counter()
    chunks = _chunker.chunk_pages(pages, meta, count)
    slim_pages = [{
        "company": x["company"], "code": x["code"], "period": x["period"],
        "physical_page": x["physical_page"], "printed_page": x["printed_page"],
        "section": x["section"], "n_tables": x["n_tables"], "text": x["text"],
    } for x in pages]
    return {"meta": meta, "pages": slim_pages, "chunks": chunks,
            "sections": [(n, s, e) for n, s, e in sections],
            "file": p.name}


def stage_parse(pdfs, workers):
    print(f"[1/4] 解析 + 切块：{len(pdfs)} 份 PDF，{workers} 个进程", flush=True)
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_work, [str(p) for p in pdfs]):
            results.append(r)
            print(f"      ✓ {r['file'][:38]:<40} 页={len(r['pages']):>4} "
                  f"块={len(r['chunks']):>4}  章节={len(r['sections'])}", flush=True)
    t_parse = time.time() - t0

    chunks, pages_all, companies = [], [], []
    for r in results:
        chunks.extend(r["chunks"])
        pages_all.extend(r["pages"])
        companies.append({
            "company": r["meta"]["company"], "code": r["meta"]["code"],
            "period": r["meta"]["period"], "file": r["file"],
            "pages": len(r["pages"]), "chunks": len(r["chunks"]),
            "sections": [{"name": n, "start": s, "end": e} for n, s, e in r["sections"]],
        })

    for i, c in enumerate(chunks):          # 全局唯一 chunk_id
        c["chunk_id"] = f"{c['company']}_{c['code']}_{c['period']}_c{i:05d}"
        c["global_index"] = i

    config.DATA_DIR.mkdir(exist_ok=True)
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(config.PAGES_JSONL, "w", encoding="utf-8") as f:
        for p in pages_all:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with open(COMPANIES_JSON, "w", encoding="utf-8") as f:
        json.dump(companies, f, ensure_ascii=False, indent=2)

    n_tab = sum(1 for c in chunks if c["has_table"])
    print(f"\n      解析耗时 {t_parse:.1f}s（已落盘，编码阶段可单独重跑）")
    print(f"      ** 总块数：{len(chunks)} **（其中表格块 {n_tab}，正文块 "
          f"{len(chunks) - n_tab}）", flush=True)
    return chunks, t_parse


def stage_embed(chunks, t_parse=0.0, force=False):
    from rag import embed
    dev = embed.pick_device()
    if config.CHUNKS_JSONL.exists() and not chunks:
        chunks = [json.loads(l) for l in
                  open(config.CHUNKS_JSONL, encoding="utf-8") if l.strip()]
    if not chunks:
        sys.exit("没有 chunk 可编码，请先执行 --stage parse")

    manifest_path = config.INDEX_DIR / "manifest.json"
    if (not force and manifest_path.exists()
            and (config.INDEX_DIR / "emb.npy").exists()):
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if old.get("n_chunks") == len(chunks):
                print(f"索引已存在且块数一致（{len(chunks)}）。如需重建请加 --force")
                return
        except Exception:
            pass

    print(f"\n[2/4] 向量编码：{config.EMBED_MODEL_DIR}  device={dev}", flush=True)
    if dev == "cuda":
        import torch
        print(f"      GPU: {torch.cuda.get_device_name(0)}", flush=True)
    t0 = time.time()
    texts = [index_text(c) for c in chunks]
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    emb = embed.encode_docs(texts, show_progress=True, device=dev,
                            resume_path=str(config.INDEX_DIR / "emb.partial.npy"))
    t_emb = time.time() - t0
    print(f"      编码耗时 {t_emb:.1f}s  shape={emb.shape}  "
          f"({len(chunks) / max(t_emb, 1e-6):.1f} 块/秒)", flush=True)

    print("\n[3/4] BM25 倒排（jieba 分词）", flush=True)
    t0 = time.time()
    from rag import bm25 as bm25_mod
    bm = bm25_mod.BM25Index.build(texts)
    print(f"      词表 {len(bm.postings)} 词  平均块长 {bm.avgdl:.0f} token  "
          f"耗时 {time.time() - t0:.1f}s", flush=True)

    print("\n[4/4] 落盘", flush=True)
    import numpy as np
    np.save(config.INDEX_DIR / "emb.npy", emb.astype("float32"))
    bm.save(config.INDEX_DIR / "bm25.pkl")
    companies = []
    if COMPANIES_JSON.exists():
        companies = json.loads(COMPANIES_JSON.read_text(encoding="utf-8"))
    manifest = {
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_chunks": len(chunks),
        "n_table_chunks": sum(1 for c in chunks if c["has_table"]),
        "n_pages": sum(c["pages"] for c in companies) if companies else None,
        "dim": int(emb.shape[1]),
        "embed_model": config.EMBED_MODEL_DIR, "device": dev,
        "chunk_tokens": config.CHUNK_TOKENS, "chunk_overlap": config.CHUNK_OVERLAP,
        "rrf_k": config.RRF_K, "top_per_route": config.TOP_PER_ROUTE,
        "final_top": config.FINAL_TOP,
        "parse_seconds": round(t_parse, 1), "embed_seconds": round(t_emb, 1),
        "companies": companies,
    }
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    print("      chunks.jsonl / pages.jsonl / emb.npy / bm25.pkl / manifest.json")
    print(f"\n完成。总块数 {len(chunks)}，解析 {t_parse:.0f}s + 编码 {t_emb:.0f}s。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 份 PDF")
    ap.add_argument("--workers", type=int, default=0, help="解析并行进程数")
    ap.add_argument("--stage", choices=["all", "parse", "embed"], default="all")
    ap.add_argument("--force", action="store_true", help="索引已存在时强制重建")
    args = ap.parse_args()

    if not config.EMBED_MODEL_DIR or not Path(config.EMBED_MODEL_DIR).exists():
        sys.exit("EMBED_MODEL_DIR 未设置或路径不存在：%r" % config.EMBED_MODEL_DIR)

    # 建索引只需跑一次：全流程重复跑而索引已在时直接返回，免得白等一次重新解析。
    manifest_path = config.INDEX_DIR / "manifest.json"
    if (args.stage == "all" and not args.force and not args.limit
            and config.CHUNKS_JSONL.exists() and manifest_path.exists()
            and (config.INDEX_DIR / "emb.npy").exists()):
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"索引已存在：{m.get('n_chunks')} 个块，构建于 {m.get('built_at')}"
                  f"（{m.get('n_pages')} 页，device={m.get('device')}）。")
            print("无需重建。要强制重建加 --force；只重跑某一阶段用 "
                  "--stage parse / --stage embed。")
            print("现在就可以起服务：python serve.py")
            return
        except Exception:
            pass                                  # 清单坏了就走正常流程

    if args.stage in ("all", "parse"):
        pdfs = sorted(config.ROOT.glob("*.pdf"))
        if args.limit:
            pdfs = pdfs[:args.limit]
        if not pdfs:
            sys.exit("当前目录没有找到 PDF")
        workers = args.workers or min(8, len(pdfs))
        chunks, t_parse = stage_parse(pdfs, workers)
    else:
        chunks, t_parse = None, 0.0

    if args.stage in ("all", "embed"):
        stage_embed(chunks, t_parse, force=args.force)


if __name__ == "__main__":
    main()
