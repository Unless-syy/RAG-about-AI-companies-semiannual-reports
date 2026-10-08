# -*- coding: utf-8 -*-
"""切块：章节边界优先；超长按 ~700 token 切、重叠 10%；表格整块不切。

每块元数据：{公司, 代码, 报告期, 章节, 页码范围, chunk_id}
"""
import re

from . import config

_SENT_SPLIT = re.compile(r"(?<=[。；！？])|(?<=[;!?])\s")


def load_counter(model_dir=None):
    """返回 token 计数函数；加载失败时退化为字符估算。"""
    model_dir = model_dir or config.EMBED_MODEL_DIR
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

        def count(text):
            return len(tok.encode(text, add_special_tokens=False)) if text else 0
        return count
    except Exception:
        def count(text):
            return int(len(text or "") * 0.75)
        return count


def _split_oversize(text, count, max_tok):
    """把单个超长文本元素切成不超过 max_tok 的片段（按句切，句太长再按行/字符切）。"""
    if count(text) <= max_tok:
        return [text]
    parts = [p for p in _SENT_SPLIT.split(text) if p and p.strip()]
    if len(parts) < 2:
        parts = [p for p in text.split("\n") if p.strip()]
    if len(parts) < 2:
        n = max(1, int(len(text) * max_tok / max(1, count(text))))
        return [text[i:i + n] for i in range(0, len(text), n)]

    out, buf, buf_tok = [], [], 0
    for p in parts:
        pt = count(p)
        if buf and buf_tok + pt > max_tok:
            out.append("".join(buf))
            buf, buf_tok = [], 0
        buf.append(p)
        buf_tok += pt
    if buf:
        out.append("".join(buf))
    return out


def _page_str(p0, p1):
    return f"p{p0}" if p0 == p1 else f"p{p0}-{p1}"


def _make_chunk(meta, section, items):
    """items: [(kind, text, 起页, 止页)]"""
    p0 = min(it[2] for it in items)
    p1 = max(it[3] for it in items)
    return {
        "chunk_id": "",                      # 末尾统一编号
        "company": meta["company"],
        "code": meta["code"],
        "period": meta["period"],
        "section": section,
        "pages": _page_str(p0, p1),
        "start_page": p0,
        "end_page": p1,
        "text": "\n\n".join(it[1] for it in items).strip(),
        "has_table": any(it[0] == "table" for it in items),
        "citation": f"[{meta['company']}·{meta['period']}·{section}·{_page_str(p0, p1)}]",
    }


def _refresh_spans(chunk):
    """块内容被追加后，重算页码范围与引用串。"""
    p0, p1 = chunk["start_page"], chunk["end_page"]
    chunk["pages"] = _page_str(p0, p1)
    chunk["citation"] = (f"[{chunk['company']}·{chunk['period']}·"
                         f"{chunk['section']}·{chunk['pages']}]")


def _tail_overlap(buf, count, overlap_tok):
    """取块尾约 overlap_tok 的内容作为下一块开头，至少留 1 个元素且不整块复制。

    尾部若是超大元素（例如一张 2000 token 的大表），不做重叠，
    否则整张表会被复制进下一个块。
    """
    keep, acc = [], 0
    for it in reversed(buf):
        if len(keep) + 1 > len(buf) - 1:     # 不允许把整块复制给下一块
            break
        n = count(it[1])
        if not keep and n > overlap_tok * 3:
            break
        if keep and acc + n > overlap_tok:
            break
        keep.insert(0, it)
        acc += n
    return keep, acc


def _chunk_stream(stream, meta, sec, count, max_tok, overlap_tok, min_tok):
    """按 token 累积成块。表格是不可分割的元素（整块进、整块出），
    但不强制独占一块——财务数据表前后往往紧邻「单位：千元」「主要财务指标」
    这类说明文字，放在同一块里反而让上下文完整。"""
    out = []
    buf, buf_tok = [], 0

    for kind, text, sp0, sp1 in stream:
        pieces = _split_oversize(text, count, max_tok) if kind == "text" else [text]
        for piece in pieces:
            t = count(piece)
            if buf and buf_tok + t > max_tok:
                out.append(_make_chunk(meta, sec, buf))
                buf, buf_tok = _tail_overlap(buf, count, overlap_tok)
            buf.append((kind, piece, sp0, sp1))
            buf_tok += t

    if not buf:
        return out

    # 章节收尾：过短的正文尾块并入前一块，避免碎块
    if (count("\n\n".join(i[1] for i in buf)) < min_tok
            and out and not out[-1]["has_table"] and out[-1]["section"] == sec):
        prev = out[-1]
        prev["text"] = (prev["text"] + "\n\n"
                        + "\n\n".join(i[1] for i in buf)).strip()
        prev["end_page"] = max(prev["end_page"], max(i[3] for i in buf))
        _refresh_spans(prev)
    else:
        out.append(_make_chunk(meta, sec, buf))
    return out


_PAGENO_ONLY = re.compile(r"^\s*(?:\d{1,4}\s*/\s*\d{1,4}|\d{1,4})\s*$")


def _cleanup(chunks, count, min_tok, max_tok):
    """扫尾：丢掉纯页码块；把过短块并进同章节的前一块。

    财务报表章节里大表一张接一张，表缝里常留下一行小字（表题、单位、
    「不适用」），单独立块既没信息量又会污染召回。
    """
    out = []
    for c in chunks:
        if not c["has_table"] and _PAGENO_ONLY.match(c["text"]):
            continue
        if (out and not c["has_table"] and count(c["text"]) < min_tok
                and out[-1]["section"] == c["section"]
                and count(out[-1]["text"]) + count(c["text"]) <= max_tok):
            prev = out[-1]
            prev["text"] = (prev["text"] + "\n\n" + c["text"]).strip()
            prev["end_page"] = max(prev["end_page"], c["end_page"])
            prev["has_table"] = prev["has_table"] or c["has_table"]
            _refresh_spans(prev)
            continue
        out.append(c)
    return out


def chunk_pages(pages, meta, count):
    """按章节切块。返回 chunk 列表（chunk_id 已按公司内序号编好）。"""
    max_tok = config.CHUNK_TOKENS
    overlap_tok = int(max_tok * config.CHUNK_OVERLAP)
    min_tok = config.MIN_CHUNK_TOKENS

    order, groups = [], {}
    for p in pages:
        sec = p["section"]
        if sec not in groups:
            groups[sec] = []
            order.append(sec)
        groups[sec].append(p)

    chunks = []
    for sec in order:
        if sec == "目录":
            continue                 # 目录是导航信息，不建索引块（页面仍保留用于溯源）
        stream = []
        for p in groups[sec]:
            for el in p["elements"]:
                if el["content"].strip():
                    pg = p["physical_page"]
                    stream.append((el["type"], el["content"],
                                   el.get("start_page", pg), el.get("end_page", pg)))
        chunks.extend(_chunk_stream(stream, meta, sec, count,
                                    max_tok, overlap_tok, min_tok))

    chunks = _cleanup(chunks, count, min_tok, max_tok)

    for i, c in enumerate(chunks):
        c["chunk_id"] = (f"{meta['company']}_{meta['code']}_{meta['period']}"
                         f"_c{i:05d}")
    return chunks
