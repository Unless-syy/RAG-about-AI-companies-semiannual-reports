# -*- coding: utf-8 -*-
"""PDF 解析：PyMuPDF 取文字坐标做正文还原，pdfplumber 取表格。

每页输出结构化元素列表，元素按阅读顺序（y 坐标）排列：
    {"type": "text"|"table", "y": float, "content": str}
表格统一渲染为 Markdown（表头完整、单元格按行列对齐）。
页码一律使用 PDF 物理页码（1-based）。印刷页码另存于 printed_page 字段，
两者可能存在偏移（封面/目录排版差异），见 README 说明。
"""
import re

import pymupdf
import pdfplumber

# 判定"列间隔"的横向阈值（pt）：同一视觉行内两个 span 间距超过它才插入分隔符
COL_GAP = 20.0
# 页眉/页脚候选区（占页高比例）
HDR_BAND = 0.10
FTR_BAND = 0.90

SECTION_RE = re.compile(r"^第\s*([一二三四五六七八九十]+)\s*节\s*[、,，:：]?\s*(.{0,30})$")
CODE_RES = [
    re.compile(r"公司代码[：:]\s*([0-9]{6})"),
    re.compile(r"证券代码[：:]\s*([0-9]{6})"),
    re.compile(r"股票代码[：:]\s*([0-9]{6})"),
]
PAGENO_RE = re.compile(r"^[\s\-—–]*([0-9]{1,4})\s*/\s*([0-9]{1,4})[\s\-—–]*$")
PAGENO_ALT_RE = re.compile(r"^[\s\-—–]*([0-9]{1,4})[\s\-—–]*$")
RUNNING_HEAD_RE = re.compile(r"^.{2,40}20\d{2}\s*年\s*半\s*年\s*度\s*报\s*告(全\s*文)?$")
# "股票代码" 后紧跟 6 位数字（排除 005930.KS 这类境外代码：数字后不能再跟数字或点）
CODE_HINT_RE = re.compile(r"(?:股票|证券)代码[：:]{0,2}(\d{6})(?![.\d])")
STOCK_PREFIX = ("60", "68", "00", "30", "83", "87", "43", "92")
# 表格上方紧邻的单位说明行，必须跟着表格走，否则财务数字会缺单位
UNIT_RE = re.compile(r"^(单位|币种)\s*[：:]")


def _norm(s):
    return re.sub(r"\s+", "", s or "")


# ---------------------------------------------------------------- 元信息

def extract_meta(pdf_path):
    """公司简称 / 代码 / 报告期全称。返回 dict。

    股票代码优先从首页「公司代码：XXXXXX」取；有些报告（如新易盛、江波龙）
    首页没有该行，代码只出现在第 6~10 页的「公司股票简况」表格里，因此再用
    pdfplumber 扫一遍前若干页的表格。
    """
    name = pdf_path.name
    short = name.split("：")[0].split(":")[0].strip()

    m = re.search(r"(\d{4})\s*年\s*半\s*年\s*度", name)
    year = m.group(1) if m else "2026"
    period = f"{year}年半年度"

    code = ""
    doc = pymupdf.open(pdf_path)
    try:
        for i in range(min(4, len(doc))):
            t = doc[i].get_text()
            hit = next((rx.search(t) for rx in CODE_RES if rx.search(t)), None)
            if hit:
                code = hit.group(1)
                break
        if not code:
            # "股票代码" 与数字同行或紧邻下一行的写法
            for i in range(min(30, len(doc))):
                for mm in CODE_HINT_RE.finditer(doc[i].get_text().replace(" ", "")):
                    if mm.group(1)[:2] in STOCK_PREFIX:
                        code = mm.group(1)
                        break
                if code:
                    break
    finally:
        doc.close()

    if not code:
        code = _code_from_tables(pdf_path)
    return {"company": short, "code": code, "period": period}


def _code_from_tables(pdf_path, max_pages=22):
    """从「公司股票简况」表格里取股票代码（表头含"股票代码"，同行的另一格是 6 位数字）。"""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(min(max_pages, len(pdf.pages))):
                try:
                    tbs = pdf.pages[i].find_tables()
                except Exception:
                    continue
                for tb in tbs:
                    try:
                        rows = tb.extract() or []
                    except Exception:
                        continue
                    for r in rows:
                        cells = [(c or "").replace("\n", "").replace(" ", "")
                                 for c in r]
                        if not any(c == "股票代码" or c == "证券代码" for c in cells):
                            continue
                        for c in cells:
                            m = re.fullmatch(r"(\d{6})", c)
                            if m and m.group(1)[:2] in STOCK_PREFIX:
                                return m.group(1)
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------- 表格

def _clean_cell(c):
    if c is None:
        return ""
    c = c.replace(" ", " ")
    # 单元格内断行（常见于长数字被折断）直接拼接，避免 "1,234.\n56" 被拆开
    c = c.replace("\n", "")
    c = re.sub(r"[ \t　]+", " ", c)
    return c.strip()


def rows_to_md(rows, title=""):
    """二维单元格 → Markdown 表格。首行若只有单格内容则视作表标题。"""
    rows = [[_clean_cell(c) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]           # 丢弃全空行
    if not rows:
        return ""
    ncol = max(len(r) for r in rows)
    rows = [r + [""] * (ncol - len(r)) for r in rows]

    # 全空列去掉（pdfplumber 偶尔多切出空列）
    keep = [j for j in range(ncol) if any(r[j] for r in rows)]
    if keep and len(keep) < ncol:
        rows = [[r[j] for j in keep] for r in rows]
    if not rows:
        return ""

    if not title and len(rows) > 1 and sum(1 for c in rows[0] if c) == 1:
        title = rows[0][0]
        rows = rows[1:]
    if len(rows) < 1:
        return ""

    header, body = rows[0], rows[1:]
    out = []
    if title:
        out.append(f"**{title}**")
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _tables_of_page(pl_page):
    """返回 [(bbox, rows)]，bbox 为 (x0, top, x1, bottom)。"""
    res = []
    for tb in pl_page.find_tables():
        try:
            rows = tb.extract()
        except Exception:
            continue
        if not rows:
            continue
        res.append((tuple(tb.bbox), rows))
    return res


def _merge_cross_page(raw_tables, heights):
    """跨页表格拼接（不修改 raw_tables）。

    仅当：上一页末表紧贴页底、本页首表紧贴页顶、两表列数一致时，
    判定为同一张跨页表，把续页行并入起页，并从续页移除。
    调用方须用 raw_tables 的 bbox 剔除正文、用返回值的 rows 渲染，
    否则续页内容会被渲染两次。

    同时返回每张表的**物理页区间**：合并后的表横跨起页与续页，页码必须
    跟着扩展，否则块的页码范围只到起页，引用页码会漏掉续页上的数字。
    返回 (merged, spans)，spans[i][k] = (起页, 止页)。
    """
    merged = [list(t) for t in raw_tables]
    spans = [[(i + 1, i + 1) for _ in t] for i, t in enumerate(raw_tables)]
    for i in range(1, len(merged)):
        if not merged[i - 1] or not merged[i]:
            continue
        pb, prow = merged[i - 1][-1]
        cb, crow = merged[i][0]
        h = heights[i]
        pw = max(len(r) for r in prow) if prow else 0
        cw = max(len(r) for r in crow) if crow else 0
        if pw and pw == cw and cb[1] < h * 0.22 and pb[3] > h * 0.70:
            merged[i - 1][-1] = (pb, prow + crow)
            merged[i] = merged[i][1:]
            s0 = spans[i - 1][-1][0]                 # 起页沿用原表
            spans[i - 1][-1] = (s0, spans[i][0][1])  # 止页扩展到续页
            spans[i] = spans[i][1:]
    return merged, spans


# ---------------------------------------------------------------- 页眉页脚

def _collect_furniture(doc):
    """统计页眉/页脚区域的高频重复文本，用于剔除。"""
    hdr, ftr = {}, {}
    n = len(doc)
    for page in doc:
        h = page.rect.height
        d = page.get_text("dict")
        for b in d["blocks"]:
            if b.get("type") != 0:
                continue
            x0, y0, x1, y1 = b["bbox"]
            txt = _norm("".join(s["text"] for l in b.get("lines", []) for s in l["spans"]))
            if not txt:
                continue
            if y1 < h * HDR_BAND:
                hdr[txt] = hdr.get(txt, 0) + 1
            elif y0 > h * FTR_BAND:
                ftr[txt] = ftr.get(txt, 0) + 1
    thr = max(2, int(n * 0.5))
    return ({t for t, c in hdr.items() if c >= thr},
            {t for t, c in ftr.items() if c >= thr})


def _is_furniture(txt, band_set, narrow_band=False):
    if not txt:
        return True
    if txt in band_set:
        return True
    if PAGENO_RE.match(txt) or (narrow_band and PAGENO_ALT_RE.match(txt)):
        return True
    return False


def _looks_like_running_head(txt):
    """页眉区形如"XX公司2026年半年度报告"的跑马灯文本（各公司措辞略有不同，
    未必达到 50% 重复率阈值，这里按模式兜底剔除）。"""
    return bool(RUNNING_HEAD_RE.match(txt))


# ---------------------------------------------------------------- 正文

def _block_lines(block):
    """text block → 行文本列表（行内按 x 排序，大间距处插 ' | '）。"""
    out = []
    for line in block.get("lines", []):
        spans = sorted(line.get("spans", []), key=lambda s: s["bbox"][0])
        parts, prev_x1 = [], None
        for sp in spans:
            t = sp["text"]
            if not t:
                continue
            x0 = sp["bbox"][0]
            if prev_x1 is not None and x0 - prev_x1 > COL_GAP:
                parts.append(" | ")
            parts.append(t)
            prev_x1 = sp["bbox"][2]
        s = "".join(parts).strip()
        if s:
            out.append(s)
    return out


def _overlaps(bbox, rects, thresh=0.4):
    """bbox 与表格区域重叠面积占比是否超过阈值。"""
    ax0, ay0, ax1, ay1 = bbox
    aarea = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    if aarea <= 0:
        return False
    for rx0, ry0, rx1, ry1 in rects:
        ix = max(0.0, min(ax1, rx1) - max(ax0, rx0))
        iy = max(0.0, min(ay1, ry1) - max(ay0, ry0))
        if ix * iy / aarea > thresh:
            return True
    return False


def _attach_units(elements):
    """把紧贴在表格上方的「单位：千元 币种：人民币」并入表格内容。

    单位行本身不在表格 bbox 内，若留在正文里，表格块就会缺少单位说明
    （中芯国际、佰维存储等以千元/万元列示，缺单位会导致数量级误读）。
    """
    out = []
    for el in elements:
        if el["type"] == "table" and out:
            prev = out[-1]
            if (prev["type"] == "text" and len(prev["content"]) < 60
                    and UNIT_RE.match(prev["content"].strip())
                    and el["y"] - prev["y"] < 80):
                out.pop()
                el["content"] = prev["content"].strip() + "\n" + el["content"]
        out.append(el)
    return out


def _printed_page(text):
    for line in text.split("\n"):
        m = PAGENO_RE.match(line.strip())
        if m:
            return int(m.group(1))
    return None


# ---------------------------------------------------------------- 主流程

def parse_pdf(pdf_path):
    """解析单个 PDF，返回 (meta, pages)。"""
    meta = extract_meta(pdf_path)
    doc = pymupdf.open(pdf_path)
    try:
        hdr_set, ftr_set = _collect_furniture(doc)
        heights = [doc[i].rect.height for i in range(len(doc))]

        with pdfplumber.open(pdf_path) as pl:
            raw_tables = [_tables_of_page(pl.pages[i]) for i in range(len(doc))]
        merged_tables, table_spans = _merge_cross_page(raw_tables, heights)

        pages = []
        for i in range(len(doc)):
            page = doc[i]
            h = heights[i]
            raw_bboxes = [b for b, _ in raw_tables[i]]   # 剔除正文用原始区域

            elements = []
            for b in page.get_text("dict")["blocks"]:
                if b.get("type") != 0:
                    continue
                x0, y0, x1, y1 = b["bbox"]
                txt = _norm("".join(s["text"] for l in b.get("lines", []) for s in l["spans"]))
                if y1 < h * HDR_BAND and (_is_furniture(txt, hdr_set)
                                          or _looks_like_running_head(txt)):
                    continue
                if y0 > h * FTR_BAND and _is_furniture(txt, ftr_set, narrow_band=True):
                    continue
                if _overlaps(b["bbox"], raw_bboxes):
                    continue                       # 表格区域交给 pdfplumber
                lines = _block_lines(b)
                if not lines:
                    continue
                elements.append({"type": "text", "y": y0, "content": "\n".join(lines)})

            for (bbox, rows), (sp0, sp1) in zip(merged_tables[i], table_spans[i]):
                md = rows_to_md(rows)               # 渲染用合并后的表格
                if md:
                    elements.append({"type": "table", "y": bbox[1], "content": md,
                                     "start_page": sp0, "end_page": sp1})

            elements.sort(key=lambda e: e["y"])
            elements = _attach_units(elements)

            pages.append({
                "pdf": pdf_path.name,
                "company": meta["company"],
                "code": meta["code"],
                "period": meta["period"],
                "physical_page": i + 1,          # PDF 物理页码（1-based）
                "printed_page": _printed_page(page.get_text()[:400]),
                "elements": elements,
                "text": "\n\n".join(e["content"] for e in elements),
                "n_tables": len(merged_tables[i]),
            })
        return meta, pages
    finally:
        doc.close()


# ---------------------------------------------------------------- 章节

def detect_sections(pages):
    """识别章节标题，回填每页的 section 字段。返回 [(section, start_page, end_page)]。"""
    def is_toc(p):
        t = p["text"]
        if not re.search(r"目\s*录", t[:400]):
            return False
        n_sec = len(re.findall(r"第\s*[一二三四五六七八九十]+\s*节", t))
        # 目录行的特征：章节名后面跟着页码（正文里的"第X节"引用不会这么密集）
        n_line = len(re.findall(r"第\s*[一二三四五六七八九十]+\s*节[^0-9\n]{0,40}[0-9]+", t))
        return n_sec >= 4 or n_line >= 4

    toc_idx = {i for i, p in enumerate(pages) if is_toc(p)}

    marks = []            # [(idx, section_name)]
    for i, p in enumerate(pages):
        if i in toc_idx:
            continue
        lines = [l.strip() for l in p["text"].split("\n")]
        for j, line in enumerate(lines):
            m = SECTION_RE.match(line)
            if not m:
                continue
            title = m.group(2).strip(" .．、,，")
            if not title:
                # 形如 "第一节" 单独成行，标题在下一行
                for k in range(j + 1, min(j + 3, len(lines))):
                    nxt = lines[k].strip(" .．、,，")
                    if nxt and not SECTION_RE.match(nxt) and len(nxt) <= 30:
                        title = nxt
                        break
            if not title:
                title = ""
            marks.append((i, f"第{m.group(1)}节 {title}".strip()))

    if not marks:
        for i, p in enumerate(pages):
            p["section"] = "目录" if i in toc_idx else "全文"
        return [("全文", 1, len(pages))]

    # 同名章节多次出现（正文引用）时保留首次出现的物理页
    seen = {}
    for idx, name in marks:
        if name not in seen:
            seen[name] = idx

    ordered = sorted(seen.items(), key=lambda kv: kv[1])
    spans = []
    for n, (name, idx) in enumerate(ordered):
        end = ordered[n + 1][1] - 1 if n + 1 < len(ordered) else len(pages) - 1
        spans.append((name, idx, end))

    first = spans[0][1]
    for i, p in enumerate(pages):
        if i in toc_idx:
            p["section"] = "目录"
        elif i < first:
            p["section"] = "前置（封面与重要提示）"
        else:
            cur = spans[0][0]
            for name, s, e in spans:
                if i >= s:
                    cur = name
            p["section"] = cur
    return [(n, s + 1, e + 1) for n, s, e in spans]
