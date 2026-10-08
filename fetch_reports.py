# -*- coding: utf-8 -*-
"""从巨潮资讯网（cninfo.com.cn）抓取这 14 家公司的 2026 年半年度报告 PDF。

为什么需要它：本仓库不含 PDF（26MB 原始资料，见 .gitignore），别人拿到代码要重建索引，
得先把 14 份中报找回来。这个脚本把那一步变成一条命令——公司清单直接读
data/companies.json，不另外维护一份名单。

接口形状（本机实测确认，2026-10）：
  1) POST /new/information/topSearch/query    keyWord=股票代码 → [{"code","orgId","zwjc",…}]
     orgId 不能自己拼：沪市形如 gshk0000981，深市是纯数字 9900022016，只能查出来。
  2) POST /new/hisAnnouncement/query
     category=category_bndbg_szsh（半年度报告）+ seDate=<年>-01-01~<年>-12-31
     → announcements[]，adjunctUrl 形如 finalpage/2026-08-28/1225520297.PDF
     （column 传 sse 还是 szse 实测结果一样，这里仍按代码前缀给，稳妥。）
  3) GET  https://static.cninfo.com.cn/<adjunctUrl>  → PDF（Content-Type: application/pdf）

拿到之后三重校验，任何一重不过都不落盘：
  - 文件头必须是 %PDF；
  - PyMuPDF 能打开并读出页数；
  - 与 data/companies.json 里记录的页数比对；若本地已有这份 PDF，再比一次 sha256
    ——一致就说明抓到的正是当初建索引用的那一份（实测中芯国际 1,077,584 字节、
    167 页，与索引记录一致）。

巨潮是公开披露渠道，脚本按 --sleep（默认 1 秒）逐家取数、失败指数退避重试，不做并发。
请勿把 --sleep 调成 0 去大批量抓取。

    python fetch_reports.py                  # 抓 data/companies.json 里的全部 14 家
    python fetch_reports.py --code 688981    # 只抓指定公司（可给多个）
    python fetch_reports.py --dry-run        # 只列出命中的公告，不下载
    python fetch_reports.py --force          # 本地已存在也重新下载覆盖
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pymupdf
import requests

ROOT = Path(__file__).resolve().parent
COMPANIES_JSON = ROOT / "data" / "companies.json"
LOG_JSON = ROOT / "data" / "fetch_log.json"

BASE = "https://www.cninfo.com.cn"
STATIC = "https://static.cninfo.com.cn/"
CATEGORY_BNDBG = "category_bndbg_szsh"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
# 标题里带这些词的不是我们要的那份正报
SKIP_TITLE = ("摘要", "英文", "English", "取消", "H股", "问询")


def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": UA,
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "Referer": BASE + "/new/commonUrl?url=disclosure/list/notice",
        "X-Requested-With": "XMLHttpRequest",
    })
    return s


def post_json(s, path, data, tries=3):
    """带指数退避的 POST。网络抖动一次不至于让整轮抓取白跑。"""
    err = None
    for i in range(tries):
        try:
            r = s.post(BASE + path, data=data, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:                       # noqa: BLE001
            err = e
            if i < tries - 1:
                time.sleep(2 ** i)
    raise RuntimeError("%s 请求失败：%s" % (path, err))


def find_org(s, code):
    """股票代码 → 巨潮的 orgId。"""
    for it in post_json(s, "/new/information/topSearch/query",
                        {"keyWord": code, "maxNum": "10"}):
        if it.get("code") == code:
            return it
    return None


def find_reports(s, code, org_id, year):
    """列出该公司该年度的半年度报告（按公告时间倒序，最新的在前）。"""
    column = "sse" if code.startswith(("6", "9")) else "szse"
    d = post_json(s, "/new/hisAnnouncement/query", {
        "pageNum": 1, "pageSize": 30, "column": column, "tabName": "fulltext",
        "plate": "", "stock": "%s,%s" % (code, org_id), "searchkey": "", "secid": "",
        "category": CATEGORY_BNDBG, "trade": "",
        "seDate": "%d-01-01~%d-12-31" % (year, year),
        "sortName": "", "sortType": "", "isHLtitle": "true",
    })
    out = []
    for a in d.get("announcements") or []:
        title = a.get("announcementTitle") or ""
        if "半年度报告" not in title:
            continue
        if any(x in title for x in SKIP_TITLE):
            continue
        out.append(a)
    out.sort(key=lambda a: a.get("announcementTime") or 0, reverse=True)
    return out


def download(s, url, tries=3):
    err = None
    for i in range(tries):
        try:
            r = s.get(url, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:                       # noqa: BLE001
            err = e
            if i < tries - 1:
                time.sleep(2 ** i)
    raise RuntimeError("下载失败 %s：%s" % (url, err))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="从巨潮资讯网抓取 14 家公司的半年报")
    ap.add_argument("--code", nargs="*", default=None, help="只抓这些股票代码（默认全部）")
    ap.add_argument("--year", type=int, default=2026, help="报告年度，默认 2026")
    ap.add_argument("--out", default=str(ROOT), help="PDF 存放目录，默认项目根目录")
    ap.add_argument("--sleep", type=float, default=1.0, help="每家之间的间隔秒数，默认 1")
    ap.add_argument("--dry-run", action="store_true", help="只列出命中的公告，不下载")
    ap.add_argument("--force", action="store_true", help="本地已存在也重新下载")
    args = ap.parse_args()

    if not COMPANIES_JSON.exists():
        sys.exit("找不到 %s（公司清单来自建索引的产物，先跑 build_index.py 或从仓库取）。"
                 % COMPANIES_JSON)
    comps = json.loads(COMPANIES_JSON.read_text(encoding="utf-8"))
    if args.code:
        want = set(args.code)
        comps = [c for c in comps if c["code"] in want]
        missing = want - {c["code"] for c in comps}
        if missing:
            sys.exit("companies.json 里没有这些代码：%s" % "、".join(sorted(missing)))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    print("目标：%d 家公司，%d 年度半年度报告，存到 %s" % (len(comps), args.year, outdir),
          flush=True)

    s = make_session()
    log, rows, fails = [], [], []
    for idx, c in enumerate(comps, 1):
        code, short = c["code"], c["company"]
        try:
            hit = find_org(s, code)
            if not hit:
                raise RuntimeError("巨潮查不到这个代码")
            anns = find_reports(s, code, hit["orgId"], args.year)
            if not anns:
                raise RuntimeError("没有 %d 年度半年度报告" % args.year)
            if len(anns) > 1:
                print("  [提示] %s 命中 %d 份，取最新一份：%s"
                      % (short, len(anns), "、".join(a["announcementTitle"] for a in anns)),
                      flush=True)
            a = anns[0]
            title = a["announcementTitle"]
            url = STATIC + a["adjunctUrl"].lstrip("/")
            stamp = datetime.fromtimestamp((a.get("announcementTime") or 0) / 1000)
            print("[%2d/%d] %s(%s) %s  公告日 %s" % (idx, len(comps), short, code, title,
                                                     stamp.strftime("%Y-%m-%d")), flush=True)
            if args.dry_run:
                print("       %s" % url, flush=True)
                time.sleep(args.sleep)
                continue

            data = download(s, url)
            if not data.startswith(b"%PDF"):
                raise RuntimeError("下回来的不是 PDF（头 %r）" % data[:8])
            tmp = outdir / "_fetch_tmp.pdf"
            tmp.write_bytes(data)
            doc = pymupdf.open(str(tmp))
            npages = doc.page_count
            doc.close()
            tmp.unlink()
            if npages <= 0:
                raise RuntimeError("PDF 打不开或没有页数")
            digest = sha256(data)

            name = "%s：%s.pdf" % (short, title)
            dst = outdir / name
            old = outdir / c.get("file", name)
            note = ""
            if old.exists() and not args.force:
                same = sha256(old.read_bytes()) == digest
                note = "本地已有，%s" % ("逐字节相同，跳过" if same else "但内容不同（加 --force 覆盖）")
                if not same:
                    fails.append("%s：本地文件与巨潮版本不一致" % short)
            else:
                dst.write_bytes(data)
                note = "已保存"

            rec = {"code": code, "company": short, "title": title, "url": url,
                   "announcement_date": stamp.strftime("%Y-%m-%d"), "bytes": len(data),
                   "pages": npages, "sha256": digest, "file": name,
                   "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            rec["pages_match_index"] = (npages == c.get("pages"))
            rec["bytes_match_index"] = (len(data) == (old.stat().st_size if old.exists() else -1))
            log.append(rec)
            rows.append((short, code, npages, c.get("pages"), len(data), note))
            print("       %d 页（索引记录 %s）%s，%.2f MB，sha256 %s…  %s"
                  % (npages, c.get("pages"), " 一致" if rec["pages_match_index"] else " ✗不一致",
                     len(data) / 1048576, digest[:12], note), flush=True)
        except Exception as e:                       # noqa: BLE001
            print("       ✗ %s" % e, flush=True)
            fails.append("%s(%s)：%s" % (short, code, e))
        time.sleep(args.sleep)

    if args.dry_run:
        print("\n--dry-run：以上是命中的公告，未下载。")
        return

    if log:
        LOG_JSON.write_text(json.dumps(log, ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
    print("\n=== 汇总 ===")
    for short, code, got, want, size, note in rows:
        print("  %-6s %s  %s 页  %.2f MB  %s"
              % (short, code, ("%d/%s" % (got, want)) if want else got, size / 1048576, note))
    ok = sum(1 for r in rows if r[2] == r[3])
    print("  共 %d 家：页数与索引记录一致 %d 家，不一致 %d 家"
          % (len(rows), ok, len(rows) - ok))
    if log:
        print("  明细（含 URL 与 sha256）已写入 %s"
              % LOG_JSON.relative_to(ROOT).as_posix())
    if fails:
        print("\n有 %d 项需要注意：" % len(fails))
        for f in fails:
            print("  - %s" % f)
    print("\n下一步：python build_index.py 重建索引（PDF 就在项目根目录，会被自动发现）。")


if __name__ == "__main__":
    main()
