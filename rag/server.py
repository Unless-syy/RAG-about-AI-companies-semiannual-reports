# -*- coding: utf-8 -*-
"""FastAPI 服务：检索 + DeepSeek 流式问答 + 前端页面。

密钥只存在于服务端环境变量，浏览器拿不到。
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag import config, llm                      # noqa: E402
from rag.retriever import Retriever              # noqa: E402

STATE = {}


@asynccontextmanager
async def lifespan(app):
    if not config.CHUNKS_JSONL.exists():
        raise RuntimeError("索引不存在，请先运行：python build_index.py")
    print("[rag] 加载索引 …", flush=True)
    STATE["retriever"] = Retriever.load()
    n = len(STATE["retriever"].chunks)
    print(f"[rag] 就绪：{n} 个块。打开 http://127.0.0.1:{config.PORT}/", flush=True)
    yield
    STATE.clear()


app = FastAPI(title="A股半年报 RAG 问答", lifespan=lifespan)

# 允许双击打开的页面（web/visual.html 走 file://）调用本服务。
# file:// 页面的 Origin 就是字面量 "null"，所以白名单只放 "null" —— 比 ["*"] 严格得多：
# https://evil.com 的 Origin 不会被序列化成 "null"，因此任意站点的网页都拿不到这个接口。
# 密钥始终只在服务端环境变量里，浏览器拿不到任何凭据。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["null"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    allow_private_network=True,      # 回 Access-Control-Allow-Private-Network（Chrome 的 LNA）
    max_age=600,
)


class AskReq(BaseModel):
    question: str


def _public_hit(h, with_text=True):
    d = {
        "chunk_id": h["chunk_id"], "company": h["company"], "code": h["code"],
        "period": h["period"], "section": h["section"], "pages": h["pages"],
        "citation": h["citation"], "has_table": h["has_table"],
        "rank": h["rank"], "rrf_score": h["rrf_score"],
        "vec_rank": h["vec_rank"], "vec_score": h["vec_score"],
        "bm25_rank": h["bm25_rank"], "bm25_score": h["bm25_score"],
        # 语义地图上的点号：前端靠它在图上圈出本次召回的这 10 块
        "global_index": h["global_index"],
    }
    if with_text:
        d["text"] = h["text"]
    return d


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/api/meta")
def meta():
    p = config.INDEX_DIR / "manifest.json"
    if not p.exists():
        raise HTTPException(404, "索引未建立")
    with open(p, "r", encoding="utf-8") as f:
        m = json.load(f)
    m.pop("companies", None)
    return m


@app.get("/api/companies")
def companies():
    p = config.INDEX_DIR / "manifest.json"
    if not p.exists():
        raise HTTPException(404, "索引未建立")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f).get("companies", [])


@app.post("/api/ask")
def ask(req: AskReq, request: Request):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "问题不能为空")
    # 记一行 Origin：双击路径上的问题（CORS / Chrome 本地网络访问拦截）
    # 能不能成功，看服务端这一行最直接——被拦时根本不会打出来。
    origin = request.headers.get("origin")
    print(f"[rag] 提问（{len(q)} 字，Origin={origin or '无（同源或非浏览器）'}）", flush=True)
    r = STATE["retriever"]
    hits = r.retrieve(q)

    def gen():
        # 先把召回结果推给前端（侧边栏立刻渲染），再流式推答案
        yield _sse("retrieval", {"hits": [_public_hit(h) for h in hits]})
        try:
            for piece in llm.answer_stream(q, hits):
                yield _sse("delta", {"text": piece})
        except Exception as e:                    # noqa: BLE001
            yield _sse("error", {"message": str(e)})
        yield _sse("done", {})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    return {"ok": True, "chunks": len(STATE.get("retriever").chunks)
            if STATE.get("retriever") else 0}


WEB_DIR = ROOT / "web"
app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
