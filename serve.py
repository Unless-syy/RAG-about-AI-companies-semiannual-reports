# -*- coding: utf-8 -*-
"""一条命令启动问答服务：

    python serve.py

监听 0.0.0.0，浏览器打开 http://127.0.0.1:8077 即可使用。
（前提：已跑过 python build_index.py 建好索引）
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import uvicorn                       # noqa: E402

from rag import config               # noqa: E402


def main():
    if not config.CHUNKS_JSONL.exists():
        sys.exit("找不到索引（%s）。请先运行：python build_index.py" % config.CHUNKS_JSONL)
    # 建索引与检索共用同一个编码模型：提问也要被编码成向量才能检索，
    # 所以这里缺模型同样是硬错误，提前报出来比在预热阶段抛栈友好。
    if not config.EMBED_MODEL_DIR or not Path(config.EMBED_MODEL_DIR).exists():
        sys.exit(
            "EMBED_MODEL_DIR 未设置或路径不存在：%r\n"
            "  检索要用它把提问编码成向量，请指向本地 Qwen3-Embedding-0.6B 目录，例如：\n"
            r"    set EMBED_MODEL_DIR=D:\ai\models\Qwen3-Embedding-0.6B        (Windows)" "\n"
            "    export EMBED_MODEL_DIR=/path/to/Qwen3-Embedding-0.6B         (Linux/macOS)\n"
            "  模型取自 HuggingFace / ModelScope 的 Qwen/Qwen3-Embedding-0.6B，约 1.2GB。"
            % config.EMBED_MODEL_DIR)
    if not config.DEEPSEEK_API_KEY:
        print("[警告] 环境变量 DEEPSEEK_API_KEY 未设置，问答会失败。", flush=True)
    print(f"[rag] 启动 http://127.0.0.1:{config.PORT}/  (监听 {config.HOST})", flush=True)
    uvicorn.run("rag.server:app", host=config.HOST, port=config.PORT,
                log_level="info", reload=False, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
