@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "data\index\emb.npy" (
  echo.
  echo [x] 还没有索引。请先运行：  python build_index.py
  echo     建索引只需跑一次，之后每次启动都只跑本脚本即可。
  echo.
  pause
  exit /b 1
)

rem 可视化页面的数据是可选的：缺了不影响问答，只是 /visual.html 打开没有地图。
if not exist "web\viewer_meta.js" (
  echo.
  echo [!] 还没有语义地图数据，可视化页面 /visual.html 会打不开地图。
  echo     要补上就再跑一次（约 30 秒）：  python build_viewer.py
  echo     只要问答的话可以忽略本提示。
  echo.
)

echo 启动中……问答页 http://127.0.0.1:8077
echo            可视化页（语料信息 + 语义地图 + 召回原文）http://127.0.0.1:8077/visual.html
python serve.py
pause
