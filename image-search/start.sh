#!/bin/bash
# 以图搜地点 —— 一键启动脚本
# 依赖：Python 3.9 + torch 2.5.1 + openai-clip（见 README.md）
set -e
cd "$(dirname "$0")"

# 优先使用项目内 venv，否则回退到系统 python3
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

# Google Maps 密钥（通过环境变量注入，勿硬编码到仓库）
if [ -z "$GOOGLE_MAPS_KEY" ]; then
  echo "⚠️  未设置 GOOGLE_MAPS_KEY，地图/搜索功能将不可用。"
  echo "   请先执行： export GOOGLE_MAPS_KEY=\"你的密钥\""
fi

PORT=${PORT:-8765}
echo "🚀 启动「以图搜地点」服务： http://127.0.0.1:${PORT}"
exec "$PY" -m uvicorn main:app --host 127.0.0.1 --port "$PORT"
