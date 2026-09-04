"""FastAPI 后端：上传图片 → CLIP 场景识别 → Google Maps 搜索相似地点。"""
import os
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import model
import places

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="以图搜地点", version="2.0")


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = os.path.join(STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    return html.replace("__MAPS_KEY__", places.API_KEY)


@app.get("/api/categories")
def get_categories():
    return {"categories": model.categories()}


@app.post("/api/search")
async def search(
    file: UploadFile = File(...),
    query: Optional[str] = Form(None),
    location: Optional[str] = Form(None),
    radius: Optional[int] = Form(None),
    limit: Optional[int] = Form(None),
):
    image_bytes = await file.read()
    if not image_bytes:
        return JSONResponse({"error": "未收到图片数据"}, status_code=400)

    try:
        predictions = model.predict(image_bytes, top_k=5)
    except Exception as e:
        return JSONResponse({"error": f"模型推理失败：{e}"}, status_code=500)

    # 确定搜索词：用户手动 query 优先，否则取 top-1 场景的英文搜索词
    search_query = (query or "").strip() or predictions[0]["label_en"]

    try:
        result_places = places.text_search(
            search_query, location=location, radius=radius, limit=limit or 20
        )
    except Exception as e:
        return JSONResponse(
            {
                "error": str(e),
                "predictions": predictions,
                "search_query": search_query,
                "places": [],
            },
            status_code=502,
        )

    return {
        "predictions": predictions,
        "search_query": search_query,
        "places": result_places,
    }


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
