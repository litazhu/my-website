# 以图搜地点（Deep Learning + Google Maps）

上传一张图片 → 深度学习场景识别 → 在 Google Maps 中搜索相似地点 → 返回坐标与实景照片。

> 本目录是**完整后端版**（FastAPI + PyTorch CLIP + Places API），可本地运行、支持更稳健的分页搜索。
> 若只需网页端体验，仓库根目录的 [`image-search.html`](../image-search.html) 是纯前端版
> （Transformers.js 浏览器端跑 CLIP），已部署到 GitHub Pages。

## 架构

```
浏览器 (static/index.html)
   │  上传图片
   ▼
FastAPI (main.py)
   │
   ├─ model.py   CLIP ViT-B/32 零样本场景识别（42 类地点/场景）
   │
   └─ places.py  Google Maps Places API
                  · textsearch  全球文本搜索（带重试 + region 降级）
                  · nearbysearch 附近搜索（限定范围，可翻页到 60 条）
                  · geocode     城市名 → 坐标
                  · photo       地点实景照片
                  · 返回 name / address / lat / lng / photo
```

## 技术要点

- **深度学习模型**：OpenAI CLIP `ViT-B/32`（约 338MB），用「图片-文本」对比学习做零样本分类，
  无需训练即可识别「海滩 / 雪山 / 城市天际线 / 寺庙 / 城堡…」等 42 类地点场景。
- **关键坑（已解决）**：CLIP 相似度必须乘以温度系数 `model.logit_scale.exp()`（≈100），
  否则 softmax 概率分布太平（约 3% 无法区分），乘上后正确类可达 90%+。
- **Google Maps 密钥**：通过环境变量 `GOOGLE_MAPS_KEY` 注入，不硬编码进仓库。

## 运行

```bash
# 1. 安装依赖（建议 Python 3.9 + venv）
pip install torch==2.5.1 torchvision==0.20.1 "numpy<2" openai-clip ftfy regex \
            fastapi uvicorn requests python-multipart pillow

# 2. 设置 Google Maps 密钥后启动
export GOOGLE_MAPS_KEY="你的密钥"
python -m uvicorn main:app --host 127.0.0.1 --port 8765
```

浏览器打开 http://127.0.0.1:8765（`start.sh` 提供了一键启动示例）。

## 环境说明

依赖 Python 3.9 + torch<2.6（因为工作区里原 `model.pkl` 是 fastai 2.7 导出的模型）。
核心依赖：

- torch==2.5.1、torchvision==0.20.1、numpy<2
- openai-clip==1.0.1、ftfy、regex
- fastapi、uvicorn、requests、python-multipart、pillow

> 注：工作区 `image_model/model.pkl` 是一个 fastai 导出的**二分类「是否猫」模型**（vocab 为
> `[False, True]`，标签函数 `is_cat`），与「搜相似地点」无关，故本项目改用 CLIP 做场景识别。
> 该模型可用 `load_learner('image_model/model.pkl')` 加载，但需先声明 `def is_cat(x): return x[0].isupper()`。

## 接口

- `POST /api/search`（multipart）：`file`（图片）+ 可选 `query`/`location`/`radius`/`limit`
  → 返回 `predictions`（场景+置信度）、`search_query`、`places`（name/address/lat/lng/photo_url）
- `GET /api/categories`：全部场景类别
- `GET /`：前端页面

### 搜索范围与返回数量

| 参数 | 说明 |
|---|---|
| `location` | 城市/地址名（如 `深圳`）或坐标（`lat,lng`，如 `22.54,114.05`） |
| `radius` | 半径（米，100~50000）。给城市名时默认 50km |
| `limit` | 返回数量（1~60） |

- **填了 `location`**：用附近搜索（Nearby Search）真正限定范围，**可翻页拉到最多 60 条**。
- **不填 `location`**：全球文本搜索（Text Search），上限 20 条（Google 对文本搜索的翻页 token
  在此密钥上不稳定，附近搜索翻页则正常）。

> 实测坑：Google Places 文本搜索对宽泛词（如 `beach`）会间歇性返回 `ZERO_RESULTS`，
> 已在 `_text_search_robust` 中加入「裸词重试 + region 降级」提升稳定性；附近搜索结果与翻页均稳定。

## 安全提示

- Google Maps 密钥会暴露在前端（用于地图 JS API），建议在 Google Cloud 控制台对该密钥
  开启「HTTP 引用来源限制」，只允许你自己的域名访问。
