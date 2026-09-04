# my-website

个人公开网页，托管于 GitHub Pages。

## 页面导航

- 📚 深度学习闪卡 · <https://litazhu.github.io/my-website/>
- 🍼 宝宝 First Words · <https://litazhu.github.io/my-website/baby.html>
- 🗣️ 宝宝说 · 听音识图 · <https://litazhu.github.io/my-website/talk.html>
- 📍 以图搜地点 · <https://litazhu.github.io/my-website/image-search.html>

## 以图搜地点

上传一张照片，浏览器本地用 **CLIP（ViT-B/32）** 做零样本场景识别，展示最可能的 Top 3 场景，并自动在 Google 地图搜索相似地点，返回坐标与实景照片。

- 纯前端实现：CLIP 推理用 [Transformers.js](https://github.com/xenova/transformers.js) 在浏览器端运行（首次使用需下载约 150MB 量化权重），地点检索用 Google Maps JS API。
- 完整后端版（FastAPI + PyTorch CLIP + Places API，支持更稳健的分页搜索）见 [`image-search/`](./image-search/) 目录。

## 深度学习闪卡体验

基于 [aiquizzes-anki](https://github.com/radekosmulski/aiquizzes-anki)（fast.ai 课程闪卡合集）制作的可交互闪卡网页，共 **79 张问答卡**，覆盖 PyTorch、Python 技巧、统计基础、BatchNorm/LayerNorm、数据增强、训练技巧、模型架构、LSTM、科研与调试等主题。

- 在线访问：<https://litazhu.github.io/my-website/>
- 独立页面：<https://litazhu.github.io/my-website/aiquizzes-flashcards.html>

### 操作方式

- 点击卡片 / 按 `空格` 翻面查看答案
- `←` / `→` 切换卡片
- `1` 标记「认识」，`2` 标记「还不熟」
- 顶部可按主题筛选，底部可打乱顺序、重置进度
