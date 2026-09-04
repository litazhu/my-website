"""深度学习场景识别：CLIP ViT-B/32 零样本分类。

对上传图片计算 CLIP 图像特征，与预定义的地点/场景文本提示做相似度匹配，
返回 top-k 场景类别（含中文名 + 英文搜索词），供后端映射到 Google Places 搜索。
"""
import io
import threading

import torch
import clip
from PIL import Image

_device = "cpu"
_model = None
_preprocess = None
_lock = threading.Lock()

# (CLIP 文本提示, 中文显示名, Google 搜索词)
PLACE_CATEGORIES = [
    ("a photo of a sandy beach with blue ocean and waves", "海滩", "beach"),
    ("a photo of a snowy mountain peak", "雪山", "snow mountain"),
    ("a photo of a green mountain landscape", "山脉", "mountain"),
    ("a photo of a dense forest with tall trees", "森林", "forest"),
    ("a photo of a desert with sand dunes", "沙漠", "desert"),
    ("a photo of a calm lake", "湖泊", "lake"),
    ("a photo of a river", "河流", "river"),
    ("a photo of a waterfall", "瀑布", "waterfall"),
    ("a photo of a canyon with rock cliffs", "峡谷", "canyon"),
    ("a photo of a city skyline with skyscrapers", "城市天际线", "skyscraper"),
    ("a photo of a medieval castle", "城堡", "castle"),
    ("a photo of an ancient temple with pagoda", "寺庙", "temple"),
    ("a photo of a gothic cathedral church", "教堂", "cathedral"),
    ("a photo of a mosque with dome and minaret", "清真寺", "mosque"),
    ("a photo of ancient ruins and historic site", "古迹遗址", "ancient ruins"),
    ("a photo of a museum interior", "博物馆", "museum"),
    ("a photo of an amusement park with roller coaster", "游乐园", "amusement park"),
    ("a photo of a botanical garden with flowers", "植物园", "botanical garden"),
    ("a photo of a zoo with animals", "动物园", "zoo"),
    ("a photo of a shopping mall", "购物中心", "shopping mall"),
    ("a photo of a night market with street food stalls", "夜市", "night market"),
    ("a photo of a ski resort with snow", "滑雪场", "ski resort"),
    ("a photo of a tropical island", "海岛", "island"),
    ("a photo of a harbor with boats", "港口", "harbor"),
    ("a photo of a lighthouse", "灯塔", "lighthouse"),
    ("a photo of a vineyard and winery", "葡萄园/酒庄", "winery"),
    ("a photo of rice terraces in countryside", "梯田", "rice terrace"),
    ("a photo of a hot spring with steam", "温泉", "hot spring"),
    ("a photo of a glacier", "冰川", "glacier"),
    ("a photo of a volcano", "火山", "volcano"),
    ("a photo of a long bridge", "大桥", "bridge"),
    ("a photo of an aquarium with fish", "水族馆", "aquarium"),
    ("a photo of a grand library", "图书馆", "library"),
    ("a photo of a sports stadium", "体育场", "stadium"),
    ("a photo of an airport terminal", "机场", "airport"),
    ("a photo of an old historic town street", "古镇老街", "old town"),
    ("a photo of an art gallery with paintings", "美术馆", "art gallery"),
    ("a photo of a royal palace", "宫殿", "palace"),
    ("a photo of a marina with yachts", "游艇码头", "marina"),
    ("a photo of a national park landscape", "国家公园", "national park"),
    ("a photo of a cave", "洞穴", "cave"),
    ("a photo of a canyon desert landscape", "荒野", "canyon"),
]


def get_model():
    global _model, _preprocess
    with _lock:
        if _model is None:
            _model, _preprocess = clip.load("ViT-B/32", device=_device)
            _model.eval()
        return _model, _preprocess


def _tokenize_all(model):
    prompts = [c[0] for c in PLACE_CATEGORIES]
    with torch.no_grad():
        text = clip.tokenize(prompts).to(_device)
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


# 缓存文本特征，避免每次请求重复编码
_text_features_cache = None


def predict(image_bytes: bytes, top_k: int = 5):
    """对图片做零样本场景分类，返回 [{label,label_en,prob}, ...] 降序。"""
    global _text_features_cache
    model, preprocess = get_model()

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = preprocess(img).unsqueeze(0).to(_device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        if _text_features_cache is None:
            _text_features_cache = _tokenize_all(model)
        logits = image_features @ _text_features_cache.T
        logits = logits * model.logit_scale.exp()  # 应用 CLIP 温度系数
        probs = logits.softmax(dim=-1)[0]

    order = probs.argsort(descending=True)
    results = []
    for i in order[: top_k]:
        idx = int(i)
        results.append({
            "label": PLACE_CATEGORIES[idx][1],
            "label_en": PLACE_CATEGORIES[idx][2],
            "prob": round(float(probs[idx]), 4),
        })
    return results


def categories():
    """返回全部类别（供前端展示/调试）。"""
    return [{"label": c[1], "label_en": c[2], "prompt": c[0]} for c in PLACE_CATEGORIES]
