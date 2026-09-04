"""Google Maps 服务封装：文本搜索 + 附近搜索 + 地理编码 + 分页拉取。

说明：
- Nearby Search（附近搜索）基于坐标，结果稳定；
- Text Search（文本搜索）对宽泛词（如 "beach"）间歇性返回 ZERO_RESULTS，
  因此全球搜索时加入「重试 + region 降级」链提升稳定性。
"""
import os
import time

import requests

# 密钥请通过环境变量 GOOGLE_MAPS_KEY 注入，勿硬编码提交到公开仓库
API_KEY = os.environ.get("GOOGLE_MAPS_KEY", "")

TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
NEARBY_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"

MAX_RESULTS = 60          # Google Places 分页上限（3 页 × 20）
DEFAULT_RADIUS = 50000    # 城市名未给半径时，默认 50km 范围
MAX_RADIUS = 50000        # Nearby Search 半径上限 50km
MIN_RADIUS = 100
TOKEN_DELAY = 3.0         # next_page_token 生效所需延迟（秒，实测需 >=3s）
TEXT_RETRY = 2            # 裸词文本搜索重试次数

# 裸词搜索失败时的 region 降级顺序（ccTLD，仅作地理偏好提示，非硬过滤）
_REGION_FALLBACKS = ["us", "gb", "au", "ca", "sg", "jp", "fr", "de"]


def _photo_url(ref: str, maxwidth: int = 800) -> str:
    return f"{PHOTO_URL}?maxwidth={maxwidth}&photo_reference={ref}&key={API_KEY}"


def _maps_url(name: str, lat, lng) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"


def geocode(address: str):
    """地址/城市名 → (lat, lng)。失败返回 None。"""
    resp = requests.get(GEOCODE_URL, params={"address": address, "key": API_KEY}, timeout=20)
    data = resp.json()
    if data.get("status") == "OK" and data.get("results"):
        loc = data["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    return None


def _clamp_radius(radius):
    if not radius:
        return None
    try:
        radius = int(radius)
    except (TypeError, ValueError):
        return None
    return max(MIN_RADIUS, min(radius, MAX_RADIUS))


def _clamp_limit(limit):
    try:
        limit = int(limit or 20)
    except (TypeError, ValueError):
        limit = 20
    return max(1, min(limit, MAX_RESULTS))


def _search_pages(url: str, first_params: dict, limit: int):
    """分页拉取原始 results 列表，自动处理 next_page_token 的生效延迟。"""
    collected = []
    params = first_params
    while len(collected) < limit:
        resp = requests.get(url, params=params, timeout=25)
        data = resp.json()
        status = data.get("status")
        # token 尚未生效时的兜底重试
        if status == "INVALID_REQUEST" and "pagetoken" in params:
            time.sleep(TOKEN_DELAY)
            resp = requests.get(url, params=params, timeout=25)
            data = resp.json()
            status = data.get("status")
        if status != "OK":
            break
        collected.extend(data.get("results", []))
        token = data.get("next_page_token")
        if not token or len(collected) >= limit:
            break
        params = {"pagetoken": token, "key": API_KEY}
        time.sleep(TOKEN_DELAY)
    return collected[:limit]


def _text_search_robust(query: str, limit: int):
    """全球文本搜索：裸词重试 + region 降级，规避宽泛词间歇性 ZERO_RESULTS。"""
    base = {"query": query, "key": API_KEY}
    # 尝试 1+2：裸词（含重试）
    for _ in range(TEXT_RETRY + 1):
        raw = _search_pages(TEXT_SEARCH_URL, base, limit)
        if raw:
            return raw
    # 尝试 3：依次附加 region 地理偏好
    for region in _REGION_FALLBACKS:
        raw = _search_pages(TEXT_SEARCH_URL, {**base, "region": region}, limit)
        if raw:
            return raw
    return []


def text_search(query: str, location: str = None, radius: int = None, limit: int = 20):
    """搜索地点，返回最多 limit 条（含坐标 + 照片 URL）。

    - location 为 "lat,lng" 或城市/地址名，radius 单位米（仅配合 location 生效）。
    - 有 location：用附近搜索真正限定范围（稳定）。
    - 无 location：全球文本搜索（带稳健降级）。
    """
    limit = _clamp_limit(limit)

    if location:
        center = None
        if "," in location:
            try:
                lat, lng = location.split(",", 1)
                center = (float(lat.strip()), float(lng.strip()))
            except ValueError:
                center = None
        else:
            center = geocode(location.strip())

        if center:
            r = _clamp_radius(radius) or DEFAULT_RADIUS
            nparams = {
                "location": f"{center[0]},{center[1]}",
                "radius": r,
                "keyword": query,
                "key": API_KEY,
            }
            raw = _search_pages(NEARBY_SEARCH_URL, nparams, limit)
            return _format(raw)
        # 地理编码失败时降级为「文本追加 + 稳健链」
        return _format(_text_search_robust(f"{query} near {location}", limit))

    return _format(_text_search_robust(query, limit))


def _check(data):
    status = data.get("status")
    if status not in ("OK", "ZERO_RESULTS"):
        msg = data.get("error_message", status)
        raise RuntimeError(f"Google Places API 错误：{status} — {msg}")


def _format(results):
    places = []
    for r in results:
        geo = r.get("geometry", {}).get("location", {})
        lat = geo.get("lat")
        lng = geo.get("lng")
        photos = r.get("photos") or []
        photo_ref = photos[0].get("photo_reference") if photos else None
        places.append({
            "name": r.get("name", ""),
            "address": r.get("formatted_address", r.get("vicinity", "")),
            "lat": lat,
            "lng": lng,
            "rating": r.get("rating"),
            "rating_count": r.get("user_ratings_total"),
            "types": r.get("types", []),
            "photo_url": _photo_url(photo_ref) if photo_ref else None,
            "maps_url": _maps_url(r.get("name", ""), lat, lng) if lat and lng else None,
        })
    return places
