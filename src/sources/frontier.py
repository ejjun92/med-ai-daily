"""Frontier AI — 주요 기관의 신규 모델 공개.

논문이 아니라 **릴리스**를 좇는다. arXiv·PubMed·S2로는 안 잡힌다:
Gemini·DeepSeek·Qwen 같은 모델은 논문 없이 모델 카드와 블로그로 먼저 나온다.

Hugging Face Hub API를 쓴다. 회사 블로그를 긁는 방법도 있지만 형식이 제각각이라
잘 깨진다. 반면 Hub는 단일 스키마이고, 실제로 원본 사이트(gail-daily-news)의
Frontier AI에 실린 TimesFM-3와 DeepSeek-V4-Flash-Vision-Exp가 여기서 그대로 잡힌다.

이 레인은 축 쿼터와 **경쟁하지 않는다**. 의료 AI 지면을 잠식하지 않도록
별도 상한을 둔다.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request

import config

API = "https://huggingface.co/api/models"
UA = {"User-Agent": "med-ai-daily/1.0 (+https://github.com/ejjun92/med-ai-daily)"}

# 파생 배포판. 원본 공개가 아니라 남의 모델을 양자화·변환해 다시 올린 것이다.
# 이걸 안 거르면 nvidia 계정이 Frontier AI를 뒤덮는다 (실측: 16건 중 13건).
_DERIVATIVE = re.compile(
    r"(-|\.)(fp8|fp4|nvfp4|int4|int8|awq|gptq|gguf|mlx|onnx|trt|tensorrt"
    r"|w4a16|w8a8|quantized|bnb|4bit|8bit)(-|\.|$)", re.I)


class FrontierUnavailable(RuntimeError):
    pass


def _get(url: str, retries: int = 3):
    last = None
    for _ in range(retries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=30) as r:
                return json.load(r)
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(2)
    raise FrontierUnavailable(f"HF 요청 실패 ({retries}회): {last}")


def _base_name(model_id: str) -> str:
    """파생판을 원본과 묶기 위한 키. Qwen3.8-27B와 Qwen3.8-27B-FP8은 같은 것이다."""
    name = model_id.split("/")[-1]
    return _DERIVATIVE.sub("-", name).rstrip("-.").lower()


def _card_summary(model_id: str, limit: int = 1200) -> str:
    """모델 카드 앞부분. 요약 프롬프트에 넣을 원재료다."""
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/{model_id}/raw/main/README.md", headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            md = r.read().decode("utf-8", "replace")
    except Exception:               # noqa: BLE001
        return ""
    body = re.sub(r"^---.*?---", "", md, flags=re.S)      # YAML front matter
    body = re.sub(r"```.*?```", " ", body, flags=re.S)    # 코드 블록
    body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)     # 이미지
    body = re.sub(r"[#*`>|\[\]()]|<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", body).strip()[:limit]


def fetch(cycle_date: str, log=print) -> list[dict]:
    """최근 창 안에서 주요 기관이 올린 신규 모델. 실패해도 예외를 밖으로 내지 않는다."""
    end = dt.date.fromisoformat(cycle_date)
    cut = (end - dt.timedelta(days=config.FRONTIER_WINDOW_DAYS - 1)).isoformat()

    seen: dict[str, dict] = {}
    skipped = 0
    for org in config.FRONTIER_ORGS:
        url = API + "?" + urllib.parse.urlencode(
            {"author": org, "sort": "createdAt", "direction": -1, "limit": 30})
        try:
            rows = _get(url)
        except FrontierUnavailable as e:
            skipped += 1
            log(f"  [frontier] ⚠️  '{org}' 건너뜀 — {e}")
            continue
        for m in rows:
            created = (m.get("createdAt") or "")[:10]
            if created < cut:
                continue
            mid = m.get("id") or ""
            if _DERIVATIVE.search(mid.split("/")[-1]):
                continue
            if (m.get("likes") or 0) < config.FRONTIER_MIN_LIKES:
                continue
            key = _base_name(mid)
            # 같은 모델의 여러 배포판이 남았다면 인기 있는 쪽만 남긴다
            if key not in seen or (m.get("likes") or 0) > (seen[key].get("likes") or 0):
                seen[key] = {"id": mid, "org": org, "created": created,
                             "likes": m.get("likes") or 0,
                             "downloads": m.get("downloads") or 0,
                             "pipeline": m.get("pipeline_tag") or "",
                             "tags": [t for t in (m.get("tags") or [])
                                      if ":" not in t and t.islower()][:5]}
        time.sleep(config.FRONTIER_REQUEST_DELAY_S)

    out = sorted(seen.values(), key=lambda d: (-d["likes"], d["id"]))[:config.FRONTIER_MAX]
    for d in out:
        d["card"] = _card_summary(d["id"])
        time.sleep(config.FRONTIER_REQUEST_DELAY_S)
    log(f"  [frontier] {cut}~{end} ({config.FRONTIER_WINDOW_DAYS}일) → {len(out)}건"
        + (f" (기관 {skipped}곳 조회 실패)" if skipped else ""))
    return out
