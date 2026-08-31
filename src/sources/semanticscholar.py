"""Semantic Scholar 수집 — 학회 proceedings 발견 + 학회 라벨 공급.

편입 근거(2026-08-14 실측): MICCAI의 PubMed 색인은 연 33~70건으로 실제
발행량(~860편)의 6%다. S2는 MICCAI 2024를 860편 전량 보유하고
DOI 99% / arXiv ID 52%를 함께 준다 — 교차 중복제거가 오히려 쉽다.

특성:
  - 초록 커버리지 55%. arXiv ID 보유분은 arXiv 초록으로 보완하고, 둘 다
    없으면 제목만으로 분류하되 요약 대상에서는 제외한다.
  - proceedings는 발표 시점에 일괄 등재되므로 날짜 롤링이 아니라
    연도 단위로 조회하고 published 원장으로 신규분만 걸러낸다.
  - 무인증은 공유 레이트리밋 풀이라 429가 흔하다. 실측에서도 즉시 걸렸다.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import config
from models import Paper

API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "title,abstract,venue,year,publicationDate,externalIds,authors,url"


class S2Unavailable(RuntimeError):
    """429 등으로 이번 실행에서 S2를 건너뛴다는 신호."""


def _request(params: dict, retries: int = 3) -> dict:
    headers = {}
    if (key := os.environ.get("S2_API_KEY")):
        headers["x-api-key"] = key          # 없어도 동작해야 한다 (원칙 2)
    url = f"{API}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503):
                time.sleep(config.S2_REQUEST_DELAY_S * (2 ** attempt) * 5)
                continue
            raise
        except Exception as e:              # noqa: BLE001
            last = e
            time.sleep(config.S2_REQUEST_DELAY_S * (2 ** attempt))
    raise S2Unavailable(f"S2 요청 실패 ({retries}회): {last}")


def _to_paper(d: dict) -> Paper | None:
    title = (d.get("title") or "").strip()
    if not title:
        return None
    ext = d.get("externalIds") or {}
    arxiv_id = ext.get("ArXiv")
    doi = ext.get("DOI")
    # arXiv에 있는 논문은 arXiv 쪽 레코드가 초록·링크가 안정적이라
    # 병합 시 그쪽을 우선한다 (ingest에서 처리). 여기서는 식별자만 실어보낸다.
    return Paper(
        title=" ".join(title.split()),
        source="s2",
        url=d.get("url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
        abstract=d.get("abstract"),          # 55%만 존재
        authors=[a.get("name", "") for a in (d.get("authors") or [])][:12],
        announced_date=(d.get("publicationDate") or
                        (f"{d['year']}-01-01" if d.get("year") else "")),
        arxiv_id=arxiv_id,
        doi=doi,
        pmid=ext.get("PubMed"),
        journal=d.get("venue") or None,      # venue.py가 직접 신뢰하는 필드
    )


def fetch(cycle_date: str, years_back: int | None = None,
          max_results: int | None = None, log=print) -> list[Paper]:
    """venue × 최근 N개 연도를 조회한다. 실패해도 예외를 밖으로 던지지 않는다."""
    yr = date.fromisoformat(cycle_date).year
    back = config.S2_YEARS_BACK if years_back is None else years_back
    years = [yr - i for i in range(back)]
    cap = config.S2_MAX if max_results is None else max_results

    out: list[Paper] = []
    seen: set[str] = set()
    skipped = 0
    # ⚠️ year는 **단일 연도**로만 질의한다. 콤마 구분("2026,2025")은 S2가
    #    지원하지 않아 사실상 빈 결과가 온다 — MICCAI 2025가 1,008편인데
    #    21편만 받았다(50배 누락). 연도별로 따로 돈다.
    for v in config.S2_VENUES:
        for y in years:
            if len(out) >= cap:
                break
            try:
                data = _request({"venue": v, "year": str(y), "fields": FIELDS})
            except S2Unavailable as e:
                skipped += 1
                log(f"  [s2] ⚠️  '{v[:30]}' {y} 건너뜀 — {e}")
                continue
            for d in (data.get("data") or []):
                p = _to_paper(d)
                if not p:
                    continue
                key = p.arxiv_id or p.doi or p.norm_title
                if key and key not in seen:
                    seen.add(key)
                    out.append(p)
                if len(out) >= cap:
                    break
            time.sleep(config.S2_REQUEST_DELAY_S)

    with_abs = sum(1 for p in out if p.abstract)
    with_arx = sum(1 for p in out if p.arxiv_id)
    log(f"  [s2] {years} → {len(out)}건 "
        f"(초록 {with_abs}, arXiv ID {with_arx}"
        + (f", venue {skipped}종 건너뜀" if skipped else "") + ")")
    return out[:cap]
