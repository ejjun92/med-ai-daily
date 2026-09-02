"""Frontier AI — 의료 밖에서 주목받은 AI 논문.

의료 AI 본문과 별개 레인이다. 이 연구자가 자기 분야만 보다 큰 흐름을
놓치지 않게 하려는 구간이다.

■ 왜 Hugging Face Daily Papers인가

  처음엔 "주요 기관(Google DeepMind, Stanford…)의 논문"으로 만들려 했으나
  **논문의 소속 기관을 알아낼 방법이 없었다.** 실측:
    - 초록·제목 텍스트 매칭: 552건 중 8건만 기관명이 등장했고 그나마
      대부분 남의 모델을 인용한 경우였다
    - OpenAlex: 기관 정보는 정확하나 arXiv 프리프린트 색인이 부실하다
      (NVIDIA 최근 30일 0건, 소프트웨어 릴리스·저널 논문이 섞임)
    - arXiv affiliation 필드 / S2 authors.affiliations: 측정하려 했으나
      두 API 모두 호출 한도에 걸려 확인하지 못했다

  Daily Papers는 커뮤니티가 upvote로 고른 피드다. 기관 필터는 아니지만
  대형 연구소 논문이 실제로 상위에 몰리고, upvote가 중요도 대용이 된다.
  arXiv API를 쓰지 않아 본문 수집의 호출 한도를 잠식하지도 않는다.

  소속 필드를 쓸 수 있게 되면 여기에 기관 필터를 덧대는 것이 다음 단계다.
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.parse
import urllib.request

import config

API = "https://huggingface.co/api/daily_papers"
UA = {"User-Agent": "med-ai-daily/1.0 (+https://github.com/ejjun92/med-ai-daily)"}


class FrontierUnavailable(RuntimeError):
    pass


def _get(page: int, limit: int):
    url = f"{API}?" + urllib.parse.urlencode({"limit": limit, "p": page})
    last = None
    for _ in range(3):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=40) as r:
                return json.load(r)
        except Exception as e:      # noqa: BLE001
            last = e
            time.sleep(2)
    raise FrontierUnavailable(f"HF Daily Papers 요청 실패: {last}")


def tier(upvotes: int) -> str:
    """주목도. 원본 사이트의 Flagship/Major/Notable을 따른다."""
    if upvotes >= config.FRONTIER_FLAGSHIP_UPVOTES:
        return "FLAGSHIP"
    return "MAJOR" if upvotes >= config.FRONTIER_MAJOR_UPVOTES else "NOTABLE"


def fetch(cycle_date: str, exclude_ids: set[str] | None = None,
          log=print) -> list[dict]:
    """최근 창에서 가장 주목받은 논문. 실패해도 예외를 밖으로 내지 않는다."""
    end = dt.date.fromisoformat(cycle_date)
    cut = (end - dt.timedelta(days=config.FRONTIER_WINDOW_DAYS - 1)).isoformat()
    exclude = exclude_ids or set()

    rows: dict[str, dict] = {}
    for page in range(config.FRONTIER_PAGES):
        batch = _get(page, 100)
        if not batch:
            break
        for x in batch:
            p = x.get("paper") or {}
            pid = p.get("id")
            if pid:
                rows[pid] = p
        time.sleep(config.FRONTIER_REQUEST_DELAY_S)

    out = []
    for pid, p in rows.items():
        if (p.get("publishedAt") or "")[:10] < cut:
            continue
        if pid in exclude:          # 본문에 이미 실린 논문은 중복해 싣지 않는다
            continue
        out.append({
            "id": pid,
            "title": (p.get("title") or "").strip(),
            "url": f"https://arxiv.org/abs/{pid}",
            "date": (p.get("publishedAt") or "")[:10],
            "upvotes": p.get("upvotes") or 0,
            "abstract": " ".join((p.get("summary") or "").split()),
            "keywords": [k for k in (p.get("ai_keywords") or []) if k][:4],
            "authors": [a.get("name") for a in (p.get("authors") or [])[:3]
                        if a.get("name")],
        })
    out.sort(key=lambda d: (-d["upvotes"], d["id"]))
    out = out[:config.FRONTIER_MAX]
    log(f"  [frontier] {cut}~{end} ({config.FRONTIER_WINDOW_DAYS}일) "
        f"→ 후보 {len(rows)}건 중 {len(out)}건")
    return out
