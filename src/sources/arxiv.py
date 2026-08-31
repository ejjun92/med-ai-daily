"""arXiv 수집 — 프리프린트 발견의 주력.

주의점 두 가지:
  - **단일 OR 쿼리**를 쓴다. 카테고리별로 따로 질의하면 cross-list 중복이
    생긴다. eess.IV 논문 대다수가 cs.CV에서 교차등재되므로 같은 논문을
    2~3회 받아 분류 비용을 중복 지불하게 된다.
  - arXiv API는 **announced 논문만** 색인하고 발표 날짜 필드를 제공하지
    않는다(submittedDate / lastUpdatedDate뿐). 롤링 윈도우가 이 근사를
    흡수한다 — 없는 필드를 찾지 말 것.
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import config
from models import Paper

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
API = "http://export.arxiv.org/api/query"


def _query(window_start: date, window_end: date) -> str:
    cats = " OR ".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    lo = window_start.strftime("%Y%m%d0000")
    hi = window_end.strftime("%Y%m%d2359")
    return f"({cats}) AND submittedDate:[{lo} TO {hi}]"


def _fetch(params: dict, retries: int = 3) -> ET.Element:
    url = f"{API}?{urllib.parse.urlencode(params)}"
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return ET.fromstring(r.read())
        except Exception as e:                       # noqa: BLE001
            last = e
            time.sleep(config.ARXIV_REQUEST_DELAY_S * (2 ** attempt))
    raise RuntimeError(f"arXiv 요청 실패 ({retries}회): {last}")


def _to_paper(e: ET.Element) -> Paper | None:
    title = " ".join((e.findtext("a:title", "", NS) or "").split())
    if not title:
        return None
    aid = (e.findtext("a:id", "", NS) or "").rsplit("/", 1)[-1]
    return Paper(
        title=title,
        source="arxiv",
        url=e.findtext("a:id", "", NS) or "",
        abstract=" ".join((e.findtext("a:summary", "", NS) or "").split()) or None,
        authors=[n.findtext("a:name", "", NS) for n in e.findall("a:author", NS)][:12],
        announced_date=(e.findtext("a:published", "", NS) or "")[:10],
        arxiv_id=aid,
        arxiv_doi=e.findtext("arxiv:doi", "", NS) or None,
        raw_comments=e.findtext("arxiv:comment", "", NS) or None,
    )


def fetch(cycle_date: str, window_days: int | None = None,
          max_results: int | None = None, log=print) -> list[Paper]:
    """cycle_date(UTC)를 끝으로 하는 롤링 윈도우 수집."""
    end = date.fromisoformat(cycle_date)
    days = config.ARXIV_WINDOW_DAYS if window_days is None else window_days
    start = end - timedelta(days=days - 1)
    cap = config.ARXIV_MAX if max_results is None else max_results

    q = _query(start, end)
    out: list[Paper] = []
    seen: set[str] = set()          # intra-run dedup (페이징 경계 중복 방지)
    page = 0
    while len(out) < cap:
        root = _fetch({"search_query": q, "start": page,
                       "max_results": min(config.ARXIV_PAGE_SIZE, cap - len(out)),
                       "sortBy": "submittedDate", "sortOrder": "descending"})
        entries = root.findall("a:entry", NS)
        if not entries:
            break
        for e in entries:
            p = _to_paper(e)
            if p and p.arxiv_id and p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                out.append(p)
        page += len(entries)
        if len(entries) < config.ARXIV_PAGE_SIZE:
            break
        time.sleep(config.ARXIV_REQUEST_DELAY_S)     # API 권장 간격

    log(f"  [arxiv] {start}~{end} ({days}일) → {len(out)}건"
        + (f" (상한 {cap} 도달)" if len(out) >= cap else ""))
    return out[:cap]
