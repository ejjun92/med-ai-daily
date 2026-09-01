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

import json
import os
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import config
from models import Paper

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
API = "http://export.arxiv.org/api/query"
STATE_PATH = os.path.join("data", "state", "arxiv.json")


def _last_ok() -> str | None:
    """마지막으로 논문을 받아온 기준일."""
    try:
        with open(STATE_PATH) as f:
            return json.load(f).get("last_ok_date")
    except (OSError, ValueError):
        return None


def _mark_ok(cycle_date: str) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"last_ok_date": cycle_date}, f)
    os.replace(tmp, tmp[:-4])       # 원자적 교체


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

    # 지난 실행이 0건이었다면 그날 창에 있던 논문은 다음 창에서 빠져 영영
    # 사라진다. 마지막 성공 지점까지 창을 뒤로 늘려 공백을 메운다.
    # (실측 2026-09-01: 00:40 UTC 실행에서 arXiv가 0건을 돌려줬다.)
    last = _last_ok()
    if last:
        gap = date.fromisoformat(last) - timedelta(days=1)     # 하루 겹쳐 잡는다
        floor = end - timedelta(days=config.ARXIV_MAX_WINDOW_DAYS - 1)
        if gap < start:
            start = max(gap, floor)
            log(f"  [arxiv] 지난 성공 {last} 이후 공백을 메운다 → {start}부터")

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

    if out:
        _mark_ok(cycle_date)
    else:
        # 0건은 정상이 아니다. arXiv는 4일 창에서 늘 수백 건이 나온다.
        # 발표 재색인 구간(00~01 UTC)이나 API 장애일 때 빈 피드가 온다 —
        # 예외가 아니므로 위의 재시도 루프에 걸리지 않는다.
        log("  [arxiv] ⚠️  0건 — 재색인 구간이거나 API 장애. 성공 표시를 남기지 "
            "않으므로 다음 실행이 이 구간을 다시 훑는다")
    log(f"  [arxiv] {start}~{end} ({(end - start).days + 1}일) → {len(out)}건"
        + (f" (상한 {cap} 도달)" if len(out) >= cap else ""))
    return out[:cap]
