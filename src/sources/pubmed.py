"""PubMed 수집 — 저널 5종 (MedIA / TMI / TPAMI / Radiology:AI / IJCARS).

MICCAI가 화이트리스트에 없는 이유: 2026-08-14 실측에서 PubMed 색인이
연 33~70건으로 실제 발행량(~860편)의 6%에 불과했다. MICCAI는 S2가 담당한다.

운영 주의:
  - NCBI 레이트리밋은 **소스 IP 단위**다. 우리가 3 req/s를 지켜도 공유 IP를
    쓰는 다른 쪽 때문에 429가 날 수 있다. 그때는 PubMed만 건너뛰고
    나머지 소스로 진행한다 — job 실패가 아니다.
  - `retmax` 기본값이 20이라 명시하지 않으면 조용히 잘린다.
  - `efetch`는 200건씩 POST로 보낸다 (GET은 URL 길이 제한에 걸린다).
"""
from __future__ import annotations

import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta

import config
from models import Paper

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedUnavailable(RuntimeError):
    """429/5xx 등으로 이번 실행에서 PubMed를 건너뛴다는 신호."""


def _common_params() -> dict:
    p = {"db": "pubmed", "tool": config.NCBI_TOOL}
    if (email := os.environ.get("NCBI_EMAIL")):
        p["email"] = email                     # repo variable로 주입 (원칙 5)
    if (key := os.environ.get("NCBI_API_KEY")):
        p["api_key"] = key                     # 있으면 3 → 10 req/s
    return p


def _delay() -> float:
    return 0.11 if os.environ.get("NCBI_API_KEY") else config.PUBMED_REQUEST_DELAY_S


def _request(endpoint: str, params: dict, data: dict | None = None,
             retries: int = 3) -> ET.Element:
    url = f"{BASE}/{endpoint}?{urllib.parse.urlencode(params)}"
    body = urllib.parse.urlencode(data).encode() if data else None
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, data=body), timeout=60) as r:
                return ET.fromstring(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503):          # 일시적 — 백오프 후 재시도
                time.sleep(_delay() * (2 ** attempt) * 5)
                continue
            raise
        except Exception as e:                          # noqa: BLE001
            last = e
            time.sleep(_delay() * (2 ** attempt))
    raise PubMedUnavailable(f"{endpoint} 실패 ({retries}회): {last}")


def _esearch(window_start: date, window_end: date, retmax: int) -> list[str]:
    journals = " OR ".join(f'"{j}"[Journal]' for j in config.PUBMED_JOURNALS)
    params = {
        **_common_params(),
        "term": f"({journals})",
        # EDAT은 NCBI 로컬 기준이나 datetype+mindate/maxdate로 넘기면 일 단위로
        # 해석된다. 우리 기준은 UTC 날짜다 (config.TIMEZONE).
        "datetype": "edat",
        "mindate": window_start.strftime("%Y/%m/%d"),
        "maxdate": window_end.strftime("%Y/%m/%d"),
        "retmax": retmax,                    # 기본 20 — 명시 필수
        "retstart": 0,
    }
    root = _request("esearch.fcgi", params)
    return [e.text for e in root.findall(".//IdList/Id") if e.text]


def _text(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    el = node.find(path)
    return "".join(el.itertext()).strip() if el is not None else None


def _to_paper(art: ET.Element) -> Paper | None:
    title = _text(art, ".//ArticleTitle")
    if not title:
        return None

    ptypes = {(e.text or "").strip() for e in art.findall(".//PublicationType")}
    if ptypes & set(config.PUBMED_EXCLUDED_TYPES):
        return None          # erratum/comment는 원논문과 같은 제목이라 오병합을 유발

    pmid = _text(art, ".//PMID")
    doi = None
    for eid in art.findall(".//ArticleId"):
        if eid.get("IdType") == "doi":
            doi = (eid.text or "").strip()

    # 초록은 여러 섹션으로 쪼개져 올 수 있다
    parts = [" ".join(e.itertext()).strip() for e in art.findall(".//Abstract/AbstractText")]
    abstract = " ".join(p for p in parts if p) or None

    authors = []
    for a in art.findall(".//Author")[:12]:
        ln, fn = _text(a, "LastName"), _text(a, "ForeName")
        if ln:
            authors.append(f"{fn} {ln}".strip())

    # 발표일: PDAT(출판일) 우선. EDAT(색인일)로 정렬하면 오래된 proceedings가
    # 대량 색인될 때 오늘 arXiv보다 앞서게 된다 (계획 D-11).
    y = _text(art, ".//Journal/JournalIssue/PubDate/Year") or ""
    m = _text(art, ".//Journal/JournalIssue/PubDate/Month") or "01"
    d = _text(art, ".//Journal/JournalIssue/PubDate/Day") or "01"
    months = {"Jan":"01","Feb":"02","Mar":"03","Apr":"04","May":"05","Jun":"06",
              "Jul":"07","Aug":"08","Sep":"09","Oct":"10","Nov":"11","Dec":"12"}
    m = months.get(m[:3], m if m.isdigit() else "01")
    announced = f"{y}-{int(m):02d}-{int(d):02d}" if y.isdigit() else ""

    return Paper(
        title=" ".join(title.split()),
        source="pubmed",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
        abstract=abstract,                    # 없어도 버리지 않는다 — 제목으로 분류
        authors=authors,
        announced_date=announced,
        pmid=pmid,
        doi=doi,
        journal=_text(art, ".//Journal/ISOAbbreviation") or _text(art, ".//Journal/Title"),
    )


def fetch(cycle_date: str, window_days: int | None = None,
          max_results: int | None = None, log=print) -> list[Paper]:
    """실패해도 예외를 밖으로 던지지 않는다 — PubMed만 건너뛰고 진행."""
    end = date.fromisoformat(cycle_date)
    days = config.PUBMED_WINDOW_DAYS if window_days is None else window_days
    start = end - timedelta(days=days - 1)
    cap = config.PUBMED_MAX if max_results is None else max_results

    try:
        pmids = _esearch(start, end, min(config.PUBMED_RETMAX, cap))
        time.sleep(_delay())
        out: list[Paper] = []
        seen: set[str] = set()
        for i in range(0, len(pmids), config.PUBMED_BATCH_SIZE):
            chunk = pmids[i:i + config.PUBMED_BATCH_SIZE]
            root = _request("efetch.fcgi", _common_params(),
                            data={"id": ",".join(chunk), "retmode": "xml"})
            for art in root.findall(".//PubmedArticle"):
                p = _to_paper(art)
                if p and p.pmid and p.pmid not in seen:
                    seen.add(p.pmid)
                    out.append(p)
            time.sleep(_delay())
        excluded = len(pmids) - len(out)
        log(f"  [pubmed] {start}~{end} ({days}일) → PMID {len(pmids)}건 → "
            f"{len(out)}건 (제외 {excluded}: erratum/comment 등)")
        return out[:cap]
    except PubMedUnavailable as e:
        log(f"  [pubmed] ⚠️  건너뜀 — {e}")
        return []
