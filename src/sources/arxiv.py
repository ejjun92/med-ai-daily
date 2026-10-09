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

NS = {"a": "http://www.w3.org/2005/Atom",
      "arxiv": "http://arxiv.org/schemas/atom",
      # 창에 몇 건이 있는지 알려준다. "다 받았는가"를 추론하지 않고
      # 사실로 판정하는 유일한 수단이다 (실측: 항상 들어 있다).
      "os": "http://a9.com/-/spec/opensearch/1.1/"}
API = "http://export.arxiv.org/api/query"
STATE_PATH = os.path.join("data", "state", "arxiv.json")


def _last_ok() -> str | None:
    """마지막으로 논문을 받아온 기준일."""
    try:
        with open(STATE_PATH) as f:
            return json.load(f).get("last_ok_date")
    except (OSError, ValueError):
        return None


def _is_iso_date(v: str | None) -> bool:
    """성공선에 쓸 수 있는 값인가.

    _last_ok는 다음 실행이 date.fromisoformat으로 읽는다. 날짜가 아닌 값을
    적으면 그 뒤로 매 실행이 조용히 소스 실패로 끝난다 — 상태 파일 하나가
    파이프라인을 영구히 막는 형태다.
    """
    try:
        date.fromisoformat(v or "")
        return True
    except ValueError:
        return False


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


class ArxivUnavailable(RuntimeError):
    """arXiv를 못 읽었다. 이번 실행만 건너뛰고 다음에 공백을 메운다."""


def _fetch(params: dict, retries: int | None = None, log=print) -> ET.Element:
    """재시도는 **반드시 로그를 남긴다**.

    이 루프는 429 한 번에 최대 7분을 조용히 잔다. 로그가 없으면 수집이 18초에서
    751초로 늘어도 원인을 로그에서 구분할 수 없다 (실측 2026-09-04: 429 흔적
    없이 733초가 사라졌다). 성공했는데 조용히 나빠지는 형태는 SSH 키 권한
    사고와 같은 계열이다 — 관측되지 않는 백오프는 안전장치가 아니다.
    """
    url = f"{API}?{urllib.parse.urlencode(params)}"
    tries = config.ARXIV_RETRIES if retries is None else retries
    last: Exception | None = None
    for attempt in range(tries):
        try:
            # 파싱까지 끝나야 성공이다. arXiv는 429/503일 때 XML 대신 HTML
            # 에러 페이지를 돌려주므로, 응답을 받은 시점에 성공을 찍으면
            # 로그가 거짓말을 한다.
            with urllib.request.urlopen(url, timeout=config.ARXIV_TIMEOUT_S) as r:
                root = ET.fromstring(r.read())
            if attempt:
                log(f"  [arxiv] 재시도 {attempt + 1}/{tries}회째에 성공")
            return root
        except Exception as e:                       # noqa: BLE001
            last = e
            # 429는 IP 단위 차단이라 몇 초로는 안 풀린다. 크게 물러선다.
            code = getattr(e, "code", None)
            why = code if code is not None else type(e).__name__
            base = config.ARXIV_RATE_LIMIT_BACKOFF_S if code in (429, 503) \
                else config.ARXIV_REQUEST_DELAY_S
            if attempt < tries - 1:
                wait = base * (2 ** attempt)
                log(f"  [arxiv] ⚠️  요청 실패 ({why}) — {wait:.0f}초 대기 후 "
                    f"재시도 {attempt + 2}/{tries}")
                time.sleep(wait)
            else:
                log(f"  [arxiv] ⚠️  요청 실패 ({why}) — {tries}회 모두 소진")
    raise ArxivUnavailable(f"arXiv 요청 실패 ({tries}회): {last}")


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
    t0 = time.monotonic()
    complete = False                # 창을 끝까지 훑었는가
    total: int | None = None        # 창의 전체 분량 (opensearch)
    # start는 **결과셋 오프셋**이라 수집 개수(len(out))와 어긋난다 — 중복제거가
    # 깎은 만큼 뒤처진다. 상한만 보고 돌면 그 차이만큼 오프셋이 더 전진해
    # arXiv의 페이징 한계(start>=10000 → 500)를 밟는다. 둘 다 본다.
    while len(out) < cap and page < min(cap, config.ARXIV_START_LIMIT):
        # 한계는 start에만 걸리는지 슬라이스 전체(start+max_results)에
        # 걸리는지 확인하지 못했다. 둘 다 안전하게 클램프한다 — 상수를
        # 바꿨을 때 10,000을 넘기는 조합이 조용히 생기는 것을 막는다.
        want = min(config.ARXIV_PAGE_SIZE, cap - len(out),
                   config.ARXIV_START_LIMIT - page)
        try:
            root = _fetch({"search_query": q, "start": page,
                           "max_results": want,
                           "sortBy": "submittedDate", "sortOrder": "descending"},
                          log=log)
        except ArxivUnavailable as e:
            # 중간 페이지가 죽어도 **앞서 받은 것은 버리지 않는다**. 예외를
            # 그대로 올리면 out이 통째로 사라진다 — 실측 2026-10-03: 1페이지
            # 2,000건을 받은 뒤 2페이지가 소진돼 그 2,000건을 폐기했고,
            # 그날 arXiv 기여가 0이 됐다. 09-02에 배운 "한 소스의 실패가
            # 전체를 죽이면 안 된다"(규칙 6)가 페이지 단위에서 되풀이된 것이다.
            if not out:
                raise                       # 첫 페이지부터 실패면 소스 실패다
            log(f"  [arxiv] ⚠️  오프셋 {page}에서 페이징 중단 — 받은 "
                f"{len(out)}건은 살린다 ({e})")
            break
        if total is None:
            total = int(root.findtext("os:totalResults", "0", NS) or 0)
        entries = root.findall("a:entry", NS)
        if not entries:
            # 빈 피드는 소진일 수도, 재색인 구간·장애일 수도 있다(아래 0건
            # 주석 참고). 중간 페이지에서 받았다면 소진으로 단정하지 않는다 —
            # 단정하면 덜 받은 것을 완주로 기록한다.
            break
        for e in entries:
            p = _to_paper(e)
            if p and p.arxiv_id and p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                out.append(p)
        page += len(entries)
        if len(entries) < want:
            complete = True             # 요청한 만큼이 안 왔다 = 마지막 페이지
            break
        time.sleep(config.ARXIV_REQUEST_DELAY_S)     # API 권장 간격

    # 완주 판정은 추론이 아니라 창 전체 분량과 대조한다. "상한까지 받았다"는
    # 완주가 아니다 — 정렬이 최신순이라 절삭분은 창의 오래된 끝이고, 그걸
    # 완주로 적으면 그 구간이 영구히 사라진다.
    # total이 0이면 필드가 없거나 못 읽은 것이다 — entry를 받았는데 전체가
    # 0일 수는 없다. 그 값을 믿으면 page>=0으로 무조건 완주가 된다.
    # 값이 있으면 **양방향으로** 덮어쓴다: 루프 안의 추론(짧은 페이지)이
    # 켠 완주도 사실이 반박할 수 있어야 한다. 한쪽으로만 쓰면 arXiv가
    # 요청보다 적게 준 날 800/13000을 완주로 기록한다.
    if total:
        complete = page >= total

    if out and complete:
        _mark_ok(cycle_date)
    elif out:
        # 부분 수집이다. cycle_date를 성공선으로 적으면 못 받은 오래된 끝이
        # 영구히 빠진다. 대신 **실제로 덮은 가장 오래된 날짜**를 적는다 —
        # _last_ok는 "여기부터는 확실히 받았다"는 경계이므로 의미가 맞고,
        # 아무것도 안 적으면 성공선이 고정돼 창이 바닥까지 늘어나는 래칫이 된다.
        #
        # ⚠️ 이것은 공백을 **메우지 못한다**. 최신순 정렬 + 상한에서는 창을
        #    넓혀도 받는 것이 똑같이 "가장 새로운 N건"이라 꼬리에 닿을 수
        #    없다(시뮬레이션 확인: 10일 중단 후 6일을 돌려도 회복 0). 여기서
        #    하는 일은 "공백을 잊지 않는 것"뿐이다 — 그래서 성공선을 앞으로
        #    **밀지 않는다**. 실제 회복은 빠진 구간만 따로 질의해야 한다.
        covered = min((p.announced_date for p in out
                       if _is_iso_date(p.announced_date)), default="")
        # 창 시작보다 과거는 이 질의가 덮을 수 없다. cross-list처럼 published가
        # 아주 오래된 entry 하나가 섞이면 성공선이 과거로 고정돼 진단값으로
        # 쓸 수 없게 된다.
        if covered:
            covered = max(covered, start.isoformat())
        prev = _last_ok()
        if covered and prev and _is_iso_date(prev):
            covered = min(covered, prev)        # 보수적으로만 움직인다
        if covered:
            _mark_ok(covered)
            log(f"  [arxiv] 부분 수집 — {covered}까지 덮었다. 그 지점을 "
                "성공선으로 남겨 다음 실행이 이어받는다")
        else:
            log("  [arxiv] 부분 수집 — 날짜를 못 읽어 성공 표시를 남기지 않는다")
    else:
        # 0건은 정상이 아니다. arXiv는 4일 창에서 늘 수백 건이 나온다.
        # 발표 재색인 구간(00~01 UTC)이나 API 장애일 때 빈 피드가 온다 —
        # 예외가 아니므로 위의 재시도 루프에 걸리지 않는다.
        log("  [arxiv] ⚠️  0건 — 재색인 구간이거나 API 장애. 성공 표시를 남기지 "
            "않으므로 다음 실행이 이 구간을 다시 훑는다")
    log(f"  [arxiv] {start}~{end} ({(end - start).days + 1}일) → {len(out)}건"
        + (f" (상한 {cap} 도달)" if len(out) >= cap else "")
        + ("" if complete or not out else f" ⚠️ 부분 (창 전체 {total or '?'}건)")
        + f" [{time.monotonic() - t0:.0f}초]")
    return out[:cap]
