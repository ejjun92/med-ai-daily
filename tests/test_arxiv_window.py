"""arXiv 수집 창 — 공백을 메우는가.

2026-09-01 09:40(=00:40 UTC) 실행에서 arXiv가 0건을 돌려줬다. arXiv 발표·
재색인 시간대와 겹쳤기 때문이다. 0건은 예외가 아니라 정상 응답이라 재시도
로직에 걸리지 않았고, 그날 창에 있던 논문은 다음 창에서 빠져 사라질 뻔했다.
"""
import datetime as dt
import json

import config
import pytest
import sources.arxiv as arxiv


@pytest.fixture
def state(tmp_path, monkeypatch):
    monkeypatch.setattr(arxiv, "STATE_PATH", str(tmp_path / "arxiv.json"))
    return tmp_path / "arxiv.json"


def _window(monkeypatch, cycle_date):
    """fetch를 실제 호출하지 않고 계산된 창만 뽑아낸다."""
    seen = {}

    def fake_fetch(params, retries=3, **_):
        seen.setdefault("q", params["search_query"])
        raise RuntimeError("stop")     # 첫 페이지에서 중단

    monkeypatch.setattr(arxiv, "_fetch", fake_fetch)
    with pytest.raises(RuntimeError):
        arxiv.fetch(cycle_date, log=lambda *_: None)
    lo, hi = seen["q"].split("submittedDate:[")[1].rstrip("]").split(" TO ")
    return lo[:8], hi[:8]


def test_default_window_is_configured_length(state, monkeypatch):
    lo, hi = _window(monkeypatch, "2026-09-30")
    start = dt.date.fromisoformat("2026-09-30") - dt.timedelta(days=config.ARXIV_WINDOW_DAYS - 1)
    assert lo == start.strftime("%Y%m%d") and hi == "20260930"


def test_window_extends_back_to_last_success(state, monkeypatch):
    """지난 실행이 실패했으면 그 구간을 다시 훑는다."""
    state.write_text(json.dumps({"last_ok_date": "2026-08-20"}))
    lo, _ = _window(monkeypatch, "2026-09-30")
    assert lo <= "20260819", "마지막 성공 이전까지 되돌아가야 한다"


def test_gap_filling_is_capped(state, monkeypatch):
    """오래 멈춰 있었다고 한 번에 수만 건을 분류하지는 않는다."""
    state.write_text(json.dumps({"last_ok_date": "2020-01-01"}))
    lo, _ = _window(monkeypatch, "2026-09-30")
    floor = dt.date(2026, 9, 30) - dt.timedelta(days=config.ARXIV_MAX_WINDOW_DAYS - 1)
    assert lo == floor.strftime("%Y%m%d")


def test_success_is_recorded_only_when_papers_returned(state, monkeypatch):
    """0건이면 성공 표시를 남기지 않는다 — 다음 실행이 다시 훑도록."""
    monkeypatch.setattr(arxiv, "_fetch", lambda p, retries=3, **_: __import__(
        "xml.etree.ElementTree", fromlist=["ElementTree"]).fromstring(
        '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'))
    assert arxiv.fetch("2026-09-01", log=lambda *_: None) == []
    assert arxiv._last_ok() is None, "0건인데 성공으로 기록했다"


# ── 재시도 관측성 (실측 2026-09-04) ─────────────────────────────
def _http_error(code):
    import urllib.error
    return urllib.error.HTTPError("http://x", code, "boom", {}, None)


def test_retry_backoff_is_logged(monkeypatch):
    """조용한 백오프는 안전장치가 아니다.

    09-04 실행에서 arXiv 수집이 18초에서 751초로 늘었는데 로그에 429도 에러도
    남지 않아 원인을 구분할 수 없었다. 재시도는 반드시 흔적을 남겨야 한다.
    """
    calls = {"n": 0}

    def fake_urlopen(url, timeout=None):
        calls["n"] += 1
        calls.setdefault("timeouts", []).append(timeout)
        if calls["n"] <= 2:
            raise _http_error(429)
        import contextlib
        import io as _io
        return contextlib.closing(_io.BytesIO(
            b'<feed xmlns="http://www.w3.org/2005/Atom"></feed>'))

    slept, lines = [], []
    monkeypatch.setattr(arxiv.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: slept.append(s))
    arxiv._fetch({"search_query": "q"}, log=lines.append)

    assert slept, "백오프가 아예 일어나지 않았다"
    waits = [l for l in lines if "대기" in l]
    assert len(waits) == 2, f"재시도 2회가 로그에 남아야 한다: {lines}"
    assert "429" in waits[0], f"실패 코드를 남겨야 원인을 구분한다: {waits[0]}"
    assert [f"{s:.0f}초" in w for s, w in zip(slept, waits)] == [True] * len(slept), \
        f"각 줄의 대기 시간이 실제와 일치해야 한다: {slept} vs {waits}"
    assert any("성공" in l for l in lines), "복구 사실도 남겨야 한다"
    # 18일 장애의 1차 원인이 timeout=60 하드코딩이었다. 설정에서 읽는지
    # 직접 본다 — 상수값만 단정하면 하드코딩으로 되돌려도 통과한다.
    assert set(calls["timeouts"]) == {config.ARXIV_TIMEOUT_S}, (
        f"타임아웃이 설정에서 오지 않는다: {calls['timeouts']}")


def test_exhausted_retries_are_logged_before_raising(monkeypatch):
    monkeypatch.setattr(arxiv.urllib.request, "urlopen",
                        lambda url, timeout=60: (_ for _ in ()).throw(_http_error(503)))
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)
    lines = []
    with pytest.raises(arxiv.ArxivUnavailable):
        arxiv._fetch({"search_query": "q"}, log=lines.append)
    assert any("모두 소진" in l for l in lines), f"소진 사실이 안 남았다: {lines}"


def test_fetch_reports_elapsed_time(state, monkeypatch):
    """수집이 느려진 것을 요약 줄에서 바로 볼 수 있어야 한다."""
    monkeypatch.setattr(arxiv, "_fetch", lambda p, retries=3, **_: __import__(
        "xml.etree.ElementTree", fromlist=["ElementTree"]).fromstring(
        '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'))
    lines = []
    arxiv.fetch("2026-09-01", log=lines.append)
    summary = [l for l in lines if "→" in l and "건" in l]
    assert summary and "초]" in summary[-1], f"소요 시간이 없다: {summary}"


# ── 부분 수집 보존 (실측 2026-10-03) ───────────────────────────
def _feed(n: int, offset: int = 0, total: int | None = None,
          day: str = "2026-10-01") -> str:
    """entry n개를 담은 Atom 피드.

    opensearch:totalResults를 반드시 싣는다 — 실제 arXiv 응답에 항상 들어 있고
    (실측 2026-10-09), fetch()가 완주 판정에 쓰는 사실이다. 빼면 그 판정이
    테스트에서 무력화된다.
    """
    items = "".join(
        f'<entry><id>http://arxiv.org/abs/26{offset + i:05d}v1</id>'
        f'<title>Paper {offset + i}</title><summary>abs</summary>'
        f'<published>{day}T00:00:00Z</published></entry>'
        for i in range(n))
    return ('<feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            f'<opensearch:totalResults>{total if total is not None else n}'
            f'</opensearch:totalResults>{items}</feed>')


def _pages(monkeypatch, outcomes):
    """_fetch를 페이지별 시나리오로 바꿔 끼운다. 예외 객체면 그걸 던진다."""
    calls = {"n": 0}

    def fake(params, retries=3, **_):
        i = calls["n"]
        calls["n"] += 1
        o = outcomes[i] if i < len(outcomes) else ArithmeticError("호출 초과")
        if isinstance(o, BaseException):
            raise o
        import xml.etree.ElementTree as ET
        return ET.fromstring(o)

    monkeypatch.setattr(arxiv, "_fetch", fake)
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)
    return calls


def test_mid_page_failure_keeps_what_was_already_fetched(state, monkeypatch):
    """2페이지가 죽어도 1페이지는 살린다.

    실측 2026-10-03: 1페이지 2,000건을 받은 뒤 2페이지가 재시도를 소진했고,
    예외가 fetch() 밖으로 나가 그 2,000건이 통째로 폐기됐다. 그날 arXiv
    기여가 0이 되어 후보가 23건으로 주저앉았다.
    """
    size = config.ARXIV_PAGE_SIZE
    _pages(monkeypatch, [_feed(size, total=size * 9), arxiv.ArxivUnavailable("429 소진")])
    got = arxiv.fetch("2026-10-03", log=lambda *_: None)
    assert len(got) == size, f"받아둔 {size}건을 버렸다 (실제 {len(got)}건)"


def test_partial_harvest_marks_only_what_it_actually_covered(state, monkeypatch):
    """부분 수집은 실행일이 아니라 **덮은 지점**을 성공선으로 적는다.

    실행일로 적으면 못 받은 오래된 끝이 영구히 사라진다(정렬이 최신순이라
    절삭분은 늘 창의 오래된 쪽이다). 반대로 아무것도 안 적으면 성공선이
    고정돼 창이 바닥까지 늘어나는 래칫이 된다. 둘 다 피해야 한다.
    """
    size = config.ARXIV_PAGE_SIZE
    _pages(monkeypatch, [_feed(size, total=size * 9, day="2026-09-28"),
                         arxiv.ArxivUnavailable("429 소진")])
    arxiv.fetch("2026-10-03", log=lambda *_: None)
    assert arxiv._last_ok() == "2026-09-28", (
        f"덮은 지점이 아니라 {arxiv._last_ok()}를 적었다")


def test_cap_truncation_is_not_treated_as_complete(state, monkeypatch):
    """상한까지 받은 것은 완주가 아니다.

    창 전체가 상한보다 크면 절삭분은 창의 오래된 끝이다. 이를 완주로 적으면
    (cycle_date 기록) 그 구간이 다음 창에서 빠져 영구히 사라진다.
    실측 2026-10-09: 30일 창 totalResults=13,239 > 상한 10,000이라 매일 해당한다.
    """
    size = config.ARXIV_PAGE_SIZE
    pages = config.ARXIV_MAX // size
    _pages(monkeypatch, [_feed(size, offset=i * size, total=99_999,
                               day="2026-09-25")
                         for i in range(pages)])
    got = arxiv.fetch("2026-10-03", log=lambda *_: None)
    assert len(got) == config.ARXIV_MAX
    assert arxiv._last_ok() == "2026-09-25", (
        f"상한 절삭을 완주로 기록했다 (last_ok={arxiv._last_ok()})")


def test_paging_never_crosses_the_api_start_limit(state, monkeypatch):
    """중복제거로 수집 개수가 오프셋보다 뒤처져도 start>=10000을 밟지 않는다.

    start는 결과셋 오프셋이고 상한은 수집 개수 기준이다. 둘이 어긋나는 만큼
    오프셋이 더 전진한다 — 실측: 페이지당 중복 1건이면 9,999건 지점에서
    start=10000 요청이 나가고, arXiv가 결정론적으로 500을 돌려준다.
    """
    size = config.ARXIV_PAGE_SIZE
    starts = []

    def fake(params, retries=3, **_):
        st = params["start"]
        starts.append(st)
        if st >= config.ARXIV_START_LIMIT:
            raise AssertionError(f"페이징 한계를 넘었다: start={st}")
        # 매 페이지가 앞 페이지와 1건 겹친다
        off = st - (1 if st else 0)
        return __import__("xml.etree.ElementTree", fromlist=["x"]).fromstring(
            _feed(params["max_results"], offset=off, total=99_999))

    monkeypatch.setattr(arxiv, "_fetch", fake)
    monkeypatch.setattr(arxiv.time, "sleep", lambda s: None)
    arxiv.fetch("2026-10-03", log=lambda *_: None)
    assert max(starts) < config.ARXIV_START_LIMIT


def test_complete_harvest_is_marked_successful(state, monkeypatch):
    """반대로 끝까지 훑었으면 성공 표시를 남긴다 — 안 남기면 매일 재훑는다."""
    size = config.ARXIV_PAGE_SIZE
    # 2페이지는 id가 겹치지 않아야 한다 — 겹치면 intra-run 중복제거가 지운다
    _pages(monkeypatch, [_feed(size), _feed(3, offset=size)])
    got = arxiv.fetch("2026-10-03", log=lambda *_: None)
    assert len(got) == size + 3
    assert arxiv._last_ok() == "2026-10-03"


def test_first_page_failure_still_fails_the_source(state, monkeypatch):
    """첫 페이지부터 죽으면 소스 실패다 — 0건을 성공으로 위장하지 않는다."""
    _pages(monkeypatch, [arxiv.ArxivUnavailable("429 소진")])
    with pytest.raises(arxiv.ArxivUnavailable):
        arxiv.fetch("2026-10-03", log=lambda *_: None)
    assert arxiv._last_ok() is None


def test_partial_harvest_is_reported_in_the_log(state, monkeypatch):
    """부분 수집은 로그에 드러나야 한다 — 조용히 적게 받으면 알아챌 수 없다."""
    _pages(monkeypatch, [_feed(config.ARXIV_PAGE_SIZE, total=99_999),
                         arxiv.ArxivUnavailable("429 소진")])
    lines = []
    arxiv.fetch("2026-10-03", log=lines.append)
    assert any("페이징 중단" in l for l in lines), lines
    assert any("부분" in l for l in lines), lines
    assert any("창 전체" in l for l in lines), f"창 분량을 안 알려준다: {lines}"


def test_missing_total_results_is_not_treated_as_complete(state, monkeypatch):
    """totalResults가 없으면 완주 판정에 쓰지 않는다.

    findtext 기본값이 "0"이라, 그 값을 믿으면 page>=0으로 무조건 완주가 된다.
    entry를 받았는데 창 전체가 0일 수는 없다.
    """
    size = config.ARXIV_PAGE_SIZE
    bare = ('<feed xmlns="http://www.w3.org/2005/Atom">'
            + "".join(f'<entry><id>http://arxiv.org/abs/26{i:05d}v1</id>'
                      f'<title>P{i}</title><summary>a</summary>'
                      f'<published>2026-09-30T00:00:00Z</published></entry>'
                      for i in range(size)) + '</feed>')
    _pages(monkeypatch, [bare, arxiv.ArxivUnavailable("429 소진")])
    arxiv.fetch("2026-10-09", log=lambda *_: None)
    assert arxiv._last_ok() == "2026-09-30", (
        f"totalResults 결측을 완주로 오판했다 (last_ok={arxiv._last_ok()})")


def test_success_line_is_always_a_parseable_date(state, monkeypatch):
    """성공선에 날짜 아닌 값을 적으면 다음 실행부터 영구히 소스 실패가 된다."""
    size = config.ARXIV_PAGE_SIZE
    broken = ('<feed xmlns="http://www.w3.org/2005/Atom" '
              'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
              '<opensearch:totalResults>99999</opensearch:totalResults>'
              + "".join(f'<entry><id>http://arxiv.org/abs/26{i:05d}v1</id>'
                        f'<title>P{i}</title><summary>a</summary>'
                        f'<published>unknown</published></entry>'
                        for i in range(size)) + '</feed>')
    _pages(monkeypatch, [broken, arxiv.ArxivUnavailable("429 소진")])
    arxiv.fetch("2026-10-09", log=lambda *_: None)
    got = arxiv._last_ok()
    assert got is None or dt.date.fromisoformat(got), f"못 읽을 값을 적었다: {got!r}"
