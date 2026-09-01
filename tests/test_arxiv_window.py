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

    def fake_fetch(params, retries=3):
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
    monkeypatch.setattr(arxiv, "_fetch", lambda p, retries=3: __import__(
        "xml.etree.ElementTree", fromlist=["ElementTree"]).fromstring(
        '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'))
    assert arxiv.fetch("2026-09-01", log=lambda *_: None) == []
    assert arxiv._last_ok() is None, "0건인데 성공으로 기록했다"
