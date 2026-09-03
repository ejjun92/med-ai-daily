"""한 소스가 죽어도 나머지로 발행한다.

2026-09-02~03 이틀 동안 자동 실행이 멈췄다. arXiv가 429를 냈는데
arxiv.fetch만 예외를 밖으로 던져 파이프라인 전체가 죽었다 —
PubMed·S2는 실패해도 건너뛰고 진행하도록 되어 있었는데 arXiv만 달랐다.
사이트는 이틀 내내 09-01 내용으로 남아 있었다.
"""
import config
import ingest
import pytest
from models import Paper


def _paper(i, src="arxiv"):
    # 소스별로 식별자를 다르게 준다 — 같으면 교차 중복제거가 합쳐 버려
    # "몇 건이 살아남았나"를 못 센다.
    return Paper(title=f"{src}-P{i}", source=src,
                 arxiv_id=f"26{abs(hash(src)) % 90 + 10}{i:04d}",
                 abstract="a", announced_date="2026-09-03")


@pytest.fixture
def sources(monkeypatch):
    """세 소스의 동작을 개별로 지정한다."""
    def setup(**behaviour):
        for name, mod in (("arxiv", ingest.arxiv_src),
                          ("pubmed", ingest.pubmed_src),
                          ("s2", ingest.s2_src)):
            b = behaviour.get(name, 3)
            if isinstance(b, Exception):
                def fn(cycle_date, log=print, _e=b, **kw):
                    raise _e
            else:
                def fn(cycle_date, log=print, _n=b, _s=name, **kw):
                    return [_paper(i, _s) for i in range(_n)]
            monkeypatch.setattr(mod, "fetch", fn)
    return setup


def test_arxiv_failure_does_not_kill_the_run(sources):
    """이 테스트가 있었다면 이틀을 잃지 않았다."""
    sources(arxiv=RuntimeError("HTTP Error 429"), pubmed=2, s2=2)
    papers, stats = ingest.collect("2026-09-03", ignore_seen=True,
                                   log=lambda *_: None)
    assert len(papers) == 4, "나머지 소스로 계속 진행해야 한다"
    assert stats.failed == ["arxiv"]


def test_failed_source_is_reported_not_hidden(sources):
    sources(arxiv=RuntimeError("HTTP Error 429"))
    _, stats = ingest.collect("2026-09-03", ignore_seen=True, log=lambda *_: None)
    assert "수집 실패" in stats.render() and "arxiv" in stats.render()


def test_all_sources_failing_yields_empty_not_crash(sources):
    """전부 죽어도 예외로 끝내지 않는다 — 빈 날 경로가 페이지를 갱신한다."""
    e = RuntimeError("down")
    sources(arxiv=e, pubmed=e, s2=e)
    papers, stats = ingest.collect("2026-09-03", ignore_seen=True, log=lambda *_: None)
    assert papers == []
    assert set(stats.failed) == {"arxiv", "pubmed", "s2"}


def test_rate_limit_backoff_is_long_enough_to_matter():
    """429는 IP 단위 차단이라 3초 간격 재시도로는 못 푼다."""
    assert config.ARXIV_RATE_LIMIT_BACKOFF_S >= 30
    total = sum(config.ARXIV_RATE_LIMIT_BACKOFF_S * (2 ** i)
                for i in range(config.ARXIV_RETRIES - 1))
    assert total >= 300, "총 대기가 5분은 넘어야 한 번의 일시적 차단을 넘긴다"
