"""파이프라인 배선 테스트 — 모델도 네트워크도 없이 돈다.

검증하는 것은 '단계가 올바른 순서로, 올바른 대상에 대해 불린다'는 것이다.
LLM 품질은 골든셋 실측이 따로 본다.
"""
import datetime as dt

import config
import pipeline
import pytest
from ledger import DeferredLedger, PublishedLedger
from models import Classification, Entry, Paper


class FakeLLM:
    """분류·요약 호출을 기록하는 대역. 실제 vLLM 대신 들어간다."""

    def __init__(self, relevant_every=1):
        self.relevant_every = relevant_every
        self.calls = []

    def __enter__(self): return self
    def __exit__(self, *a): self.closed = True

    def chat_json(self, prompts, schema, max_tokens):
        self.calls.append((len(prompts), sorted(schema["properties"])))
        if "korean_summary" in schema["properties"]:
            return [{"tags": ["t"], "korean_summary":
                     "이 논문은 문제를 다룬다. 방법을 제안한다. 결과가 좋았다. " * 2}
                    for _ in prompts]
        out = []
        for i, _ in enumerate(prompts):
            rel = (i % self.relevant_every) == 0
            out.append({"is_relevant": rel, "axis": "surgical_video",
                        "category_id": "surgical_vlp", "stars": 4, "rationale": "x"})
        return out


def papers(n, prefix="2601."):
    return [Paper(title=f"Paper {i}", source="arxiv", arxiv_id=f"{prefix}{i:05d}",
                  abstract="초록", announced_date="2026-08-30",
                  url=f"https://arxiv.org/abs/{prefix}{i:05d}") for i in range(n)]


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """수집·모델·원장을 전부 가짜로 갈아끼운다."""
    from models import Paper as P
    state = {"llm": FakeLLM(), "collected": papers(30)}

    def fake_collect(cycle_date, **kw):
        from ingest import IngestStats
        st = IngestStats(per_source={"arxiv": len(state["collected"])})
        return list(state["collected"]), st

    monkeypatch.setattr(pipeline.ingest, "collect", fake_collect)
    monkeypatch.setattr(pipeline, "_replayed_papers", lambda *a, **k: [])
    monkeypatch.setattr("llm.LocalLLM", lambda log=print: state["llm"])
    monkeypatch.setattr(pipeline, "PublishedLedger",
                        lambda: PublishedLedger(str(tmp_path / "pub")))
    monkeypatch.setattr(pipeline, "DeferredLedger",
                        lambda: DeferredLedger(str(tmp_path / "def")))
    # 렌더 캐시도 임시 경로로 돌린다. 안 그러면 테스트가 실제 저장소의
    # data/entries/를 덮어써서, 나중에 render --all이 가짜 데이터로
    # 진짜 아카이브 페이지를 밀어낸다.
    monkeypatch.setattr(pipeline, "ENTRIES_DIR", str(tmp_path / "entries"))
    state["out"] = tmp_path / "docs"
    return state


def run(wired, **kw):
    kw.setdefault("out_dir", str(wired["out"]))
    kw.setdefault("log", lambda *_: None)
    return pipeline.run("2026-08-31", **kw)


# ── 단계 순서와 대상 ─────────────────────────────────────────
def test_summary_runs_only_on_selected_papers(wired):
    """요약을 후보 전체에 돌리면 비용이 30배가 된다. 선별된 것만 요약한다."""
    wired["collected"] = papers(400)
    n = run(wired)
    classify_calls = [c for c in wired["llm"].calls if "is_relevant" in c[1]]
    summary_calls = [c for c in wired["llm"].calls if "korean_summary" in c[1]]
    assert sum(c[0] for c in classify_calls) == 400, "분류는 후보 전체에"
    assert sum(c[0] for c in summary_calls) == n <= config.DAILY_MAX, "요약은 선별분에만"


def test_no_model_started_on_empty_day(wired, monkeypatch):
    """arXiv는 금·토 발표가 없다. 빈 날에 32B를 올리고 내릴 이유가 없다."""
    wired["collected"] = []
    boom = lambda *a, **k: pytest.fail("후보 0건인데 모델을 띄웠다")
    monkeypatch.setattr("llm.LocalLLM", boom)
    assert run(wired) == 0
    assert (wired["out"] / "index.html").exists(), "빈 날에도 페이지는 갱신한다"


def test_dry_run_touches_neither_model_nor_ledger(wired, monkeypatch, tmp_path):
    monkeypatch.setattr("llm.LocalLLM",
                        lambda *a, **k: pytest.fail("dry-run이 모델을 띄웠다"))
    run(wired, dry_run=True)
    assert not (wired["out"] / "index.html").exists()


def test_gpu_released_even_though_run_succeeds(wired):
    run(wired)
    assert getattr(wired["llm"], "closed", False), "GPU를 반납하지 않았다"


# ── 원장 (AC-3) ──────────────────────────────────────────────
def test_ledger_written_after_render_not_before(wired, monkeypatch, tmp_path):
    """렌더가 터졌는데 원장에 '게시됨'이 남으면 그 논문은 영원히 안 나온다."""
    monkeypatch.setattr(pipeline.render_mod, "render",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("렌더 실패")))
    with pytest.raises(RuntimeError):
        run(wired)
    assert len(PublishedLedger(str(tmp_path / "pub"))) == 0


def test_irrelevant_papers_are_deferred_not_dropped(wired, tmp_path):
    """무관 판정도 버리지 않는다 — 프롬프트를 고치면 회수 대상이다."""
    wired["llm"] = FakeLLM(relevant_every=3)      # 1/3만 관련
    wired["collected"] = papers(30)
    run(wired)
    dl = DeferredLedger(str(tmp_path / "def"))
    assert len(dl) == 20, "무관 20건이 보류에 남아야 한다"
    assert all(r.payload for r in dl.active("2026-08-31", prompt_version="v99")), \
        "payload 없이는 재분류할 수 없다"


def test_quota_leftovers_are_deferred_with_reason(wired, tmp_path):
    wired["collected"] = papers(400)
    run(wired)
    dl = DeferredLedger(str(tmp_path / "def"))
    reasons = {r.reason for r in dl.active("2026-08-31", prompt_version="v99")}
    assert "quota" in reasons


def test_ignore_seen_does_not_write_ledger(wired, tmp_path):
    """재생성 모드가 원장을 오염시키면 다음 날 정상 논문이 사라진다."""
    run(wired, ignore_seen=True)
    assert len(PublishedLedger(str(tmp_path / "pub"))) == 0


# ── 학회 부스트 시점 (D-13) ──────────────────────────────────
def test_venue_boost_applied_after_selection(wired, monkeypatch):
    """선별 전에 부스트하면 축 비율이 깨진다."""
    seen = {}

    real_select = pipeline.selection.select

    def spy(entries, *a, **k):
        seen["boosted_at_select"] = any(e.boosted_stars is not None for e in entries)
        return real_select(entries, *a, **k)

    monkeypatch.setattr(pipeline.selection, "select", spy)
    run(wired)
    assert seen["boosted_at_select"] is False


# ── 실패 처리 ────────────────────────────────────────────────
def test_main_returns_nonzero_on_failure(monkeypatch, capsys):
    """0이 아닌 종료 코드라야 run_daily.sh가 push를 건너뛴다."""
    monkeypatch.setattr(pipeline, "run",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("터짐")))
    assert pipeline.main(["run", "--quiet"]) == 1
    assert "터짐" in capsys.readouterr().err


def test_main_run_succeeds(wired, monkeypatch):
    assert pipeline.main(["run", "--quiet", "--date", "2026-08-31",
                          "--ignore-seen", "--out-dir", str(wired["out"])]) == 0


# ── 테스트가 실제 저장소를 건드리지 않는다 ────────────────────
def test_run_writes_no_files_into_the_repo(wired, tmp_path):
    """실제로 겪은 사고: 테스트가 만든 가짜 항목 16건이 data/entries/에 남아
    render --all 때 진짜 아카이브 페이지(44건)를 밀어냈다.

    임시 경로 밖으로 새는 쓰기가 하나라도 있으면 여기서 걸린다.
    """
    import pathlib
    real = pathlib.Path("data/entries")
    before = {p.name for p in real.iterdir()} if real.is_dir() else set()
    run(wired)
    after = {p.name for p in real.iterdir()} if real.is_dir() else set()
    assert after == before, f"저장소에 파일이 생겼다: {after - before}"
    assert (tmp_path / "entries").is_dir(), "캐시가 임시 경로에 만들어져야 한다"
