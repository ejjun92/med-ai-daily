"""GPU 선택 — 어느 장을 잡고, 언제 물러나는가.

연구용 GPU를 밀어내지 않는 것이 이 코드의 목적이다 (원칙 5).
빈 장이 있으면 쓰고, 다 차 있으면 기다렸다가 건너뛴다.
"""
import config
import llm
import pytest


@pytest.fixture
def free(monkeypatch):
    """장별 여유 VRAM을 주입한다."""
    state = {}

    def setter(**mb):
        state.clear()
        state.update({k.lstrip("g"): v for k, v in mb.items()})
        monkeypatch.setattr(llm, "gpu_free_mb", lambda: dict(state) or None)

    setter(g0=48000, g1=48000, g2=48000, g3=48000)
    return setter


BUSY = 1000
IDLE = 48000


def test_prefers_gpu_zero_when_free(free):
    assert llm.pick_gpu(log=lambda *_: None) == "0"


def test_falls_through_to_next_free_card(free):
    """0번이 점유돼 있으면 1번으로 넘어간다."""
    free(g0=BUSY, g1=IDLE, g2=IDLE, g3=IDLE)
    assert llm.pick_gpu(log=lambda *_: None) == "1"


def test_skips_every_busy_card_in_order(free):
    free(g0=BUSY, g1=BUSY, g2=BUSY, g3=IDLE)
    assert llm.pick_gpu(log=lambda *_: None) == "3"


def test_returns_none_when_all_busy(free):
    """다 차 있으면 남의 작업을 밀어내지 않고 물러난다."""
    free(g0=BUSY, g1=BUSY, g2=BUSY, g3=BUSY)
    assert llm.pick_gpu(log=lambda *_: None) is None


def test_threshold_is_config_driven(free):
    """요구치를 아슬아슬하게 만족하는 장도 쓴다."""
    need = config.GPU_FREE_VRAM_REQUIRED_MB
    free(g0=need - 1, g1=need, g2=BUSY, g3=BUSY)
    assert llm.pick_gpu(log=lambda *_: None) == "1"


def test_selection_is_deterministic_not_greediest(free):
    """가장 여유가 큰 장이 아니라 선호 순서를 따른다.

    순서를 고정해야 같은 상황에서 늘 같은 장을 잡아, 어느 장이 이 작업용인지
    예측할 수 있다.
    """
    free(g0=IDLE, g1=IDLE * 2, g2=IDLE * 3, g3=IDLE * 4)
    assert llm.pick_gpu(log=lambda *_: None) == "0"


def test_proceeds_when_nvidia_smi_unavailable(monkeypatch):
    """조회가 안 된다고 멈추면 GPU가 멀쩡해도 매일 건너뛴다."""
    monkeypatch.setattr(llm, "gpu_free_mb", lambda: None)
    assert llm.pick_gpu(log=lambda *_: None) == config.GPU_DEVICES[0]


def test_unknown_card_counts_as_busy(free):
    """nvidia-smi가 일부 장만 보고해도 없는 장을 잡으려 들지 않는다."""
    free(g0=BUSY)
    assert llm.pick_gpu(log=lambda *_: None) is None


# ── 대기 후 건너뛰기 ─────────────────────────────────────────
def test_waits_then_gives_up(monkeypatch):
    slept = []
    monkeypatch.setattr(llm.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(llm, "pick_gpu", lambda log=print: None)
    engine = llm.LocalLLM(log=lambda *_: None)
    assert engine._wait_for_gpu() is None
    assert len(slept) == config.GPU_WAIT_RETRIES, "설정한 횟수만큼 기다려야 한다"


def test_takes_card_as_soon_as_one_frees_up(monkeypatch):
    calls = {"n": 0}

    def flaky(log=print):
        calls["n"] += 1
        return "2" if calls["n"] > 1 else None

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(llm, "pick_gpu", flaky)
    monkeypatch.setattr(llm, "gpu_free_mb", lambda: {"2": 48000})
    engine = llm.LocalLLM(log=lambda *_: None)
    assert engine._wait_for_gpu() == "2"
