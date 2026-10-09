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


# ── 예산 계산 (실측 2026-10-07·08: 이틀 연속 발행 0건) ──────────
def test_utilization_is_derived_from_free_not_total():
    """vLLM의 gpu_memory_utilization은 전체 VRAM 기준이다.

    48GB 장에서 0.85는 41.8GB를 요구한다. 다른 작업이 22GB를 쓰는 날
    게이트(24GB)는 통과하는데 vLLM이 KV 캐시를 못 잡고 죽었다 —
    "ValueError: No available memory for the cache blocks". 그 이틀은
    사이트가 갱신되지 않았다.
    """
    total = 49_140
    tight = llm.gpu_utilization_for(27_135, total)      # 실측 그 날의 여유
    assert tight < config.GPU_MEMORY_UTILIZATION, "여유를 반영하지 않았다"
    assert tight * total <= 27_135, (
        f"여유 27,135MB보다 많은 {tight * total:,.0f}MB를 요구한다")


def test_utilization_keeps_headroom_below_free():
    """꽉 채우면 조회와 할당 사이에 다른 작업이 조금 잡아도 OOM이 난다."""
    total = 49_140
    for free in (24_000, 30_000, 40_000):
        got = llm.gpu_utilization_for(free, total) * total
        assert got < free, f"여유 {free}를 그대로 다 쓴다 ({got:.0f})"


def test_utilization_on_empty_device_matches_the_global_cap():
    """장이 통째로 비면 종전과 같은 예산을 쓴다 — 느려지면 안 된다."""
    total = 49_140
    assert llm.gpu_utilization_for(total, total) == config.GPU_MEMORY_UTILIZATION


def test_utilization_survives_unreadable_total():
    """nvidia-smi가 전체 VRAM을 못 주면 종전 상수로 되돌아간다."""
    assert llm.gpu_utilization_for(0, 0) == config.GPU_MEMORY_UTILIZATION


def test_gate_and_utilization_are_consistent():
    """게이트를 통과한 장은 실제로 **모델이 떠야** 한다.

    게이트는 여유만 보고 통과시키는데 util은 그 여유에서 계산되므로, 둘이
    어긋나면 두 방향 모두로 깨진다 — 게이트가 낮으면 KV 캐시가 말라 사실상
    순차 처리가 되고(10-07·08은 아예 크래시), 높으면 아무 장도 못 골라
    매일 건너뛴다(실측 2026-10-09: 게이트 30,700에서 pick_gpu()가 None).
    유도식을 복제하지 않고 성질만 본다.
    """
    TOTAL = 49_140          # RTX A6000
    budget = llm.gpu_utilization_for(config.GPU_FREE_VRAM_REQUIRED_MB, TOTAL) * TOTAL
    seqs = (budget - config.GPU_WEIGHTS_MB) / config._kv_mb_per_seq()
    assert seqs >= config.MIN_CONCURRENT_SEQS * 0.95, (
        f"게이트 {config.GPU_FREE_VRAM_REQUIRED_MB:,}MB에서 동시 시퀀스가 "
        f"{seqs:.1f}개 — 하한 {config.MIN_CONCURRENT_SEQS}개에 못 미친다")


def test_gate_fits_this_server_under_normal_occupancy():
    """게이트가 실제 점유 상황에서 장을 하나도 못 고르면 안 된다.

    실측 2026-10-09: 이 서버는 4-way 작업이 장당 ~21.5GB를 잡아 네 장 모두
    여유가 27GB 선에 고정된다. 게이트를 30,700으로 올렸더니 매일 100%
    건너뛰는 상태가 됐다 — 느린 성공을 영구 미발행과 바꾼 셈이었다.
    """
    OCCUPIED_FREE_MB = 27_000      # 그 작업이 도는 동안의 장별 여유
    assert config.GPU_FREE_VRAM_REQUIRED_MB <= OCCUPIED_FREE_MB, (
        f"게이트 {config.GPU_FREE_VRAM_REQUIRED_MB:,}MB가 평시 여유 "
        f"{OCCUPIED_FREE_MB:,}MB를 넘어 아무 장도 못 고른다")


def test_engine_receives_the_derived_utilization_not_the_constant(monkeypatch):
    """계산한 util이 실제로 vLLM에 전달되는지 본다.

    순수 함수만 검증하면 배선을 되돌려도(gpu_memory_utilization=
    config.GPU_MEMORY_UTILIZATION) 전부 통과한다 — timeout 회귀 보호에
    적용한 기준을 여기에도 적용한다.
    """
    import sys
    import types

    seen = {}

    class FakeLLM:
        def __init__(self, **kw):
            seen.update(kw)

    fake = types.ModuleType("vllm")
    fake.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake)
    # 장이 반만 빈 상황 — 상수를 쓰면 없는 메모리를 요구한다
    FREE, TOTAL = 27_135, 49_140
    monkeypatch.setattr(llm, "pick_gpu", lambda log=print: "0")
    monkeypatch.setattr(llm, "gpu_memory", lambda: {"0": (FREE, TOTAL)})

    inst = llm.LocalLLM(log=lambda *_: None)
    inst.__enter__()

    expected = llm.gpu_utilization_for(FREE, TOTAL)
    assert seen["gpu_memory_utilization"] == expected, (
        f"엔진이 {seen['gpu_memory_utilization']}을 받았다 (기대 {expected})")
    assert seen["gpu_memory_utilization"] < config.GPU_MEMORY_UTILIZATION, (
        "상수를 그대로 넘겼다 — 10-07·08 크래시가 재현된다")


def test_unreadable_total_vram_is_logged_not_silent(monkeypatch):
    """조회 실패로 상수 폴백할 때 흔적을 남겨야 한다.

    안 남기면 '여유는 충분해 보이는데 KV 캐시 할당에서 죽는다'는 형태가
    되고 로그에 단서가 0이다 (10-07·08이 그 형태였다).
    """
    import sys
    import types

    fake = types.ModuleType("vllm")
    fake.LLM = lambda **kw: None
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setattr(llm, "pick_gpu", lambda log=print: "0")
    monkeypatch.setattr(llm, "gpu_memory", lambda: None)   # 조회 실패

    lines = []
    llm.LocalLLM(log=lines.append).__enter__()
    assert any("전체 VRAM 조회 실패" in l for l in lines), lines
