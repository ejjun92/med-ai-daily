"""망가진 LLM 출력 복구 회귀 테스트.

이 테스트가 지키는 것: guided decoding + temperature 0 조합에서 모델이 값을 다
쓴 뒤 반복 루프에 빠져도, 이미 생성된 내용은 잃지 않는다. 골든셋 20편을 실제
모델에 돌려 받은 원문을 픽스처로 고정해 두었다 (GPU 없이 돈다).
"""
import json
import pathlib

import pytest

from llm import salvage_json
from summarize import trim_to_sentence, validate

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "degenerate_llm_outputs.json"
CASES = json.loads(FIXTURE.read_text())["cases"]


def _parse(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return salvage_json(raw)


# ── 복구 자체 ────────────────────────────────────────────────
def test_every_degenerate_output_yields_something():
    """20편 모두에서 최소한 태그는 건진다 — 통째로 잃는 경우는 없다."""
    for c in CASES:
        got = _parse(c["raw"])
        assert got is not None, c["title"]
        assert got.get("tags"), c["title"]


def test_summary_survives_in_most_cases():
    """요약 회수율. 실측 기준선을 아래로 못 내려가게 못 박는다.

    17/20을 목표가 아니라 하한으로 둔다 — 복구 로직을 건드리다 조용히
    나빠지는 것이 이 테스트가 막으려는 사고다.
    """
    kept = sum(1 for c in CASES if validate(_parse(c["raw"])) is not None)
    assert kept >= 17, f"요약 회수 {kept}/20 — 기준선 17 미만"


def test_open_string_recovery_beats_comma_cut():
    """문자열이 열린 채 끊긴 경우 태그만 남기고 끝내면 안 된다.

    복구 경로 순서가 뒤집히면 정확히 이 테스트가 깨진다 (실측에서 4편을 잃었다).
    """
    raw = ('{"tags": ["brain-decoding", "fmri"], "korean_summary": '
           '"fMRI 신호에서 시각 자극을 복원하는 문제를 다룬다. 피험자별 인코더와 '
           '공유 확산 디코더를 결합해 학습한다. 세 번째 문장이 여기서 잘')
    got = salvage_json(raw)
    assert got is not None
    assert "korean_summary" in got, "열린 문자열 복구가 절단 경로에 밀렸다"
    assert got["tags"] == ["brain-decoding", "fmri"]


def test_trailing_garbage_after_closed_string():
    """값을 닫은 뒤 잉여물을 뱉는 형태 (실측 절반이 이 형태였다)."""
    raw = ('{"tags": ["a"], "korean_summary": "요약이 여기서 온전히 끝난다. " '
           '"  UsageIdUsageIdUsageId')
    got = salvage_json(raw)
    assert got is not None and got.get("korean_summary")


def test_half_million_chars_of_whitespace():
    """공백 50만 자를 뱉은 실측 사례가 있었다 — 지연이나 예외 없이 처리한다."""
    raw = '{"tags": ["a"], "korean_summary": "요약이다.' + " " * 500_000
    assert salvage_json(raw) is not None


def test_garbage_input_returns_none():
    """복구할 게 없으면 조용히 이상한 값을 만들지 않는다."""
    for bad in ("", "I cannot answer that", "null", "{", '{"tag'):
        assert salvage_json(bad) is None, bad


def test_valid_json_passes_through_unchanged():
    obj = {"tags": ["a", "b"], "korean_summary": "온전한 요약이다."}
    assert salvage_json(json.dumps(obj, ensure_ascii=False)) == obj


# ── 복구 이후 다듬기 ─────────────────────────────────────────
def test_no_recovered_summary_ends_mid_sentence():
    """복구된 요약이 어절 중간에서 끝난 채로 화면에 나가면 안 된다."""
    for c in CASES:
        s = validate(_parse(c["raw"]))
        if s is None:
            continue
        assert s.korean_summary.endswith(("다.", ".", "!", "?")), c["title"]


def test_repetition_garbage_never_reaches_output():
    """폭주 문자열이 요약 본문에 섞여 나가지 않는다."""
    for c in CASES:
        s = validate(_parse(c["raw"]))
        if s is None:
            continue
        assert "UsageId" not in s.korean_summary
        assert "URLException" not in s.korean_summary
        assert "  " not in s.korean_summary, c["title"]
