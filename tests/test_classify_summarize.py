"""추론 계층의 순수 로직 — 모델 없이 돈다.

LLM 호출 자체는 검증하지 않는다. 검증하는 것은 '모델이 뭘 뱉든 파이프라인이
무너지지 않는다'는 성질이다.
"""
import config
import pytest
from classify import SCHEMA as C_SCHEMA, build_prompt as c_prompt, parse
from models import Paper
from summarize import SCHEMA as S_SCHEMA, build_prompt as s_prompt, hangul_ratio, validate


def paper(**kw):
    return Paper(title=kw.pop("title", "A Test Paper"), source="arxiv", **kw)


# ── 분류 ─────────────────────────────────────────────────────
def test_prompt_lists_every_category():
    """분류체계를 프롬프트에 통째로 싣는다 — 항목이 늘면 자동 반영돼야 한다."""
    p = c_prompt(paper(abstract="x"))
    for c in config.CATEGORIES:
        assert c.id in p, c.id
    assert len(config.CATEGORY_IDS) == 17


def test_schema_enums_match_config():
    """스키마 enum과 설정이 어긋나면 모델이 없는 카테고리를 고른다."""
    assert C_SCHEMA["properties"]["category_id"]["enum"] == list(config.CATEGORY_IDS)
    assert C_SCHEMA["properties"]["axis"]["enum"] == list(config.AXIS_KEYS)


def test_missing_abstract_says_so_instead_of_pretending():
    p = c_prompt(paper(abstract=None))
    assert "초록 미확보" in p


def test_unparseable_becomes_undecided_not_dropped():
    """판정 불가는 버리지 않는다 — 놓치는 것이 잡음보다 비싸다 (원칙 1)."""
    c = parse(None)
    assert c.is_relevant and c.undecided and c.stars is None


def test_category_wins_when_axis_disagrees():
    """축과 카테고리가 어긋나면 enum으로 강제된 카테고리를 믿는다."""
    c = parse({"is_relevant": True, "axis": "medical_imaging",
               "category_id": "surgical_vlp", "stars": 4, "rationale": "x"})
    assert c.axis == "surgical_video"


def test_out_of_range_stars_become_none_not_clamped():
    """별점을 조용히 보정하면 근거 없는 값이 화면에 나간다. 미정으로 둔다."""
    for bad in (0, 9, "4", None, 3.5):
        assert parse({"is_relevant": True, "category_id": "surgical_vlp",
                      "stars": bad, "rationale": ""}).stars is None


def test_unknown_category_and_axis_falls_back_to_undecided():
    c = parse({"is_relevant": True, "axis": "sports", "category_id": "nope",
               "stars": 3, "rationale": ""})
    assert c.undecided


# ── 요약 ─────────────────────────────────────────────────────
def test_prompt_has_no_domain_example_to_leak():
    """예시의 도메인 표현이 무관한 논문 요약으로 새어나간 적이 있다."""
    p = s_prompt(paper(abstract="x"))
    for leaked in ("PeskaVLP", "dynamic time warping", "zero-shot 수술 단계 인식"):
        assert leaked not in p, leaked


def test_summary_schema_puts_tags_first():
    """guided decoding은 스키마 순서대로 생성한다. 절삭에 강한 순서를 지킨다."""
    assert list(S_SCHEMA["properties"]) == ["tags", "korean_summary"]


@pytest.mark.parametrize("text,expected", [
    ("수술 비디오를 다룬다.", 0.8),
    ("This paper proposes a method.", 0.0),
])
def test_hangul_ratio_catches_english_output(text, expected):
    assert (hangul_ratio(text) > 0.5) == (expected > 0.5)


def test_english_summary_rejected():
    """'ASCII만은 아님' 검사는 무력하다 — 마침표 하나로 통과한다."""
    eng = ("This paper proposes a novel framework for surgical video understanding "
           "using hierarchical knowledge augmentation and procedure-aware losses.")
    assert validate({"korean_summary": eng, "tags": ["a"]}) is None


def test_too_short_rejected_and_falls_back_to_title():
    assert validate({"korean_summary": "짧다.", "tags": ["a"]}) is None
    assert validate(None) is None


def test_tags_normalised():
    s = validate({"korean_summary": "수술 비디오 이해를 위한 사전학습에서 내레이션 노이즈 문제를 다룬다. "
           "지식을 계층적으로 증강하고 대조학습으로 문장-클립을 정렬한다. "
           "공개 벤치마크에서 기존 방법을 상회했다.",
                  "tags": ["Surgical Video", "  VLP  ", ""]})
    assert s.tags == ["surgical-video", "vlp"]


def test_tags_capped():
    s = validate({"korean_summary": "수술 비디오 이해를 위한 사전학습에서 내레이션 노이즈 문제를 다룬다. "
           "지식을 계층적으로 증강하고 대조학습으로 문장-클립을 정렬한다. "
           "공개 벤치마크에서 기존 방법을 상회했다.",
                  "tags": [f"t{i}" for i in range(20)]})
    assert len(s.tags) == config.SUMMARY_TAGS_MAX
