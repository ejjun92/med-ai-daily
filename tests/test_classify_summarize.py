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
    assert len(config.CATEGORY_IDS) == 18


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


# ── 폭주 잔재 걸러내기 (미리보기 육안 검토에서 발견) ────────
_BASE = ("뇌 신호에서 시각 정보를 추출하는 문제를 다룬다. "
         "뇌의 계층적 구조를 모방한 정렬 프레임워크를 사용해 EEG에서 특징을 뽑는다. ")


@pytest.mark.parametrize("tail,ok", [
    ("공개 벤치마크에서 기존 방법을 상회했다.", True),
    ("CLIP과 transformer 기반 self-supervised 학습을 쓴다.", True),
    ("이 접근법은 뇌 기능의ERRUuser_MetaData ошибки를.", False),      # 키릴
    ("请确保您的提供的摘要仅基于此论文的摘要内容.", False),            # 한자
    ("fMRIuser 💬user 내용이 이어진다.", False),                     # 역할 마커
    ("<|im_start|>assistant 이어서 설명한다.", False),
])
def test_degenerate_tails_rejected_but_technical_terms_kept(tail, ok):
    """길이·한글비율·문장부호를 다 통과하고도 화면에 나간 형태들이다.

    영문 전문용어는 그대로 살려야 하므로 라틴 문자는 건드리지 않는다.
    """
    got = validate({"korean_summary": _BASE + tail, "tags": ["a"]})
    assert (got is not None) is ok


# ── 축 경계: 과잉포섭 방지 (2026-08-31 사용자 지적으로 발견) ──
def test_prompt_states_what_brain_decoding_is_not():
    """v1은 "뇌 신호를 다루면 brain_decoding"이라고만 썼다.

    그 결과 뇌 연결성·질환분류·감정인식이 전부 끌려 들어와, 실제 페이지의
    brain_decoding 18건 중 진짜 디코딩 논문은 1건이었다. 골든셋 20편이 전부
    '진짜'라서 재현율 20/20으로도 이 실패가 보이지 않았다.

    프롬프트가 경계를 명시하는지 확인한다 — 정의만 주면 모델은 넓게 잡는다.
    """
    p = c_prompt(paper(abstract="x"))
    assert "뇌를 다룬다는 것만으로는 부족하다" in p
    for excluded in ("질환 진단", "harmonization", "환자군을 분류"):
        assert excluded in p, f"제외 대상 '{excluded}'가 프롬프트에 없다"
    for included in ("BCI", "foundation model", "표현학습"):
        assert included in p, f"포함 대상 '{included}'가 프롬프트에 없다"


def test_negative_samples_exist_in_golden_set():
    """재현율만 재는 표본은 과잉포섭을 잡지 못한다.

    이 파일이 검사하는 것은 표본의 존재다. 실제 판정은 GPU가 필요하므로
    tests/test_golden_recall.py(@pytest.mark.llm)가 본다.
    """
    import json
    import pathlib
    g = json.loads((pathlib.Path(__file__).parent.parent
                    / "data" / "golden_set.json").read_text())
    negs = g.get("negatives", [])
    assert len(negs) >= 8, "brain_decoding 음성 표본이 부족하다"
    assert all(n["must_not_be_axis"] == "brain_decoding" for n in negs)
    assert all(n.get("why") for n in negs), "왜 아닌지 남겨야 다음 사람이 판단할 수 있다"
    # 경계를 넓혔으면 '새로 포함돼야 할 것'도 표본으로 남긴다 — 안 그러면
    # 다음 사람이 다시 좁히면서 조용히 되돌린다.
    pos = g.get("positives_broadened", [])
    assert len(pos) >= 3 and all(x["must_be_axis"] == "brain_decoding" for x in pos)


def test_prompt_version_bumped_when_rules_change():
    """프롬프트를 고치면 버전을 올려야 보류분이 재분류된다 (원장 버전 게이트).

    안 올리면 v1으로 잘못 걸러진 논문들이 영원히 회수되지 않는다.
    """
    assert config.CLASSIFY_PROMPT_VERSION not in ("v1", "v2"), \
        "brain_decoding 규칙을 고쳤으면 CLASSIFY_PROMPT_VERSION을 올려야 한다"
