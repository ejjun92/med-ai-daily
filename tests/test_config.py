"""설정 정합성 — 축·카테고리·목록이 서로 어긋나지 않는지."""
from collections import Counter
import config


def test_axis_ratios_sum_to_one():
    assert abs(sum(a.ratio for a in config.AXES) - 1.0) < 1e-9


def test_category_count_and_unique_ids():
    # 개수를 못 박는 이유: 카테고리를 늘리면 분류 프롬프트가 길어지고 enum이
    # 바뀐다. 의도적 변경이면 이 숫자도 함께 고치라는 뜻이다.
    assert len(config.CATEGORIES) == 22
    assert len(config.CATEGORY_IDS) == len(set(config.CATEGORY_IDS))


def test_every_category_belongs_to_exactly_one_known_axis():
    for c in config.CATEGORIES:
        assert c.axis in config.AXIS_KEYS, f"{c.id}의 축 {c.axis}가 정의에 없다"


def test_every_axis_has_at_least_one_category():
    counts = Counter(c.axis for c in config.CATEGORIES)
    for a in config.AXES:
        assert counts[a.key] >= 1, f"{a.key}에 카테고리가 없다"


def test_axis_targets_sum_near_daily_target():
    # 반올림 오차를 감안해 ±2
    assert abs(sum(a.target for a in config.AXES) - config.DAILY_TARGET) <= 2


def test_venue_boost_list_disjoint_from_pubmed_whitelist():
    """같은 목록이면 PubMed 항목 100%가 무조건 부스트되어 축이 재가중된다."""
    assert not (set(config.PUBMED_JOURNALS) & set(config.VENUE_BOOST_LIST))


def test_source_caps_positive():
    for name in ("ARXIV_MAX", "PUBMED_MAX", "S2_MAX", "DEFERRED_DAILY_MAX"):
        assert getattr(config, name) > 0


def test_neurips_and_icml_include_full_names():
    """골든셋 실측: 두 논문 모두 comments에 약칭 없이 정식 명칭만 썼다."""
    assert any("Neural Information" in p for p in config.VENUE_BOOST_LIST["NeurIPS"])
    assert any("Machine Learning" in p for p in config.VENUE_BOOST_LIST["ICML"])


def test_overflow_factor_actually_lifts_the_ceiling():
    """이 값이 DAILY_MAX/DAILY_MIN 미만이면 유휴 칸 재배분이 통째로 죽는다.

    ceiling은 max(bounds[1], round(DAILY_MIN * ratio * FACTOR))이므로,
    FACTOR가 작으면 max()가 ceiling을 bounds[1]로 눌러 ②-b가 아무것도 더
    넣지 못한다 — 예외도 경고도 없이 조용히 무력화된다. 이 파일이 그걸 막는다.
    """
    assert config.AXIS_OVERFLOW_FACTOR >= config.DAILY_MAX / config.DAILY_MIN
    for a in config.AXES:
        raw = round(config.DAILY_MIN * a.ratio * config.AXIS_OVERFLOW_FACTOR)
        assert raw >= a.bounds[1], (
            f"{a.key}: 천장 원값 {raw} < 최대치 {a.bounds[1]} — FACTOR가 너무 낮다")
        assert a.ceiling > a.bounds[1], f"{a.key}: 천장이 최대치와 같아 재배분 불가"
