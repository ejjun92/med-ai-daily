"""설정 정합성 — 축·카테고리·목록이 서로 어긋나지 않는지."""
from collections import Counter
import config


def test_axis_ratios_sum_to_one():
    assert abs(sum(a.ratio for a in config.AXES) - 1.0) < 1e-9


def test_seventeen_categories_unique_ids():
    assert len(config.CATEGORIES) == 17
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
