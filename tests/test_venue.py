"""학회 라벨 추출 — 테스트 문자열은 전부 골든셋의 실제 comments다."""
import pytest
import venue
from models import Venue


@pytest.mark.parametrize("comments,ref_year,want_key,want_ws", [
    # 약칭 없이 정식 명칭만 쓴 실제 사례
    ("Accepted at the 38th Conference on Neural Information Processing Systems (NeurIPS 2024)",
     2024, "NeurIPS", False),
    ("Forty-Second International Conference on Machine Learning (ICML 2025)",
     2025, "ICML", False),
    ("Early accepted by MICCAI 2025", 2025, "MICCAI", False),
    ("Accepted to MICCAI 2025", 2025, "MICCAI", False),
    ("ECCV 2024. Project: https://weihaox.github.io/UMBRAE", 2024, "ECCV", False),
    # 워크숍 — 본회의와 수락 기준이 다르다. 오타 'presentationn' 포함
    ("Accepted at CVPRW 2026 (AI4RWC Oral presentationn)", 2026, "CVPR", True),
    ("Accepted to the Thoracic Image Analysis (TIA) Workshop at MICCAI 2026",
     2026, "MICCAI", True),
])
def test_extracts_real_golden_set_comments(comments, ref_year, want_key, want_ws):
    v = venue.extract(comments=comments, ref_year=ref_year)
    assert v is not None, f"추출 실패: {comments[:50]}"
    assert v.key == want_key
    assert v.is_workshop == want_ws


@pytest.mark.parametrize("comments,ref_year", [
    ("Submitted to CVPR 2026", 2026),
    ("Under review at MICCAI 2026", 2026),
    ("Extended version of our CVPR 2024 paper", 2026),
    ("Uses the MICCAI 2018 challenge dataset", 2026),   # 오래된 연도 = 참조
    ("6 pages, 3 tables, 3 figures", 2025),             # 학회 정보 없음
])
def test_rejects_non_acceptance(comments, ref_year):
    assert venue.extract(comments=comments, ref_year=ref_year) is None


@pytest.mark.parametrize("journal,want", [
    ("Med Image Anal", "MedIA"),
    ("IEEE Trans Med Imaging", "IEEE TMI"),
    ("IEEE Trans Pattern Anal Mach Intell", "TPAMI"),
    ("Radiol Artif Intell", "Radiology: AI"),
    ("Int J Comput Assist Radiol Surg", "IJCARS"),
])
def test_matches_nlm_abbreviations(journal, want):
    """PubMed는 NLM 축약형을 반환한다 — of/and/on이 없다."""
    v = venue.extract(journal=journal)
    assert v is not None and v.key == want


def test_boost_caps_at_five():
    v = Venue(name="MICCAI 2025", key="MICCAI", year=2025)
    assert venue.boost(5, v) == 5
    assert venue.boost(4, v) == 5


def test_boost_skips_workshop():
    w = Venue(name="CVPRW 2026", key="CVPR", year=2026, is_workshop=True)
    assert venue.boost(4, w) == 4


def test_boost_noop_without_venue():
    assert venue.boost(3, None) == 3
