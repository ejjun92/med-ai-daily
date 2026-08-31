"""쿼터 배분 — AC-3·AC-4의 로직 보증. 모델 없이 전량 검증한다."""
import random

import pytest

import config
from models import Classification, Entry, Paper
from selection import select


def mk(axis: str, stars: int, idx: int, date_: str = "2026-08-30") -> Entry:
    return Entry(
        paper=Paper(title=f"{axis}-{idx}", source="arxiv",
                    arxiv_id=f"26{idx:06d}", announced_date=date_),
        classification=Classification(is_relevant=True, axis=axis,
                                      category_id=None, stars=stars),
    )


def pool(**per_axis: int) -> list[Entry]:
    out, i = [], 0
    for axis, n in per_axis.items():
        for k in range(n):
            i += 1
            out.append(mk(axis, stars=5 - (k % 5), idx=i))
    return out


def test_normal_allocation_respects_axis_targets():
    r = select(pool(brain_decoding=40, surgical_video=40,
                    dl_methodology=40, medical_imaging=40))
    counts = r.counts_by_axis()
    for a in config.AXES:
        assert counts[a.key] == a.target, f"{a.key}: {counts[a.key]} != {a.target}"
    assert config.DAILY_MIN <= len(r.entries) <= config.DAILY_MAX


def test_one_axis_exhausted_reports_shortfall():
    r = select(pool(brain_decoding=40, surgical_video=40,
                    dl_methodology=40, medical_imaging=2))
    assert "medical_imaging" in r.shortfall_by_axis
    assert r.counts_by_axis()["medical_imaging"] == 2


def test_cross_axis_redistribution_fills_up_to_minimum():
    """한 축이 고갈되면 다른 축 잉여로 채우되 각 축 최대치는 넘지 않는다."""
    r = select(pool(brain_decoding=60, surgical_video=60,
                    dl_methodology=60, medical_imaging=0))
    assert len(r.entries) >= config.DAILY_MIN
    counts = r.counts_by_axis()
    for a in config.AXES:
        assert counts[a.key] <= a.bounds[1], f"{a.key}가 최대치를 넘겼다"


def test_global_shortage_publishes_what_exists_without_padding():
    """부족하면 적게 싣는다. 관련 없는 항목으로 채우지 않는다 (D-2)."""
    entries = pool(brain_decoding=5, surgical_video=5)
    entries += [Entry(paper=Paper(title=f"irrelevant{i}", source="arxiv",
                                  arxiv_id=f"27{i:06d}"),
                      classification=Classification(is_relevant=False))
                for i in range(50)]
    r = select(entries)
    assert len(r.entries) == 10
    assert all(e.classification.is_relevant for e in r.entries)


def test_oversupply_settles_at_axis_targets():
    """공급이 충분하면 축 목표치의 합(=51)에서 멈춘다.

    60까지 채우지 않는 이유: 남는 9자리를 어느 축에 줄지 정해야 하는데,
    그 순간 35/30/20/15 비율이 깨진다. 비율 유지가 더 중요하다.
    """
    r = select(pool(brain_decoding=200, surgical_video=200,
                    dl_methodology=200, medical_imaging=200))
    assert len(r.entries) == sum(a.target for a in config.AXES)
    assert config.DAILY_MIN <= len(r.entries) <= config.DAILY_MAX
    assert not r.truncated_ids          # 목표치 합 < DAILY_MAX 라 절삭은 안 일어난다


def test_truncation_fires_when_max_forced_below_targets():
    """절삭 분기는 안전망이다 — 축 목표치 합이 DAILY_MAX보다 작아
    평상시엔 도달하지 않는다. 상한을 낮춰 강제로 태워 동작을 고정한다."""
    r = select(pool(brain_decoding=200, surgical_video=200,
                    dl_methodology=200, medical_imaging=200),
               daily_min=10, daily_max=20)
    assert len(r.entries) == 20
    assert len(r.truncated_ids) > 0


def test_deterministic_under_input_shuffling():
    """배치 결과 도착 순서가 임의여도 같은 페이지가 나와야 한다 (원칙 4).

    별점만으로 정렬하면 동점이 입력 순서로 결정되고, stable sort라
    수집 순서가 바뀌면 다른 60건이 실린다.
    """
    base = pool(brain_decoding=40, surgical_video=40,
                dl_methodology=40, medical_imaging=40)
    first = [e.paper.primary_id for e in select(base).entries]
    for seed in (1, 7, 42):
        shuffled = base[:]
        random.Random(seed).shuffle(shuffled)
        got = [e.paper.primary_id for e in select(shuffled).entries]
        assert got == first, f"seed={seed}에서 출력이 달라졌다"


def test_ties_broken_by_date_then_id():
    """같은 별점이면 최신 발표일 우선, 그다음 id 오름차순."""
    a = mk("brain_decoding", 5, 1, "2026-08-01")
    b = mk("brain_decoding", 5, 2, "2026-08-30")
    r = select([a, b])
    assert r.entries[0].paper.announced_date == "2026-08-30"


def test_higher_stars_come_first():
    lo = mk("brain_decoding", 2, 1)
    hi = mk("brain_decoding", 5, 2)
    r = select([lo, hi])
    assert r.entries[0].classification.stars == 5


def test_unrelated_entries_land_in_deferred():
    entries = pool(brain_decoding=100)
    r = select(entries)
    assert len(r.deferred_ids) == len(entries) - len(r.entries)
