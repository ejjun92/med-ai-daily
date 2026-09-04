"""쿼터 배분 — AC-3·AC-4의 로직 보증. 모델 없이 전량 검증한다."""
import random

import pytest

import config
from models import Classification, Entry, Paper
from selection import _apportion, select


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


def full_pool(n: int = 200) -> list[Entry]:
    """모든 축을 넉넉히 채운 풀.

    축 목록을 설정에서 가져온다 — 하드코딩하면 축을 하나 추가할 때마다
    이 파일이 조용히 의미를 잃는다 (실제로 EHR 축 추가 때 깨졌다).
    """
    return pool(**{a.key: n for a in config.AXES})


def test_normal_allocation_respects_axis_targets():
    r = select(full_pool(40))
    counts = r.counts_by_axis()
    for a in config.AXES:
        assert counts[a.key] == a.target, f"{a.key}: {counts[a.key]} != {a.target}"
    assert config.DAILY_MIN <= len(r.entries) <= config.DAILY_MAX


def test_one_axis_exhausted_reports_shortfall():
    r = select(pool(**{a.key: 40 for a in config.AXES}) if False else pool(brain_decoding=40, surgical_video=40,
                    dl_methodology=40, medical_imaging=2))
    assert "medical_imaging" in r.shortfall_by_axis
    assert r.counts_by_axis()["medical_imaging"] == 2


def test_cross_axis_redistribution_fills_up_to_minimum():
    """②-a 경로: 최대치 안에서 하한을 채울 수 있으면 최대치를 넘지 않는다.

    ⚠️ "최대치를 넘지 않는다"는 select()의 전역 불변식이 **아니다**. 이 풀은
    ②-a 여유(8칸)가 부족분(4칸)보다 커서 ②-b가 발동하지 않는 형상이다.
    전역 불변식은 axis.ceiling이며 그쪽은 아래 유휴 칸 재배분 테스트가 지킨다.
    """
    r = select(pool(brain_decoding=60, surgical_video=60,
                    dl_methodology=60, medical_imaging=0))
    assert len(r.entries) == config.DAILY_MIN
    assert not r.overflow_by_axis, "이 형상은 ②-b를 타지 않아야 한다"
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
    """공급이 충분하면 축 목표치의 합에서 멈춘다.

    DAILY_MAX까지 채우지 않는 이유: 남는 자리를 어느 축에 줄지 정해야 하는데,
    그 순간 설정한 비율이 깨진다. 비율 유지가 더 중요하다.
    """
    r = select(full_pool())
    assert len(r.entries) == sum(a.target for a in config.AXES)
    assert config.DAILY_MIN <= len(r.entries) <= config.DAILY_MAX
    assert not r.truncated_ids          # 목표치 합 < DAILY_MAX 라 절삭은 안 일어난다


def test_truncation_fires_when_max_forced_below_targets():
    """절삭 분기는 안전망이다 — 축 목표치 합이 DAILY_MAX보다 작아
    평상시엔 도달하지 않는다. 상한을 낮춰 강제로 태워 동작을 고정한다."""
    r = select(full_pool(), daily_min=10, daily_max=20)
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


# ── 유휴 칸 재배분 (실측 2026-09-04) ────────────────────────────
def _starved_pool():
    """09-04 실제 모양: 두 축은 공급이 마르고 나머지는 넘친다.

    그날 관련 판정 298건 중 35건만 실리고 263건이 quota로 밀렸다.
    brain 6/18·surgical 3/16으로 25칸이 유휴였는데, 재배분이 각 축
    최대치에 막혀 그 칸이 죽었다.
    """
    return pool(brain_decoding=6, surgical_video=3,
                dl_methodology=100, ehr_clinical=100, medical_imaging=100)


def test_idle_capacity_moves_to_axes_that_still_have_supply():
    r = select(_starved_pool())
    counts = r.counts_by_axis()
    assert counts["brain_decoding"] == 6 and counts["surgical_video"] == 3
    assert len(r.entries) >= config.DAILY_MIN, (
        f"공급이 남아 있는데 {len(r.entries)}건에서 멈췄다 — 유휴 칸이 죽었다")
    assert any(counts[a.key] > a.bounds[1] for a in config.AXES), (
        "유휴 칸을 넘겨받았다면 어느 축은 제 최대치를 넘어야 한다")


def test_idle_redistribution_stops_at_daily_min():
    """유휴 칸을 넘긴다고 상한까지 밀어붙이지는 않는다 — 하한에서 멈춘다."""
    r = select(_starved_pool())
    assert len(r.entries) == config.DAILY_MIN


def test_apportion_splits_by_ratio_not_round_robin():
    """배분은 비중 비례다. 라운드로빈이면 13% 축과 17% 축이 같은 수를 받는다.

    부족분이 작으면 두 방식의 결과가 우연히 일치한다(5칸이면 둘 다 2/2/1).
    라운드로빈을 실제로 배제하려면 갈라지는 크기를 써야 한다 — 20칸에서
    비중배분은 8/6/6, 라운드로빈은 7/7/6이다.
    """
    recv = ("dl_methodology", "ehr_clinical", "medical_imaging")
    alloc = _apportion(20, {k: 100 for k in recv})
    assert sum(alloc.values()) == 20
    assert alloc["dl_methodology"] > alloc["ehr_clinical"], (
        f"비중 0.17이 0.14보다 많이 받아야 한다: {alloc}")
    ratios = {a.key: a.ratio for a in config.AXES}
    for hi, lo_ in zip(recv, recv[1:]):
        assert (alloc[hi] >= alloc[lo_]) == (ratios[hi] >= ratios[lo_])


def test_apportion_sum_is_exact_and_respects_capacity():
    """합은 정확히 min(부족분, 총 capacity). 공급 이상으로 배정하지 않는다."""
    cap = {"dl_methodology": 2, "ehr_clinical": 1, "medical_imaging": 0}
    alloc = _apportion(20, cap)
    assert sum(alloc.values()) == 3
    for k, v in alloc.items():
        assert v <= cap[k], f"{k}: {v} > capacity {cap[k]}"


def test_apportion_is_independent_of_dict_order():
    """입력 dict 순서가 결과를 바꾸면 안 된다 (원칙 4)."""
    keys = [a.key for a in config.AXES]
    base = _apportion(17, {k: 50 for k in keys})
    for rot in range(1, len(keys)):
        rotated = {k: 50 for k in keys[rot:] + keys[:rot]}
        assert _apportion(17, rotated) == base


def test_idle_redistribution_never_pads_with_irrelevant():
    """자리가 남아도 관련 없는 항목은 절대 올리지 않는다 (D-2)."""
    entries = pool(brain_decoding=6, surgical_video=3, dl_methodology=10)
    entries += [Entry(paper=Paper(title=f"irrelevant{i}", source="arxiv",
                                  arxiv_id=f"28{i:06d}"),
                      classification=Classification(is_relevant=False))
                for i in range(80)]
    r = select(entries)
    assert len(r.entries) == 19          # 있는 관련 논문 전부, 그 이상은 없다
    assert all(e.classification.is_relevant for e in r.entries)


def test_idle_redistribution_is_deterministic():
    base = _starved_pool()
    first = [e.paper.primary_id for e in select(base).entries]
    for seed in (1, 7, 42):
        shuffled = base[:]
        random.Random(seed).shuffle(shuffled)
        assert [e.paper.primary_id for e in select(shuffled).entries] == first, \
            f"seed={seed}에서 출력이 달라졌다"


def test_single_axis_supply_cannot_take_over_the_page():
    """공급이 한 축뿐인 날에도 그 축이 페이지를 통째로 가져가지 못한다.

    실측: 천장이 없을 때 최대치 8인 medical_imaging이 40칸을 전부 먹었다.
    최대치를 "완화"하는 것과 "제거"하는 것은 다르다.
    """
    for a in config.AXES:
        r = select(pool(**{a.key: 200}))
        got = r.counts_by_axis()[a.key]
        assert got <= a.ceiling, f"{a.key}: {got} > 천장 {a.ceiling}"
        assert got < config.DAILY_MIN, f"{a.key}가 페이지를 독점했다 ({got}건)"


def test_overflow_is_reported_not_silent():
    """최대치를 넘겨받았으면 그 사실이 결과에 남아야 한다."""
    r = select(_starved_pool())
    assert r.overflow_by_axis, "재배분이 일어났는데 아무 기록도 없다"
    counts = r.counts_by_axis()
    for key, over in r.overflow_by_axis.items():
        assert over == counts[key] - config.AXIS_BY_KEY[key].bounds[1]
