"""쿼터 배분 — 순수 함수.

⚠️ 파일명이 select.py가 아닌 이유: 파이썬 표준 라이브러리에 `select` 모듈이
   있어 이름이 충돌한다. pytest 등이 stdlib select를 먼저 로드하면 sys.modules에
   그것이 들어가 우리 모듈을 가린다 (증상: "select expected at least 3 arguments").
   계획 문서의 src/select.py 표기는 이 파일로 읽는다.
 LLM 없이 전량 테스트 가능하다 (계획 D-1).

LLM은 논문마다 독립적으로 (관련성, 축, 카테고리, 별점)만 판정하고,
"어느 축에서 몇 개를 뽑을지"는 여기서 결정한다. LLM에게 쿼터를 맡기면
결과가 비결정적이고 테스트가 불가능해진다.
"""
from __future__ import annotations

import config
from models import Entry, SelectionResult


def _sort_key(e: Entry):
    """전순서 정렬 (계획 D-15).

    별점만으로 정렬하면 동점이 입력 순서로 결정된다. 파이썬 sort는 stable
    이므로 같은 입력이라도 수집 순서가 흔들리면 다른 페이지가 나온다 —
    원칙 4(결정론) 위반이자 골든셋 재현율 측정에 지터를 넣는다.
    """
    return (-(e.classification.stars or 0),
            e.paper.sort_date() or "",       # 문자열 역순은 아래 reverse로 처리 못하므로
            e.paper.primary_id)


def _sorted(entries: list[Entry]) -> list[Entry]:
    # stars 내림차순 → 발표일 내림차순 → id 오름차순
    return sorted(entries, key=lambda e: (
        -(e.classification.stars or 0),
        _neg_date(e.paper.sort_date()),
        e.paper.primary_id,
    ))


def _neg_date(d: str) -> str:
    """날짜 문자열을 내림차순 정렬하기 위한 보수 변환."""
    if not d:
        return "~"          # 날짜 없는 항목은 뒤로 ('~'는 숫자·하이픈보다 큼)
    return "".join(chr(ord("9") - int(c)) if c.isdigit() else c for c in d)


def select(entries: list[Entry],
           daily_min: int | None = None,
           daily_max: int | None = None) -> SelectionResult:
    """축별 배분 → 축 간 재배분 → 상한 절삭.

    "보충"은 `is_relevant: true`이면서 다른 축 쿼터에 밀린 항목만 대상으로
    하는 **축 간 재배분**이다. 관련 없는 항목으로 채우는 패딩이 아니다 —
    패딩하면 별점의 신뢰가 깨진다 (계획 D-2).
    """
    lo = config.DAILY_MIN if daily_min is None else daily_min
    hi = config.DAILY_MAX if daily_max is None else daily_max

    relevant = [e for e in entries if e.classification.is_relevant
                and e.classification.axis in config.AXIS_KEYS]
    result = SelectionResult()

    # ① 축별 목표치만큼 배분
    by_axis: dict[str, list[Entry]] = {a: [] for a in config.AXIS_KEYS}
    for e in relevant:
        by_axis[e.classification.axis].append(e)

    chosen: list[Entry] = []
    leftovers: list[Entry] = []
    for axis in config.AXES:
        pool = _sorted(by_axis[axis.key])
        take = pool[:axis.target]
        chosen.extend(take)
        leftovers.extend(pool[axis.target:])
        if len(take) < axis.bounds[0]:
            result.shortfall_by_axis[axis.key] = axis.bounds[0] - len(take)

    # ② 총합이 하한 미달이면 축 간 재배분 (각 축 최대치 준수)
    if len(chosen) < lo:
        counts = {a.key: sum(1 for e in chosen if e.classification.axis == a.key)
                  for a in config.AXES}
        for e in _sorted(leftovers):
            if len(chosen) >= lo:
                break
            ax = config.AXIS_BY_KEY[e.classification.axis]
            if counts[ax.key] < ax.bounds[1]:      # 축 최대치를 넘기지 않는다
                chosen.append(e)
                counts[ax.key] += 1

    # ③ 상한 절삭
    chosen = _sorted(chosen)
    if len(chosen) > hi:
        result.truncated_ids = [e.paper.primary_id for e in chosen[hi:]]
        chosen = chosen[:hi]

    result.entries = chosen
    picked = {e.paper.primary_id for e in chosen}
    result.deferred_ids = [e.paper.primary_id for e in entries
                           if e.paper.primary_id not in picked]
    return result
