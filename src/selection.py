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


def _apportion(deficit: int, capacity: dict[str, int]) -> dict[str, int]:
    """빈자리 deficit개를 축 **비중대로** 나눈다 (최대잉여법).

    capacity[k]는 축 k가 더 받을 수 있는 최대 개수 — 남은 공급과 축 천장 중
    작은 쪽이다. 합은 정확히 min(deficit, 총 capacity)가 된다.

    라운드로빈은 비중을 무시해 13% 축과 30% 축에 같은 수를 준다. 비율은 이
    제품의 편집 방침이므로 비율을 유지한 채 나눈다. 결정적이다 (원칙 4) —
    소수부 동률은 config.AXES 순서로 깨므로 입력 dict 순서에 좌우되지 않는다.

    ⚠️ 이 배분은 **축 안에서만** 별점 순이다. 축 간에는 비율이 우선하므로
       비중 큰 축의 3점이 비중 작은 축의 5점보다 먼저 실릴 수 있다. 비율
       유지를 택한 결과이지 정렬 버그가 아니다 (D-2의 "패딩 금지"와는 다른
       층위 — 여기 오르는 것은 전부 is_relevant=True다).
    """
    order = {a.key: i for i, a in enumerate(config.AXES)}
    alloc = {k: 0 for k in capacity}
    while deficit > 0:
        room = {a.key: capacity.get(a.key, 0) - alloc.get(a.key, 0)
                for a in config.AXES}
        recv = [a for a in config.AXES if room[a.key] > 0]
        if not recv:
            break
        total = sum(a.ratio for a in recv)
        exact = {a.key: deficit * a.ratio / total for a in recv}
        take = {a.key: min(int(exact[a.key]), room[a.key]) for a in recv}
        rest = deficit - sum(take.values())
        for a in sorted(recv, key=lambda x: (-(exact[x.key] % 1), order[x.key])):
            if rest <= 0:
                break
            if take[a.key] < room[a.key]:
                take[a.key] += 1
                rest -= 1
        got = sum(take.values())
        assert got > 0, "recv가 비지 않으면 최소 1은 배정된다"
        if got <= 0:        # 종료 보증 — assert는 -O에서 사라진다.
            break           # 도달 불가능이지만, 불변식이 깨져도 매달리진 않는다.
        for k, n in take.items():
            alloc[k] += n
        deficit -= got
    return alloc


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
        spare: dict[str, list[Entry]] = {k: [] for k in config.AXIS_KEYS}
        for e in _sorted(leftovers):
            ax = config.AXIS_BY_KEY[e.classification.axis]
            if len(chosen) < lo and counts[ax.key] < ax.bounds[1]:
                chosen.append(e)                   # 축 최대치까지는 그대로 채운다
                counts[ax.key] += 1
            else:
                spare[ax.key].append(e)            # ②-b가 쓸 잔량

        # ②-b 모든 축이 최대치에 걸렸는데도 하한에 못 미치는 경우.
        #     공급이 마른 축이 남긴 **유휴 칸**을 아직 공급이 있는 축에 넘긴다.
        #     축 최대치는 "다양성 보장"이지 "게시 금지"가 아니다 — 최대치를
        #     지키느라 관련 논문을 버리면 리스트업 품질(원칙 1)이 깨진다.
        #     (실측 2026-09-04: 관련 298건 중 35건만 실리고 263건이 quota로
        #      밀렸다. brain 6/18·surgical 3/16으로 25칸이 유휴였는데 나머지
        #      세 축은 전부 최대치에 걸려 있었다.)
        #     넘기되 axis.ceiling에서 멈춘다 — 최대치를 없애면 공급이 한 축에만
        #     있는 날 그 축이 페이지를 통째로 가져간다.
        if len(chosen) < lo:
            capacity = {a.key: min(len(spare[a.key]),
                                   max(0, a.ceiling - counts[a.key]))
                        for a in config.AXES}
            alloc = _apportion(lo - len(chosen), capacity)
            for a in config.AXES:
                chosen.extend(spare[a.key][:alloc.get(a.key, 0)])

    # ③ 상한 절삭
    chosen = _sorted(chosen)
    if len(chosen) > hi:
        result.truncated_ids = [e.paper.primary_id for e in chosen[hi:]]
        chosen = chosen[:hi]

    result.entries = chosen
    # 초과분은 **절삭까지 끝난 최종 개수**로 잰다. ②-b 시점에 재면 뒤이어
    # ③이 잘라낸 만큼이 어긋나 0건인 축을 초과로 보고하게 된다.
    # ①·②-a는 bounds[1]을 넘지 않으므로 여기 남는 초과는 전부 ②-b의 것이다.
    for key, n in result.counts_by_axis().items():
        over = n - config.AXIS_BY_KEY[key].bounds[1]
        if over > 0:
            result.overflow_by_axis[key] = over
    picked = {e.paper.primary_id for e in chosen}
    result.deferred_ids = [e.paper.primary_id for e in entries
                           if e.paper.primary_id not in picked]
    return result
