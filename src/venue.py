"""학회·저널 라벨 추출 (계획 D-13).

핵심 원칙 두 가지:
  1. LLM이 아니라 코드가 판정한다. "CVPR에 붙었나"는 사실 확인이지 판단이
     아니고, 코드로 하면 단위 테스트로 고정된다 (D-1과 같은 철학).
  2. 부스트는 select() **이후에** 적용한다. 선별에 영향을 주면 별점의 의미
     (내용 판정)가 소스 아티팩트로 오염된다.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

import config
from models import Venue

# 골든셋 실측이 이 설계를 규정했다:
#   PeskaVLP → "the 38th Conference on Neural Information Processing Systems"
#   MindLLM  → "Forty-Second International Conference on Machine Learning"
# 둘 다 약칭이 없다. 약칭만 찾는 정규식이면 놓친다.
_YEAR = re.compile(r"\b(19|20|21)\d{2}\b")

# "CVPRW", "CVPR Workshop", "CVPR-W" → 워크숍. 본회의와 수락 기준이 다르다.
_WORKSHOP = re.compile(r"\b(workshops?|[A-Z]{3,7}W)\b", re.I)

_EXTRA_TOKENS = ("oral", "spotlight", "highlight", "early accept", "best paper")


def _compile(patterns: tuple[str, ...]) -> re.Pattern:
    # 단어 경계로 감싸 오탐을 줄이되, 약칭 뒤 워크숍 접미사(CVPRW)를 허용한다.
    # W? 없이 \bCVPR\b로만 두면 "CVPRW"에 경계가 없어 매칭이 실패한다.
    return re.compile("|".join(rf"\b(?:{p})W?\b" for p in patterns), re.I)


_VENUE_RE = {k: _compile(v) for k, v in config.VENUE_BOOST_LIST.items()}
_NEGATIVE_RE = re.compile("|".join(re.escape(p) for p in config.VENUE_NEGATIVE_CONTEXT), re.I)


def _plausible_year(text: str, ref_year: Optional[int] = None) -> Optional[int]:
    """텍스트에서 연도를 뽑되, 논문 시점 기준으로 최근 것만 인정한다.

    'Uses the MICCAI 2018 challenge dataset' 같은 참조를 수락으로 오인하지
    않기 위한 가드. 기준은 **논문 발표 연도**다 — 오늘 기준으로 하면
    2024년 논문의 "NeurIPS 2024"가 오래됐다는 이유로 탈락한다
    (골든셋 PeskaVLP에서 실제로 발생).
    연도가 없으면 None(부스트는 여전히 가능), 오래된 것만 있으면 -1.
    """
    now = ref_year or date.today().year
    lo, hi = now - config.VENUE_YEAR_TOLERANCE, now + config.VENUE_YEAR_TOLERANCE
    years = [int(m.group()) for m in _YEAR.finditer(text or "")]
    recent = [y for y in years if lo <= y <= hi]
    if recent:
        return max(recent)
    return -1 if years else None      # -1 = 오래된 연도만 있음 → 부스트 금지


def extract(comments: str | None = None, journal: str | None = None,
            s2_venue: str | None = None, ref_year: Optional[int] = None) -> Optional[Venue]:
    """세 경로에서 학회 라벨을 뽑는다. 신뢰도 순: S2 > 저널명 > arXiv comments.

    S2의 `venue` 필드는 구조화된 메타데이터라 정규식을 거치지 않고 신뢰한다.
    arXiv `comments`는 사람이 쓴 자유 텍스트라 가드가 필요하다.
    """
    # ① S2 venue — 가장 신뢰도 높음
    if s2_venue:
        for key, rx in _VENUE_RE.items():
            if rx.search(s2_venue):
                y = _plausible_year(s2_venue, ref_year)
                return Venue(name=s2_venue.strip(), key=key,
                             year=None if y in (None, -1) else y,
                             is_workshop=bool(_WORKSHOP.search(s2_venue)))

    # ② PubMed 저널명 — 구조화 필드라 부정문맥 가드 불필요
    if journal:
        for key, rx in _VENUE_RE.items():
            if rx.search(journal):
                return Venue(name=journal.strip(), key=key, year=None)

    # ③ arXiv comments — 자유 텍스트, 가드 필요
    if comments:
        if _NEGATIVE_RE.search(comments):
            return None                      # "submitted to" 등은 게재 확정이 아니다
        y = _plausible_year(comments, ref_year)
        if y == -1:
            return None                      # 오래된 연도만 → 참조로 본다
        for key, rx in _VENUE_RE.items():
            if rx.search(comments):
                extras = tuple(t for t in _EXTRA_TOKENS if t in comments.lower())
                return Venue(name=comments.strip()[:120], key=key, year=y,
                             is_workshop=bool(_WORKSHOP.search(comments)),
                             extras=extras)
    return None


def boost(stars: int | None, venue: Optional[Venue]) -> int:
    """별점 +1 (상한 5, 중첩 없음).

    oral/spotlight는 추가 부스트하지 않고 태그로만 남긴다 — 중첩하면 별점이
    학회 신호에 지배되어 내용 판정이 묻힌다.
    워크숍은 본회의와 수락 기준이 달라 부스트하지 않는다.
    """
    s = stars or 0
    if venue is None or venue.is_workshop:
        return s
    return min(s + config.VENUE_BOOST, config.STAR_MAX)
