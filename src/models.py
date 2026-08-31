"""파이프라인을 흐르는 데이터 구조."""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import config


# ─────────────────────────────────────────────────────────────────
# 정규화 — 교차 중복제거의 기반 (계획 D-12)
# ─────────────────────────────────────────────────────────────────

_VERSION_SUFFIX = re.compile(r"v\d+$")
_LATEX = re.compile(r"\$[^$]*\$|\\[a-zA-Z]+\{[^}]*\}|\\[a-zA-Z]+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_arxiv_id(raw: str) -> str:
    """'2401.12345v2' → '2401.12345'.

    정규화하지 않으면 개정판이 매번 신규로 잡혀 AC-2가 실패한다.
    """
    return _VERSION_SUFFIX.sub("", (raw or "").strip().rsplit("/", 1)[-1])


def normalize_doi(raw: str | None) -> str | None:
    """DOI 정규화. arXiv DataCite DOI는 매칭에서 제외한다.

    arXiv는 모든 제출물에 10.48550/arXiv.* 를 발급하는데, 이는 저널 DOI와
    절대 일치하지 않는다. 걸러내지 않으면 'DOI가 항상 존재'하게 되어
    제목 매칭 단계가 도달 불가능한 죽은 코드가 된다.
    """
    if not raw:
        return None
    d = raw.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if d.startswith(prefix):
            d = d[len(prefix):]
    return None if d.startswith(config.ARXIV_DOI_PREFIX) else (d or None)


def normalize_title(raw: str | None) -> str:
    """제목 정규화 — 소문자, LaTeX·구두점·공백 제거.

    퍼지 매칭은 쓰지 않는다. 오탐은 논문 소실(원칙 1)이고, 결정론적이어야
    테스트가 가능하다(원칙 4).
    """
    t = _LATEX.sub(" ", (raw or "").lower())
    return _NON_ALNUM.sub("", t)


# ─────────────────────────────────────────────────────────────────
# 논문
# ─────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    """수집된 후보 논문 하나. 소스 간 병합의 결과일 수 있다."""
    title: str
    source: str                              # arxiv | pubmed | s2
    url: str = ""
    abstract: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    announced_date: str = ""                 # YYYY-MM-DD (UTC)

    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    arxiv_doi: Optional[str] = None          # 10.48550/* — 매칭엔 안 쓰고 보관만

    raw_comments: Optional[str] = None       # arXiv comments 원문 (venue 추출용)
    journal: Optional[str] = None            # PubMed/S2 저널·학회명
    venue: Optional["Venue"] = None          # 추출된 학회 라벨

    def __post_init__(self) -> None:
        if self.arxiv_id:
            self.arxiv_id = normalize_arxiv_id(self.arxiv_id)
        self.doi = normalize_doi(self.doi)

    @property
    def norm_title(self) -> str:
        return normalize_title(self.title)

    @property
    def primary_id(self) -> str:
        """원장의 대표 키. arXiv > DOI > PMID > 제목 순."""
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        if self.doi:
            return f"doi:{self.doi}"
        if self.pmid:
            return f"pmid:{self.pmid}"
        return f"title:{self.norm_title[:60]}"

    def to_payload(self) -> dict:
        """보류 원장에 실을 최소 메타.

        재분류에는 제목·초록·학회가 있으면 충분하다. 저자 목록과 원문
        코멘트는 싣지 않는다 — 원장이 커지면 매일 push가 무거워진다.
        """
        return {k: v for k, v in (
            ("title", self.title), ("source", self.source), ("url", self.url),
            ("abstract", self.abstract), ("announced_date", self.announced_date),
            ("arxiv_id", self.arxiv_id), ("doi", self.doi), ("pmid", self.pmid),
            ("journal", self.journal),
        ) if v}

    @classmethod
    def from_payload(cls, d: dict) -> "Paper":
        """원장에서 되살린다. 없는 필드는 기본값 — 되살린 논문도 정상 항목이다."""
        return cls(title=d.get("title", ""), source=d.get("source", "arxiv"),
                   url=d.get("url", ""), abstract=d.get("abstract"),
                   announced_date=d.get("announced_date", ""),
                   arxiv_id=d.get("arxiv_id"), doi=d.get("doi"),
                   pmid=d.get("pmid"), journal=d.get("journal"))

    def identity(self) -> dict[str, str]:
        """식별자 집합. 하나라도 일치하면 같은 논문으로 본다 (계획 D-4).

        단일 키를 쓰면 프리프린트로 게시된 논문이 몇 달 뒤 저널 게재분으로
        다른 소스에서 들어올 때 다른 키가 되어 재게시된다 — AC-2가 정상
        경로에서 깨진다.
        """
        out = {"primary_id": self.primary_id, "norm_title": self.norm_title}
        for k in ("arxiv_id", "doi", "pmid"):
            if (v := getattr(self, k)):
                out[k] = v
        return out

    def sort_date(self) -> str:
        return self.announced_date or ""


@dataclass(frozen=True)
class Venue:
    """게재 확정이 확인된 학회·저널."""
    name: str                 # 근거가 된 원문 그대로 (arXiv comments·저널명).
                              # 표시용이 아니다 — 화면에는 tag를 쓴다.
    key: str                  # VENUE_BOOST_LIST의 키
    year: Optional[int] = None
    is_workshop: bool = False  # 워크숍은 본회의와 수락 기준이 다르다
    extras: tuple[str, ...] = ()   # oral / spotlight — 태그로만 쓴다

    @property
    def tag(self) -> str:
        """화면·프롬프트에 쓰는 라벨. key는 VENUE_BOOST_LIST의 키라 이미
        표시형이다 ("MICCAI", "NeurIPS", "IEEE TMI"). name은 원문이라 못 쓴다."""
        return f"{self.key} {self.year}" if self.year else self.key


# ─────────────────────────────────────────────────────────────────
# 분류 · 요약 · 최종 항목
# ─────────────────────────────────────────────────────────────────

@dataclass
class Classification:
    is_relevant: bool
    axis: Optional[str] = None
    category_id: Optional[str] = None
    stars: Optional[int] = None
    rationale: str = ""
    undecided: bool = False    # 파싱 실패·거부 — 버리지 않고 보류로 보낸다 (D-6)


@dataclass
class Summary:
    korean_summary: str
    tags: list[str] = field(default_factory=list)


@dataclass
class Entry:
    """페이지에 실리는 항목."""
    paper: Paper
    classification: Classification
    summary: Optional[Summary] = None
    boosted_stars: Optional[int] = None   # venue 부스트 적용 후 (표시용)

    @property
    def title_only(self) -> bool:
        """초록 없이 제목만으로 분류된 항목.

        일부 소스(Springer LNCS 등)는 초록을 공개 API로 주지 않는다. MICCAI
        표본에서 55%가 여기 해당했다. 초록이 없다고 감추면 리스트업 목적과
        충돌하므로 게시하되, 요약을 지어내지 않고 없다고 표시한다.
        """
        return not self.paper.abstract

    @property
    def display_stars(self) -> int:
        return self.boosted_stars if self.boosted_stars is not None else (self.classification.stars or 0)

    @property
    def display_tags(self) -> list[str]:
        """LLM 태그 + venue 태그. venue 태그는 AC-7 assert 이후에 붙는다."""
        tags = list(self.summary.tags) if self.summary else []
        if self.paper.venue:
            v = self.paper.venue.tag
            if v not in tags:
                tags.append(v)
        return tags


@dataclass
class SelectionResult:
    """select()의 반환값. 무엇이 잘렸는지도 함께 보고한다."""
    entries: list[Entry] = field(default_factory=list)
    shortfall_by_axis: dict[str, int] = field(default_factory=dict)
    truncated_ids: list[str] = field(default_factory=list)
    deferred_ids: list[str] = field(default_factory=list)

    def counts_by_axis(self) -> dict[str, int]:
        out = {a: 0 for a in config.AXIS_KEYS}
        for e in self.entries:
            if e.classification.axis in out:
                out[e.classification.axis] += 1
        return out
