"""세 소스 통합 + 교차 중복제거 + 원장 대조 (계획 D-12).

매칭 순서가 중요하다:
  ① externalIds.ArXiv 직접 키 (S2 ↔ arXiv) — 가장 확실
  ② DOI 일치 — arXiv DataCite DOI는 제외한 뒤
  ③ 정규화 제목 일치 — ①②가 실패했을 때만

②가 실패해도 ③으로 내려가야 한다. "DOI가 없으면 제목"으로 짜면, arXiv가
모든 제출물에 10.48550/* 를 발급하므로 DOI는 항상 존재하고 저널 DOI와는
절대 일치하지 않아 ③이 도달 불가능한 죽은 코드가 된다.

퍼지 매칭은 쓰지 않는다 — 오탐은 논문 소실(원칙 1)이고 결정론적이어야
테스트가 가능하다(원칙 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import config
import venue as venue_mod
from ledger import DeferredLedger, PublishedLedger
from models import Paper
from sources import arxiv as arxiv_src
from sources import pubmed as pubmed_src
from sources import semanticscholar as s2_src

# 병합 우선순위 — 초록은 arXiv가 가장 안정적, venue 라벨은 S2가 가장 신뢰도 높음
_ABSTRACT_PRIORITY = {"arxiv": 0, "pubmed": 1, "s2": 2}
_VENUE_PRIORITY = {"s2": 0, "pubmed": 1, "arxiv": 2}


@dataclass
class IngestStats:
    per_source: dict[str, int] = field(default_factory=dict)
    merged: int = 0                # 교차 중복제거로 합쳐진 건수
    already_published: int = 0
    deferred_reentered: int = 0
    truncated: dict[str, int] = field(default_factory=dict)
    deferred_expired: int = 0
    already_judged: int = 0        # 같은 프롬프트로 이미 판정해 건너뛴 수
    capped: list[str] = field(default_factory=list)   # 상한에 걸린 소스
    failed: list[str] = field(default_factory=list)   # 조회 자체가 실패한 소스
    title_only: int = 0            # 초록 미확보 (게시는 하되 요약 없음)

    def render(self) -> str:
        src = " / ".join(f"{k} {v}" for k, v in self.per_source.items())
        trunc = " / ".join(f"{k} {v}" for k, v in self.truncated.items()) or "없음"
        return (f"수집 [{src}] | 병합 {self.merged} | 기게시 제외 {self.already_published} "
                f"| 기판정 제외 {self.already_judged} "
                f"| 보류 재진입 {self.deferred_reentered} | 절삭 [{trunc}] "
                f"| TTL 만료 {self.deferred_expired} | 초록미확보 {self.title_only}"
                + (f" | ⚠️ 상한 도달 [{', '.join(self.capped)}]" if self.capped else "")
                + (f" | ⚠️ 수집 실패 [{', '.join(self.failed)}]" if self.failed else ""))


def _merge(base: Paper, other: Paper) -> Paper:
    """두 레코드를 합친다. 각 필드는 신뢰도 높은 소스 것을 취한다."""
    if _ABSTRACT_PRIORITY.get(other.source, 9) < _ABSTRACT_PRIORITY.get(base.source, 9):
        if other.abstract:
            base.abstract = other.abstract
    elif not base.abstract and other.abstract:
        base.abstract = other.abstract

    # 식별자는 있는 쪽을 채운다
    for f in ("arxiv_id", "doi", "pmid", "arxiv_doi", "raw_comments"):
        if not getattr(base, f) and getattr(other, f):
            setattr(base, f, getattr(other, f))

    # venue 라벨은 신뢰도 순
    if other.journal and (
        not base.journal
        or _VENUE_PRIORITY.get(other.source, 9) < _VENUE_PRIORITY.get(base.source, 9)
    ):
        base.journal = other.journal

    if not base.announced_date and other.announced_date:
        base.announced_date = other.announced_date
    if not base.authors and other.authors:
        base.authors = other.authors
    return base


def cross_dedupe(papers: list[Paper]) -> tuple[list[Paper], int]:
    """3단계 매칭으로 같은 논문을 하나로 합친다."""
    by_arxiv: dict[str, Paper] = {}
    by_doi: dict[str, Paper] = {}
    by_title: dict[str, Paper] = {}
    out: list[Paper] = []
    merged = 0

    for p in papers:
        hit = None
        if p.arxiv_id:
            hit = by_arxiv.get(p.arxiv_id)          # ① arXiv ID 직접 키
        if hit is None and p.doi:
            hit = by_doi.get(p.doi)                 # ② DOI (arXiv DataCite 제외됨)
        if hit is None and p.norm_title:
            hit = by_title.get(p.norm_title)        # ③ 정규화 제목
        if hit is not None:
            _merge(hit, p)
            merged += 1
        else:
            out.append(p)
            hit = p
        # 인덱스 갱신 — 병합으로 새 식별자가 생겼을 수 있다
        if hit.arxiv_id:
            by_arxiv.setdefault(hit.arxiv_id, hit)
        if hit.doi:
            by_doi.setdefault(hit.doi, hit)
        if hit.norm_title:
            by_title.setdefault(hit.norm_title, hit)
    return out, merged


def _truncate(papers: list[Paper], source: str, cap: int) -> tuple[list[Paper], int]:
    """소스별 상한 적용. 전역 단일 상한이면 대량 색인된 proceedings 볼륨이
    그날 arXiv를 통째로 밀어낼 수 있다 (계획 D-11)."""
    subset = [p for p in papers if p.source == source]
    if len(subset) <= cap:
        return subset, 0
    subset.sort(key=lambda p: p.sort_date(), reverse=True)   # 최신순
    return subset[:cap], len(subset) - cap


def collect(cycle_date: str, *, ignore_seen: bool = False,
            replay_deferred: bool = False, dry_run: bool = False,
            log=print) -> tuple[list[Paper], IngestStats]:
    """하루치 후보를 모은다."""
    stats = IngestStats()
    raw: list[Paper] = []

    for name, fn, cap in (
        ("arxiv", arxiv_src.fetch, config.ARXIV_MAX),
        ("pubmed", pubmed_src.fetch, config.PUBMED_MAX),
        ("s2", s2_src.fetch, config.S2_MAX),
    ):
        try:
            got = fn(cycle_date, log=log)
        except Exception as e:                # noqa: BLE001
            # 한 소스가 죽어도 나머지로 발행한다. arXiv만 예외를 밖으로 던져
            # 파이프라인 전체가 죽었고, 그 탓에 이틀 동안 사이트가 멈췄다
            # (2026-09-02~03, arXiv 429). 못 가져온 구간은 arXiv 상태 파일이
            # 기억해 다음 실행이 창을 늘려 메운다.
            log(f"  [{name}] ⚠️  수집 실패 — 이 소스만 건너뛴다: {e}")
            stats.failed.append(name)
            continue
        if len(got) >= cap:
            # 소스가 스스로 상한에서 멈춘 경우다. _truncate는 초과분만 세므로
            # 이 경로를 못 잡는다 — 상한이 곧 누락인데 화면에 안 보였다.
            stats.capped.append(name)
        kept, cut = _truncate(got, name, cap)
        stats.per_source[name] = len(kept)
        if cut:
            stats.truncated[name] = cut
        raw.extend(kept)

    deduped, stats.merged = cross_dedupe(raw)

    # venue 라벨 추출 (부스트는 select() 이후에 적용된다 — D-13)
    for p in deduped:
        ref_year = int(p.announced_date[:4]) if p.announced_date[:4].isdigit() else None
        p.venue = venue_mod.extract(comments=p.raw_comments, journal=p.journal,
                                    s2_venue=p.journal if p.source == "s2" else None,
                                    ref_year=ref_year)

    # 원장 대조
    if not ignore_seen:
        pub = PublishedLedger()
        before = len(deduped)
        deduped = [p for p in deduped if not pub.is_published(p)]
        stats.already_published = before - len(deduped)

        dl = DeferredLedger()
        stats.deferred_expired = len(dl.expired(cycle_date))
        active = dl.active(cycle_date, force=replay_deferred)
        stats.deferred_reentered = len(active)

        if not replay_deferred:
            # 같은 프롬프트로 이미 탈락시킨 논문은 다시 분류하지 않는다.
            # 30일 창에서는 이게 없으면 매일 수천 건을 재판정한다.
            before = len(deduped)
            deduped = [p for p in deduped if not dl.is_deferred(p)]
            stats.already_judged = before - len(deduped)

    stats.title_only = sum(1 for p in deduped if not p.abstract)
    log(f"  [ingest] {stats.render()}")
    return deduped, stats
