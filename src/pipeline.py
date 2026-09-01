"""수집 → 분류 → 선별 → 요약 → 렌더 → 원장 기록.

⚠️ 이 파일을 직접 실행할 때 `if __name__ == "__main__":` 가드가 반드시
   필요하다. vLLM v1 엔진은 워커를 spawn으로 띄우며 메인 스크립트를
   재import하는데, 가드가 없으면 자식 프로세스가 파이프라인 전체를
   다시 실행한다 (설치 중 실제로 겪었다).

요약은 **선별된 40~60건에만** 돌린다. 분류는 후보 전체(~1,500건)에 필요하지만
요약은 게시할 것만 있으면 된다. 여기서 비용이 30배 갈린다.
"""
from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys
import traceback

import classify as classify_mod
import config
import ingest
import render as render_mod
import selection
import summarize as summarize_mod
import venue as venue_mod
from ledger import DeferredLedger, PublishedLedger
from models import Entry, Paper
from render import PageMeta

KST = dt.timezone(dt.timedelta(hours=9))


def _today_kst() -> str:
    return dt.datetime.now(KST).date().isoformat()


class Reporter:
    """진행 상황을 stdout으로. cron 로그가 유일한 관측 창구다."""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.t0 = dt.datetime.now(KST)

    def __call__(self, msg: str = "") -> None:
        if self.quiet:
            return
        el = (dt.datetime.now(KST) - self.t0).total_seconds()
        print(f"[{el:6.0f}s] {msg}", flush=True)


def _replayed_papers(cycle_date: str, force: bool, log) -> list[Paper]:
    """보류 원장에서 재분류 대상을 되살린다.

    payload가 없는 옛 기록은 되살릴 수 없다 — 그 사실을 세어서 알린다.
    조용히 건너뛰면 회수 기제가 도는 줄 알고 지나간다.
    """
    dl = DeferredLedger()
    recs = dl.active(cycle_date, force=force)
    papers, unusable = [], 0
    for r in recs:
        if r.payload:
            papers.append(Paper.from_payload(r.payload))
        else:
            unusable += 1
    if recs:
        log(f"  [replay] 보류 재진입 {len(papers)}건"
            + (f" (payload 없어 복원 불가 {unusable}건)" if unusable else ""))
    return papers


def _classify_all(llm, papers: list[Paper], log) -> list[Entry]:
    """후보 전체를 분류하고 Entry로 만든다. 관련 없음도 함께 돌려준다."""
    entries: list[Entry] = []
    total = len(papers)
    for i in range(0, total, config.CLASSIFY_BATCH):
        chunk = papers[i:i + config.CLASSIFY_BATCH]
        log(f"  [classify] {i + len(chunk)}/{total}")
        for p, c in zip(chunk, classify_mod.classify(llm, chunk, log=lambda *_: None)):
            entries.append(Entry(paper=p, classification=c))
    return entries


def _apply_venue_boost(entries: list[Entry]) -> int:
    """학회 부스트는 선별 **이후**에 붙인다 (D-13).

    선별 전에 붙이면 부스트가 쿼터 배분을 밀어내 축 비율이 깨진다.
    """
    n = 0
    for e in entries:
        boosted = venue_mod.boost(e.classification.stars, e.paper.venue)
        if boosted != (e.classification.stars or 0):
            e.boosted_stars = boosted
            n += 1
    return n


def run(cycle_date: str | None = None, *, dry_run: bool = False,
        ignore_seen: bool = False, replay_deferred: bool = False,
        out_dir: str | None = None, log=print) -> int:
    """하루치를 만든다. 반환값은 게시 건수."""
    cycle_date = cycle_date or _today_kst()
    out_dir = out_dir or config.DOCS_DIR
    log(f"=== {cycle_date} 시작 (dry_run={dry_run}) ===")

    # 1) 수집
    papers, stats = ingest.collect(cycle_date, ignore_seen=ignore_seen,
                                   replay_deferred=replay_deferred,
                                   dry_run=dry_run, log=log)
    if not ignore_seen:
        papers += _replayed_papers(cycle_date, replay_deferred, log)

    if dry_run:
        # 모델을 띄우지 않는다. 수집 경로만 점검하는 모드다.
        log(f"=== dry-run 종료: 후보 {len(papers)}건 ===")
        return 0

    # 후보가 없으면 모델을 띄우지 않는다. arXiv는 금·토 저녁에 발표하지 않아
    # 주 2회는 구조적으로 저물량이다 — 빈 날에 32B를 100초 올리고 내릴 이유가 없다.
    if not papers:
        log("  후보 0건 — 모델을 띄우지 않고 빈 페이지를 갱신한다")
        render_mod.render([], pathlib.Path(out_dir), PageMeta(
            data_date=cycle_date,
            generated_at=dt.datetime.now(KST).replace(microsecond=0),
            source_counts=dict(stats.per_source)), log=log)
        log(f"=== {cycle_date} 완료: 0건 게시 ===")
        return 0

    # 2) 분류 + 3) 선별 + 4) 요약 — GPU를 쥐는 구간을 한 번으로 묶는다
    from llm import LocalLLM
    with LocalLLM(log=log) as llm:
        entries = _classify_all(llm, papers, log)

        relevant = [e for e in entries if e.classification.is_relevant]
        dropped = [e for e in entries if not e.classification.is_relevant]
        log(f"  [classify] 관련 {len(relevant)} / 무관 {len(dropped)}")

        result = selection.select(relevant)
        chosen = result.entries
        log(f"  [select] {len(chosen)}건 선별 (축별 {result.counts_by_axis()})")
        summaries = summarize_mod.summarize(llm, [e.paper for e in chosen], log=log)
        for e, s in zip(chosen, summaries):
            e.summary = s

    boosted = _apply_venue_boost(chosen)

    # 5) 렌더
    meta = PageMeta(
        data_date=cycle_date,
        generated_at=dt.datetime.now(KST).replace(microsecond=0),
        source_counts=dict(stats.per_source),
        shortfall_by_axis=dict(result.shortfall_by_axis),
        excluded_count=len(dropped),
        truncated_count=len(result.truncated_ids),
        boosted_count=boosted,
        capped_sources=list(stats.capped),
    )
    render_mod.render(chosen, pathlib.Path(out_dir), meta, log=log)

    # 6) 원장 — 렌더가 성공한 뒤에 쓴다.
    #    먼저 쓰면 렌더가 터졌을 때 게시되지 않은 논문이 '기게시'로 남아
    #    영원히 다시 나오지 않는다.
    if not ignore_seen:
        PublishedLedger().add([e.paper for e in chosen], cycle_date)
        # 관련 있다고 판정됐지만 자리가 없어 못 실은 것들. 이걸 안 남기면
        # 분류 프롬프트를 고쳐도 회수할 대상이 없다.
        picked = {e.paper.primary_id for e in chosen}
        left_out = [e for e in relevant if e.paper.primary_id not in picked]
        DeferredLedger().defer(
            [(e.paper, "not_relevant") for e in dropped]
            + [(e.paper, "undecided" if e.classification.undecided else "quota")
               for e in left_out],
            cycle_date)
        log(f"  [ledger] 게시 {len(chosen)} / 보류 {len(dropped) + len(left_out)}")
        DeferredLedger().prune(cycle_date, log=log)

    log(f"=== {cycle_date} 완료: {len(chosen)}건 게시 ===")
    return len(chosen)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pipeline", description="일일 다이제스트 생성")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="수집부터 렌더까지 한 번 실행")
    r.add_argument("--date", help="기준일 YYYY-MM-DD (기본: 오늘 KST)")
    r.add_argument("--dry-run", action="store_true",
                   help="수집까지만. 모델을 띄우지 않는다")
    r.add_argument("--ignore-seen", action="store_true",
                   help="원장을 무시하고 다시 만든다. 원장에 쓰지도 않는다")
    r.add_argument("--replay-deferred", action="store_true",
                   help="프롬프트 버전과 무관하게 TTL 내 보류분을 재분류")
    r.add_argument("--out-dir", help=f"출력 경로 (기본: {config.DOCS_DIR})")
    r.add_argument("--quiet", action="store_true")

    args = ap.parse_args(argv)
    log = Reporter(quiet=args.quiet)
    try:
        run(args.date, dry_run=args.dry_run, ignore_seen=args.ignore_seen,
            replay_deferred=args.replay_deferred, out_dir=args.out_dir, log=log)
        return 0
    except Exception:                       # noqa: BLE001
        # 실패를 조용히 삼키지 않는다. cron 로그에 전문을 남기고 0이 아닌 코드로
        # 끝내야 run_daily.sh가 push를 건너뛴다.
        traceback.print_exc()
        log("=== 실패 ===")
        return 1


if __name__ == "__main__":
    sys.exit(main())
