"""정적 HTML 생성.

autoescape를 켜고 요약·제목·태그에 `|safe`를 쓰지 않는다 (R-11). 이 값들은
LLM과 외부 API에서 온 문자열이라 신뢰할 수 없다. arXiv 제목에는 LaTeX(`$\\ell_1$`)와
`<`, `&`가 흔히 들어 있어, 이스케이프가 꺼지면 조용히 페이지가 깨진다.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import pathlib
import re
from typing import Iterable, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

import config
from models import Entry
from selection import _neg_date

TEMPLATE_DIR = pathlib.Path(__file__).resolve().parent.parent / "templates"

_SOURCE_LABEL = {"arxiv": "arXiv", "pubmed": "PubMed", "s2": "Semantic Scholar"}

# 소스마다 날짜 정밀도가 다르다 — S2는 "2025-09"처럼 월까지만 주기도 한다.
# 형식에 안 맞는 값(빈 문자열, "unknown")은 지어내지 말고 그냥 빼야 한다.
_DATE_RE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def display_date(raw: str | None) -> str:
    """화면에 낼 수 있는 날짜만 통과시킨다.

    파싱 못 하는 값을 그대로 뿌리면 "unknown"이 발표일 자리에 찍힌다
    (실측: 미리보기에서 6건).
    """
    raw = (raw or "").strip()
    return raw if _DATE_RE.match(raw) else ""


@dataclasses.dataclass
class PageMeta:
    """푸터에 실리는 이번 실행의 사실들.

    성공만 적지 않는다 — 미달·제외·절삭도 같이 적는다. 조용히 적게 나오는 것이
    이 제품에서 가장 위험한 실패다 (원칙 1).
    """
    data_date: str
    generated_at: dt.datetime
    source_counts: dict[str, int] = dataclasses.field(default_factory=dict)
    shortfall_by_axis: dict[str, int] = dataclasses.field(default_factory=dict)
    published_count: int = 0
    title_only_count: int = 0
    excluded_count: int = 0       # 관련 없음으로 판정된 수. 요약 실패와 다르다 —
    # 한동안 푸터에 "제외 276건 (요약 검증 실패)"로 잘못 나갔다.
    summary_failed_count: int = 0
    truncated_count: int = 0
    boosted_count: int = 0
    capped_sources: list[str] = dataclasses.field(default_factory=list)

    @property
    def generated_kst(self) -> str:
        return self.generated_at.strftime("%Y-%m-%d %H:%M")

    @property
    def stale_days(self) -> Optional[int]:
        """데이터가 며칠 낡았는지. 경고 기준 미만이면 None(배너 없음)."""
        try:
            d = dt.date.fromisoformat(self.data_date)
        except ValueError:
            return None
        days = (self.generated_at.date() - d).days
        return days if days >= config.STALENESS_WARN_DAYS else None

    @property
    def source_line(self) -> str:
        if not self.source_counts:
            return "—"
        return " · ".join(f"{_SOURCE_LABEL.get(k, k)} {v}건"
                          for k, v in self.source_counts.items())

    @property
    def shortfall_line(self) -> str:
        parts = [f"{_axis_label(k)} {v}건" for k, v in self.shortfall_by_axis.items() if v]
        return " · ".join(parts)


def _axis_label(key: str) -> str:
    for a in config.AXES:
        if a.key == key:
            return a.label
    return key


def _entry_sort_key(e: Entry):
    """별점↓, 발표일↓, id↑. selection.py와 같은 전순서를 쓴다.

    배치 도착 순서가 임의여도 같은 페이지가 나와야 하므로 id까지 넣어
    동점을 끊는다 (원칙 4).
    """
    return (-e.display_stars, _neg_date(e.paper.sort_date()), e.paper.primary_id)


def _build_sections(entries: Iterable[Entry]):
    """축 → 카테고리 → 항목. 빈 축·빈 카테고리는 내보내지 않는다.

    순서는 config.AXES / config.CATEGORIES 선언 순서를 따른다 — 매일 같은
    자리에 같은 주제가 있어야 훑어보기 편하다.
    """
    by_cat: dict[str, list[Entry]] = {}
    for e in entries:
        cid = e.classification.category_id
        if cid in config.CATEGORY_BY_ID:
            by_cat.setdefault(cid, []).append(e)

    sections = []
    for axis in config.AXES:
        groups = []
        for cat in config.CATEGORIES:
            if cat.axis != axis.key:
                continue
            got = by_cat.get(cat.id)
            if got:
                groups.append((cat, sorted(got, key=_entry_sort_key)))
        if groups:
            sections.append((axis, groups))
    return sections


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        # 확장자 목록에 'j2'를 반드시 넣는다. select_autoescape의 기본값은
        # .html/.xml만이라, base.html.j2 같은 이름은 이스케이프가 꺼진 채
        # 렌더된다 — 조용히 터지는 종류의 사고다.
        autoescape=select_autoescape(["html", "xml", "j2", "html.j2"]),
        trim_blocks=True, lstrip_blocks=True,
    )
    env.globals["source_label"] = lambda s: _SOURCE_LABEL.get(s, s)
    env.globals["display_date"] = display_date
    return env


def _archive_dates(archive_dir: pathlib.Path) -> list[str]:
    """보관된 날짜를 최신순으로. 파일 시스템이 유일한 사실 원천이다."""
    if not archive_dir.is_dir():
        return []
    dates = [p.stem for p in archive_dir.glob("*.html") if p.stem != "index"]
    return sorted((d for d in dates if _is_date(d)), reverse=True)


def _is_date(s: str) -> bool:
    try:
        dt.date.fromisoformat(s)
        return True
    except ValueError:
        return False


def render(entries: list[Entry], out_root: pathlib.Path | str,
           meta: PageMeta, log=print) -> list[pathlib.Path]:
    """페이지를 쓰고 만들어진 경로를 돌려준다.

    `index.html`과 `research_latest.html`에 같은 내용을 쓴다. 리다이렉트로
    처리하면 Pages 루트에서 한 번 깜빡이고, 무엇보다 링크를 공유했을 때
    받는 쪽이 빈 페이지를 먼저 본다. 수십 KB 중복이 그보다 싸다.
    """
    out_root = pathlib.Path(out_root)
    archive_dir = out_root / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    env = _env()
    meta.published_count = meta.published_count or len(entries)
    meta.title_only_count = meta.title_only_count or sum(1 for e in entries if e.title_only)
    meta.boosted_count = meta.boosted_count or sum(
        1 for e in entries if e.boosted_stars is not None)
    meta.summary_failed_count = meta.summary_failed_count or sum(
        1 for e in entries if e.summary is None and not e.title_only)

    sections = _build_sections(entries)
    digest = env.get_template("digest.html.j2")

    written: list[pathlib.Path] = []
    for path, rel in ((out_root / "index.html", ""),
                      (out_root / "research_latest.html", ""),
                      (archive_dir / f"{meta.data_date}.html", "../")):
        html = digest.render(site_title=config.SITE_TITLE, sections=sections,
                             meta=meta, rel=rel,
                             page_title=f"{config.SITE_TITLE} — {meta.data_date}")
        path.write_text(html, encoding="utf-8")
        written.append(path)

    # 아카이브 인덱스는 방금 쓴 파일까지 포함해 다시 만든다
    idx = archive_dir / "index.html"
    idx.write_text(
        env.get_template("archive_index.html.j2").render(
            site_title=config.SITE_TITLE, dates=_archive_dates(archive_dir),
            meta=meta, rel="../", page_title=f"{config.SITE_TITLE} — 아카이브"),
        encoding="utf-8")
    written.append(idx)

    # Jekyll이 _로 시작하는 경로를 삼키지 않도록 (Pages 브랜치 소스)
    (out_root / ".nojekyll").touch()

    log(f"  [render] {len(entries)}건 → {len(written)}개 파일")
    return written
