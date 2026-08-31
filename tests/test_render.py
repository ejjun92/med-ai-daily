"""렌더링 테스트.

가장 중요한 것은 이스케이프다. arXiv 제목에는 LaTeX(`$\\ell_1$`)와 `<`, `&`가
흔하고, 요약은 LLM이 만든 신뢰할 수 없는 문자열이다. autoescape가 꺼지면
페이지가 조용히 깨지거나 스크립트가 실행된다 (R-11).
"""
import datetime as dt
import re

import config
import pytest
from models import Classification, Entry, Paper, Summary, Venue
from render import PageMeta, render

KST = dt.timezone(dt.timedelta(hours=9))
NOW = dt.datetime(2026, 8, 31, 9, 42, tzinfo=KST)


def entry(title="A Paper", cat="surgical_vlp", stars=4, summary="요약입니다. " * 12,
          tags=("surgical-video",), abstract="x", venue=None, date="2026-08-30",
          source="arxiv", url="https://arxiv.org/abs/2601.00001", aid="2601.00001",
          authors=("Kim, A.",), boosted=None):
    p = Paper(title=title, source=source, url=url, abstract=abstract,
              announced_date=date, arxiv_id=aid, authors=list(authors), venue=venue)
    return Entry(paper=p,
                 classification=Classification(is_relevant=True, stars=stars,
                                               axis=config.CATEGORY_BY_ID[cat].axis,
                                               category_id=cat),
                 summary=Summary(korean_summary=summary, tags=list(tags)) if summary else None,
                 boosted_stars=boosted)


def meta(date="2026-08-31", **kw):
    return PageMeta(data_date=date, generated_at=NOW, **kw)


def build(tmp_path, entries, m=None):
    paths = render(entries, tmp_path, m or meta(), log=lambda *_: None)
    return {p.name if p.parent == tmp_path else f"archive/{p.name}": p.read_text()
            for p in paths}


# ── 이스케이프 (R-11) ────────────────────────────────────────
DANGEROUS = [
    ("LaTeX", r"Sparse Coding with $\ell_1$ & $\alpha<\beta$ Regularization"),
    ("부등호", "Attention <script>alert(1)</script> Is All You Need"),
    ("앰퍼샌드", "Vision & Language: Q&A for Surgery"),
    ("따옴표", 'A "Foundation" Model for Chest X-Ray'),
]


@pytest.mark.parametrize("label,title", DANGEROUS, ids=[d[0] for d in DANGEROUS])
def test_title_is_escaped(tmp_path, label, title):
    html = build(tmp_path, [entry(title=title)])["index.html"]
    assert "<script>" not in html
    if "&" in title:
        assert "&amp;" in html
    if "<" in title:
        assert "&lt;" in html


def test_llm_summary_cannot_inject_html(tmp_path):
    """요약은 LLM이 만든 문자열이다. 절대 신뢰하지 않는다."""
    evil = ('<img src=x onerror=alert(1)> 이 요약은 공격을 시도한다. '
            '두 번째 문장이다. 세 번째 문장으로 길이를 맞춘다.')
    html = build(tmp_path, [entry(summary=evil)])["index.html"]
    # 태그가 이스케이프되면 onerror는 실행되지 않는 평범한 글자다.
    assert "<img" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html


def test_tags_are_escaped(tmp_path):
    html = build(tmp_path, [entry(tags=("<b>bold</b>", "a&b"))])["index.html"]
    assert "<b>bold</b>" not in html
    assert "&lt;b&gt;" in html and "a&amp;b" in html


def test_url_attribute_is_escaped(tmp_path):
    html = build(tmp_path, [entry(url='https://x.test/a"onmouseover="alert(1)')])["index.html"]
    assert 'onmouseover="alert' not in html


def test_autoescape_covers_j2_extension():
    """select_autoescape 기본값은 .html/.xml만이다. .j2가 빠지면 전부 무방비다."""
    from render import _env
    assert _env().autoescape("base.html.j2") is True
    assert _env().autoescape("digest.html.j2") is True


# ── 필수 필드 (AC-9) ─────────────────────────────────────────
def test_entry_shows_every_required_field(tmp_path):
    # key는 VENUE_BOOST_LIST의 키(표시형), name은 근거 원문이다.
    v = Venue(name="Accepted to MICCAI 2025, 8 pages", key="MICCAI", year=2025)
    e = entry(title="ReSurgSAM2: Referring Segmentation", stars=4, venue=v,
              summary="수술 장면 분할 문제를 다룬다. " * 5, tags=("surgical-video",))
    html = build(tmp_path, [e])["index.html"]
    assert "ReSurgSAM2" in html                       # 제목
    assert "https://arxiv.org/abs/2601.00001" in html  # 원문 링크
    assert "arXiv" in html                             # 출처
    assert "2026-08-30" in html                        # 발표일
    assert "★★★★☆" in html                            # 별점
    assert "수술 장면 분할" in html                     # 한국어 요약
    assert "surgical-video" in html                    # 태그
    assert "MICCAI 2025" in html                       # 학회 라벨
    assert "8 pages" not in html, "venue.name(원문)이 화면에 새어나갔다"


def test_title_only_entry_says_so_instead_of_inventing(tmp_path):
    html = build(tmp_path, [entry(abstract=None, summary=None)])["index.html"]
    assert "초록 미확보" in html


def test_boosted_stars_win_over_raw(tmp_path):
    html = build(tmp_path, [entry(stars=3, boosted=4)])["index.html"]
    assert "★★★★☆" in html and "★★★☆☆" not in html


# ── 구조와 정렬 (AC-8) ───────────────────────────────────────
def test_axes_appear_in_config_order(tmp_path):
    es = [entry(cat="surgical_vlp"), entry(cat="fmri_visual_decoding", aid="2601.2")]
    html = build(tmp_path, es)["index.html"]
    order = [a.label for a in config.AXES if a.label in html]
    positions = [html.index(lbl) for lbl in order]
    assert positions == sorted(positions), "축 순서가 설정 선언 순서와 다르다"


def test_entries_sorted_by_stars_desc(tmp_path):
    es = [entry(title="Low", stars=2, aid="2601.1"),
          entry(title="High", stars=5, aid="2601.2"),
          entry(title="Mid", stars=3, aid="2601.3")]
    html = build(tmp_path, es)["index.html"]
    assert html.index("High") < html.index("Mid") < html.index("Low")


def test_empty_axes_and_categories_are_omitted(tmp_path):
    html = build(tmp_path, [entry(cat="surgical_vlp")])["index.html"]
    assert "Brain Decoding" not in html


def test_renders_with_no_entries(tmp_path):
    """0건이어도 페이지는 나온다 — 조용히 사라지지 않는다."""
    html = build(tmp_path, [])["index.html"]
    assert "게시할 항목이 없습니다" in html


# ── 아카이브 (AC-9) ──────────────────────────────────────────
def test_archive_file_and_index_created(tmp_path):
    render([entry()], tmp_path, meta(date="2026-08-31"), log=lambda *_: None)
    render([entry()], tmp_path, meta(date="2026-08-30"), log=lambda *_: None)
    assert (tmp_path / "archive" / "2026-08-31.html").exists()
    assert (tmp_path / "archive" / "2026-08-30.html").exists()
    idx = (tmp_path / "archive" / "index.html").read_text()
    assert idx.index("2026-08-31") < idx.index("2026-08-30"), "아카이브는 최신순"


def test_archive_index_ignores_non_date_files(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "notes.html").write_text("x")
    render([entry()], tmp_path, meta(date="2026-08-31"), log=lambda *_: None)
    assert "notes" not in (tmp_path / "archive" / "index.html").read_text()


def test_archive_page_links_back_up_one_level(tmp_path):
    out = build(tmp_path, [entry()])
    assert 'href="../index.html"' in out["archive/2026-08-31.html"]
    assert 'href="index.html"' in out["index.html"]


def test_latest_and_index_have_same_content(tmp_path):
    out = build(tmp_path, [entry()])
    assert out["index.html"] == out["research_latest.html"]


def test_nojekyll_written(tmp_path):
    render([entry()], tmp_path, meta(), log=lambda *_: None)
    assert (tmp_path / ".nojekyll").exists()


# ── 푸터: 실패도 보인다 (AC-15, D-19) ────────────────────────
def test_footer_reports_counts_and_shortfall(tmp_path):
    m = meta(source_counts={"arxiv": 1200, "pubmed": 80, "s2": 240},
             shortfall_by_axis={"medical_imaging": 3, "surgical_video": 0},
             excluded_count=2, truncated_count=1, boosted_count=7)
    html = build(tmp_path, [entry()], m)["index.html"]
    assert "arXiv 1200건" in html and "PubMed 80건" in html
    assert "의료영상 AI 3건" in html
    assert "surgical_video" not in html, "0인 축은 노이즈다"
    assert "관련 없음" in html and "2건" in html
    assert "요약 검증 실패" not in html, "관련 없음 건수를 요약 실패로 표시하면 안 된다"
    assert "1건" in html and "7건" in html
    assert "2026-08-31 09:42 KST" in html


def test_stale_banner_appears_only_when_old(tmp_path):
    fresh = build(tmp_path, [entry()], meta(date="2026-08-31"))["index.html"]
    assert 'class="banner"' not in fresh

    old = build(tmp_path, [entry()], meta(date="2026-08-20"))["index.html"]
    assert "11일 지났습니다" in old


def test_stale_threshold_is_config_driven(tmp_path):
    d = NOW.date() - dt.timedelta(days=config.STALENESS_WARN_DAYS)
    html = build(tmp_path, [entry()], meta(date=d.isoformat()))["index.html"]
    assert 'class="banner"' in html


def test_title_only_count_reported_in_footer(tmp_path):
    es = [entry(aid="2601.1"), entry(aid="2601.2", abstract=None, summary=None)]
    html = build(tmp_path, es)["index.html"]
    assert "초록 미확보 1건" in html


# ── 가로 스크롤 방지 ─────────────────────────────────────────
def test_long_unbroken_title_wraps(tmp_path):
    html = build(tmp_path, [entry(title="A" * 300)])["index.html"]
    assert "word-break" in html or "overflow-wrap" in html


# ── 실제 데이터가 드러낸 결함 (미리보기 육안 검토) ──────────
@pytest.mark.parametrize("raw,shown", [
    ("2026-08-30", True), ("2025-09", True), ("2024", True),
    ("unknown", False), ("", False), ("2026-08-30T00:00:00Z", False),
])
def test_only_parseable_dates_are_shown(tmp_path, raw, shown):
    """소스마다 날짜 정밀도가 다르다. 형식에 안 맞으면 지어내지 말고 뺀다."""
    html = build(tmp_path, [entry(date=raw)])["index.html"]
    assert (f"<span>{raw}</span>" in html) is shown


def test_summary_failure_is_stated_not_hidden(tmp_path):
    """초록이 있는데 요약이 실패하면, 빈 카드가 아니라 이유를 보여준다."""
    e = entry(abstract="있음", summary=None)
    html = build(tmp_path, [e])["index.html"]
    assert "요약 생성 실패" in html
    assert "초록 미확보" not in html


def test_summary_failure_does_not_drop_the_paper(tmp_path):
    """요약 실패로 논문을 잃지 않는다 — 리스트업이 목적이다 (원칙 2)."""
    html = build(tmp_path, [entry(title="Kept Anyway", abstract="있음", summary=None)])["index.html"]
    assert "Kept Anyway" in html


def test_summary_failure_counted_in_footer(tmp_path):
    es = [entry(aid="2601.1"),
          entry(aid="2601.2", abstract="있음", summary=None),
          entry(aid="2601.3", abstract=None, summary=None)]
    html = build(tmp_path, es)["index.html"]
    assert "요약 실패" in html and "초록 미확보 1건" in html
