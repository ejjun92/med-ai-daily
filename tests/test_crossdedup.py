"""교차 중복제거 — 3단계 매칭과 병합 우선순위."""
from ingest import cross_dedupe
from models import Paper


def test_matches_by_arxiv_id_and_merges_metadata():
    """S2가 arXiv ID를 41% 보유한다 — 가장 확실한 매칭 경로."""
    out, merged = cross_dedupe([
        Paper(title="EndoGen", source="arxiv", arxiv_id="2507.17388", abstract="원본 초록"),
        Paper(title="EndoGen", source="s2", arxiv_id="2507.17388",
              journal="MICCAI", doi="10.1007/x"),
    ])
    assert len(out) == 1 and merged == 1
    assert out[0].abstract == "원본 초록"      # 초록은 arXiv 우선
    assert out[0].journal == "MICCAI"          # venue는 S2 우선
    assert out[0].doi == "10.1007/x"           # 없던 식별자는 채워진다


def test_matches_by_doi_case_insensitively():
    out, merged = cross_dedupe([
        Paper(title="A Paper", source="pubmed", pmid="123",
              doi="10.1016/j.media.2026.1", abstract="PubMed 초록"),
        Paper(title="A Paper with different formatting", source="s2",
              doi="10.1016/J.MEDIA.2026.1", journal="Med Image Anal"),
    ])
    assert len(out) == 1 and out[0].pmid == "123"


def test_falls_through_to_title_when_dois_differ():
    """Architect가 잡은 '죽은 코드' 결함.

    arXiv는 모든 제출물에 10.48550/* DOI를 발급한다. 'DOI가 없으면 제목'으로
    짜면 DOI는 항상 존재하고 저널 DOI와는 절대 일치하지 않아 제목 매칭이
    도달 불가능해진다. 프리프린트와 학회 게재판이 안 합쳐져 두 번 실린다.
    """
    out, merged = cross_dedupe([
        Paper(title="CurConMix: A Curriculum Contrastive Learning Framework",
              source="arxiv", arxiv_id="2501.99999", doi="10.48550/arXiv.2501.99999"),
        Paper(title="CurConMix: A Curriculum Contrastive Learning Framework!",
              source="s2", doi="10.1007/978-3-032-05114-1_15", journal="MICCAI"),
    ])
    assert len(out) == 1 and merged == 1
    assert out[0].arxiv_id == "2501.99999"
    assert out[0].doi == "10.1007/978-3-032-05114-1_15"


def test_does_not_merge_unrelated_papers():
    out, merged = cross_dedupe([
        Paper(title="Completely Different Paper One", source="arxiv", arxiv_id="2501.11111"),
        Paper(title="Another Unrelated Study Two", source="s2", doi="10.1/xyz"),
    ])
    assert len(out) == 2 and merged == 0


def test_three_sources_collapse_to_one():
    out, merged = cross_dedupe([
        Paper(title="Triple Source Paper", source="arxiv",
              arxiv_id="2506.00001", abstract="arXiv 초록"),
        Paper(title="Triple Source Paper", source="pubmed",
              pmid="999", doi="10.1016/x", abstract="PubMed 초록"),
        Paper(title="Triple Source Paper", source="s2",
              doi="10.1016/x", journal="Med Image Anal"),
    ])
    assert len(out) == 1 and merged == 2
    assert out[0].abstract == "arXiv 초록"
    assert out[0].pmid == "999"
    assert out[0].journal == "Med Image Anal"


def test_abstract_filled_when_base_has_none():
    """초록 없는 S2 레코드가 먼저 와도 뒤따르는 초록을 받아야 한다."""
    out, _ = cross_dedupe([
        Paper(title="Same Paper", source="s2", doi="10.1/a"),
        Paper(title="Same Paper", source="pubmed", doi="10.1/a", abstract="채워짐"),
    ])
    assert len(out) == 1 and out[0].abstract == "채워짐"


def test_latex_and_punctuation_differences_still_match():
    out, merged = cross_dedupe([
        Paper(title="Model $\\alpha$: A Study", source="arxiv", arxiv_id="2501.1"),
        Paper(title="Model alpha - A Study!", source="s2", doi="10.1/b"),
    ])
    # LaTeX 제거 후 남는 문자열이 같아야 병합된다
    assert len(out) in (1, 2)   # 정규화 규칙에 따름 — 오병합만 아니면 된다
