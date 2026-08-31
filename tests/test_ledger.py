"""원장 — 재게시 차단과 보류 풀 유계성."""
from datetime import date, timedelta

import pytest

import config
from ledger import DeferredLedger, PublishedLedger

# 실제 프롬프트 버전이 무엇이든 '바뀐 버전'이어야 한다.
# "v2"를 하드코딩했다가 config가 v2로 올라가면서 테스트 4개가 조용히
# 의미를 잃었다 — 재진입이 0건인데 통과하는 상태가 됐다.
CHANGED = config.CLASSIFY_PROMPT_VERSION + "-changed"
from models import Paper, normalize_arxiv_id, normalize_doi, normalize_title


@pytest.fixture
def pub(tmp_path):
    return PublishedLedger(str(tmp_path / "published"))


@pytest.fixture
def deferred(tmp_path):
    return DeferredLedger(str(tmp_path / "deferred"))


# ── 정규화 ────────────────────────────────────────────────────
@pytest.mark.parametrize("raw,want", [
    ("2401.12345v2", "2401.12345"),
    ("2401.12345", "2401.12345"),
    ("http://arxiv.org/abs/2410.00263v2", "2410.00263"),
])
def test_version_suffix_stripped(raw, want):
    """정규화하지 않으면 개정판이 매번 신규로 잡혀 AC-2가 깨진다."""
    assert normalize_arxiv_id(raw) == want


def test_arxiv_datacite_doi_excluded():
    """arXiv는 모든 제출물에 이 DOI를 발급한다. 매칭에 쓰면 제목 경로가 죽는다."""
    assert normalize_doi("10.48550/arXiv.2401.1") is None
    assert normalize_doi("https://doi.org/10.1016/J.MEDIA.1") == "10.1016/j.media.1"


def test_title_normalization_ignores_latex_and_punctuation():
    a = normalize_title("ReSurgSAM2: $\\alpha$ Tracking!")
    b = normalize_title("resurgsam2 tracking")
    assert a == b


# ── published: 식별자 집합 ────────────────────────────────────
def test_blocks_republish_across_source_and_time(pub):
    """프리프린트 게시 후 몇 달 뒤 저널 게재분이 다른 소스로 들어오는 정상 경로."""
    pre = Paper(title="CurConMix: A Curriculum Contrastive Learning Framework",
                source="arxiv", arxiv_id="2501.99999")
    pub.add([pre], "2025-01-10")

    later = Paper(title="CurConMix: A Curriculum Contrastive Learning Framework",
                  source="s2", doi="10.1007/978-3-032-05114-1_15")
    assert later.primary_id != pre.primary_id      # 키가 다르다
    assert pub.is_published(later)                 # 그래도 잡혀야 한다


def test_unrelated_paper_passes(pub):
    pub.add([Paper(title="One", source="arxiv", arxiv_id="2501.00001")], "2025-01-10")
    assert not pub.is_published(Paper(title="Totally Other", source="s2", doi="10.1/x"))


def test_survives_reload(tmp_path):
    root = str(tmp_path / "published")
    p = Paper(title="Persisted", source="arxiv", arxiv_id="2501.00002")
    PublishedLedger(root).add([p], "2025-02-01")
    assert PublishedLedger(root).is_published(p)


# ── deferred: 버전 게이트 ─────────────────────────────────────
def _fill(dl, days: int, per_day: int, start=date(2025, 3, 1)):
    for i in range(days):
        day = (start + timedelta(days=i)).isoformat()
        papers = [Paper(title=f"p{i}-{j}", source="arxiv", arxiv_id=f"25{i:02d}.{j:05d}")
                  for j in range(per_day)]
        dl.defer([(p, "not_relevant") for p in papers], day)
    return (start + timedelta(days=days - 1)).isoformat()


def test_no_reentry_while_prompt_unchanged(deferred):
    """같은 프롬프트로 재분류하면 같은 판정이 나온다 — 순수 낭비다."""
    last = _fill(deferred, days=5, per_day=50)
    assert deferred.active(last) == []


def test_reentry_only_after_prompt_change(deferred):
    last = _fill(deferred, days=5, per_day=50)
    got = deferred.active(last, prompt_version=CHANGED)
    assert 0 < len(got) <= config.DEFERRED_DAILY_MAX


def test_pool_stays_bounded_over_20_days(deferred):
    """무조건 매일 재진입이면 풀이 TTL에 비례해 선형 증가하고, 상한이 상시
    발동하면서 오래된 것(=회수 대상)부터 잘려 회수 기제가 장식이 된다."""
    last = _fill(deferred, days=20, per_day=240)
    assert len(deferred) > 4000                                   # 누적은 크지만
    assert deferred.active(last) == []                            # 재진입은 0
    assert len(deferred.active(last, prompt_version=CHANGED)) <= config.DEFERRED_DAILY_MAX


def test_reentry_is_oldest_first(deferred):
    last = _fill(deferred, days=10, per_day=50)
    got = deferred.active(last, prompt_version=CHANGED)
    assert got == sorted(got, key=lambda r: r.first_seen)


def test_ttl_expiry_is_countable(deferred):
    """TTL 만료는 영구 소실이므로 조용히 사라지면 안 된다."""
    last = _fill(deferred, days=20, per_day=10)
    assert len(deferred.expired(last)) > 0


def test_force_replay_ignores_version(deferred):
    last = _fill(deferred, days=5, per_day=50)
    assert len(deferred.active(last, force=True)) > 0


# ── 보류분 재분류에 필요한 payload (Phase 6) ──────────────────
def test_deferred_record_carries_enough_to_reclassify(deferred):
    """payload가 없으면 --replay-deferred가 동작하지 않는 빈 플래그가 된다.

    재진입 대상을 '골라낼' 수는 있어도 그 논문을 '다시 판정할' 수는 없다 —
    제목·초록이 없기 때문이다.
    """
    p = Paper(title="MIRAGE: fMRI to image", source="arxiv",
              arxiv_id="2605.17198", abstract="We present MIRAGE...",
              announced_date="2026-05-16", url="https://arxiv.org/abs/2605.17198")
    deferred.defer([(p, "not_relevant")], "2026-08-31")

    rec = DeferredLedger(deferred.root).active("2026-08-31", prompt_version=CHANGED)[0]
    back = Paper.from_payload(rec.payload)
    assert back.title == p.title
    assert back.abstract == p.abstract
    assert back.primary_id == p.primary_id


def test_payload_omits_bulky_fields(deferred):
    """원장은 매일 push된다. 저자 목록과 코멘트 원문까지 실으면 무거워진다."""
    p = Paper(title="T", source="arxiv", arxiv_id="1", abstract="a",
              authors=[f"Author {i}" for i in range(30)],
              raw_comments="Accepted to MICCAI 2025. " * 20)
    payload = p.to_payload()
    assert "authors" not in payload and "raw_comments" not in payload


def test_old_records_without_payload_still_load(deferred, tmp_path):
    """payload 이전에 쌓인 기록이 있어도 터지지 않는다 — 복원만 못 할 뿐이다."""
    import json, os
    root = str(tmp_path / "old")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "2026-08.jsonl"), "w") as f:
        f.write(json.dumps({"primary_id": "arxiv:1", "first_seen": "2026-08-31",
                            "prompt_version": "v1", "reason": "quota",
                            "title": "T"}) + "\n")
    recs = DeferredLedger(root).active("2026-08-31", prompt_version=CHANGED)
    assert len(recs) == 1 and recs[0].payload is None


# ── 샤드 정리 (Phase 6, 실측으로 발견) ────────────────────────
def test_prune_removes_only_shards_past_ttl(deferred, tmp_path):
    """__init__이 모든 샤드를 읽는다. 안 지우면 로드 시간이 무한히 자란다.

    하루 700건 × 1년이면 25만 건을 매 실행마다 파싱한다.
    """
    import os
    os.makedirs(deferred.root, exist_ok=True)
    for month in ("2026-03", "2026-06", "2026-07", "2026-08"):
        with open(os.path.join(deferred.root, f"{month}.jsonl"), "w") as f:
            f.write('{"primary_id": "x", "first_seen": "%s-01", '
                    '"prompt_version": "v1", "reason": "quota"}\n' % month)

    DeferredLedger(deferred.root).prune("2026-08-31", log=lambda *_: None)
    left = sorted(n for n in os.listdir(deferred.root) if n.endswith(".jsonl"))
    # TTL 14일 + 여유 31일 → 2026-07 이후만 남는다
    assert left == ["2026-07.jsonl", "2026-08.jsonl"]


def test_prune_is_safe_on_empty_dir(tmp_path):
    assert DeferredLedger(str(tmp_path / "nope")).prune("2026-08-31",
                                                        log=lambda *_: None) == 0


def test_quota_records_carry_no_payload(deferred):
    """쿼터 탈락분은 같은 프롬프트로 다시 돌려도 같은 판정이다.

    초록까지 보관하면 연 0.31GB가 git에 쌓인다 (실측: 하루 607건 중 556건이 quota).
    """
    p = Paper(title="T", source="arxiv", arxiv_id="1", abstract="긴 초록" * 200)
    deferred.defer([(p, "quota")], "2026-08-31")
    rec = DeferredLedger(deferred.root).active("2026-08-31", prompt_version=CHANGED)[0]
    assert rec.payload is None
    assert rec.title == "T", "식별 정보는 남는다 — 본 적 있는지 추적은 된다"


def test_not_relevant_records_keep_payload(deferred):
    """프롬프트를 고쳤을 때 회수할 대상은 바로 이쪽이다."""
    p = Paper(title="T", source="arxiv", arxiv_id="1", abstract="초록")
    deferred.defer([(p, "not_relevant")], "2026-08-31")
    rec = DeferredLedger(deferred.root).active("2026-08-31", prompt_version=CHANGED)[0]
    assert rec.payload and rec.payload["abstract"] == "초록"
