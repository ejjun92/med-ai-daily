"""원장 — published(식별자 집합) / deferred(버전 게이트).

JSONL append-only + 월별 샤딩을 쓴다. 단일 JSON 배열을 통째로 재작성하면
`git pull --rebase` 시 거의 모든 줄에서 충돌한다 — 사용자가 config.py를
수정해 커밋하는 상황이 예상되므로(계획 R-7) 실제로 부딪힌다.
한 줄 추가는 깔끔하게 rebase되고 git 델타도 거의 공짜다.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Iterable, Iterator, Optional

import config
from models import Paper

PUBLISHED_DIR = os.path.join("data", "published")
DEFERRED_DIR = os.path.join("data", "deferred")

_ID_FIELDS = ("arxiv_id", "doi", "pmid", "norm_title")


def _shard(root: str, day: str) -> str:
    return os.path.join(root, f"{day[:7]}.jsonl")   # YYYY-MM.jsonl


def _iter_jsonl(root: str) -> Iterator[dict]:
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(root, name), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue      # 손상된 한 줄이 원장 전체를 막지 않는다


def _append(root: str, day: str, records: Iterable[dict]) -> int:
    records = list(records)
    if not records:
        return 0
    os.makedirs(root, exist_ok=True)
    with open(_shard(root, day), "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


# ─────────────────────────────────────────────────────────────────
# published — 식별자 집합 매칭
# ─────────────────────────────────────────────────────────────────

class PublishedLedger:
    """게시 완료 논문. 넷 중 하나라도 일치하면 중복으로 본다 (계획 D-4).

    단일 키(arXiv ID 또는 pmid:)만 쓰면, 프리프린트로 게시한 논문이 몇 달 뒤
    저널 게재분으로 PubMed/S2에 색인될 때 다른 키가 되어 재게시된다.
    arXiv 윈도우가 4일이라 같은 실행 내 교차 중복제거로도 못 잡는다 —
    AC-2가 예외가 아니라 정상 경로에서 깨진다.
    """

    def __init__(self, root: str = PUBLISHED_DIR):
        self.root = root
        self._index: dict[str, set[str]] = {k: set() for k in _ID_FIELDS}
        self._count = 0
        for rec in _iter_jsonl(root):
            self._add_to_index(rec)

    def _add_to_index(self, rec: dict) -> None:
        for k in _ID_FIELDS:
            if (v := rec.get(k)):
                self._index[k].add(v)
        self._count += 1

    def __len__(self) -> int:
        return self._count

    def is_published(self, paper: Paper) -> bool:
        ident = paper.identity()
        for k in _ID_FIELDS:
            v = ident.get(k)
            if v and v in self._index[k]:
                return True
        return False

    def add(self, papers: Iterable[Paper], day: str) -> int:
        recs = []
        for p in papers:
            rec = {"published_at": day, **p.identity(), "title": p.title, "source": p.source}
            recs.append(rec)
            self._add_to_index(rec)
        return _append(self.root, day, recs)


# ─────────────────────────────────────────────────────────────────
# deferred — 프롬프트 버전 게이트
# ─────────────────────────────────────────────────────────────────

@dataclass
class DeferredRecord:
    primary_id: str
    first_seen: str            # YYYY-MM-DD
    prompt_version: str
    reason: str                # not_relevant | quota | truncated | undecided
    title: str = ""
    payload: dict | None = None   # 재분류에 필요한 최소 메타


class DeferredLedger:
    """탈락 논문. TTL 내에 **분류 프롬프트가 바뀌었을 때만** 재진입한다.

    무조건 매일 재진입시키면 후보 풀이 TTL에 비례해 선형 증가하고
    (300~900건/일 × 14일), 상한이 상시 발동하면서 오래된 것부터 잘린다.
    그런데 오래된 것이 바로 회수 대상이므로 회수 기제가 장식이 된다.
    게다가 같은 프롬프트로 재분류하면 같은 판정이 나오므로 순수 낭비다.
    """

    def __init__(self, root: str = DEFERRED_DIR):
        self.root = root
        self._records: list[DeferredRecord] = [
            DeferredRecord(
                primary_id=r.get("primary_id", ""),
                first_seen=r.get("first_seen", ""),
                prompt_version=r.get("prompt_version", ""),
                reason=r.get("reason", ""),
                title=r.get("title", ""),
                payload=r.get("payload"),
            )
            for r in _iter_jsonl(root)
        ]
        # 같은 primary_id가 여러 번 들어올 수 있다 — 최신 기록만 유효
        self._latest: dict[str, DeferredRecord] = {}
        for r in self._records:
            if r.primary_id:
                self._latest[r.primary_id] = r

    def __len__(self) -> int:
        return len(self._latest)

    def defer(self, items: Iterable[tuple[Paper, str]], day: str,
              prompt_version: str | None = None) -> int:
        pv = prompt_version or config.CLASSIFY_PROMPT_VERSION
        recs = []
        for paper, reason in items:
            rec = DeferredRecord(primary_id=paper.primary_id, first_seen=day,
                                 prompt_version=pv, reason=reason, title=paper.title)
            self._latest[rec.primary_id] = rec
            recs.append({
                "primary_id": rec.primary_id, "first_seen": day,
                "prompt_version": pv, "reason": reason, "title": paper.title,
            })
        return _append(self.root, day, recs)

    def active(self, today: str, prompt_version: str | None = None,
               force: bool = False, limit: int | None = None) -> list[DeferredRecord]:
        """재진입 대상. 기본은 프롬프트 버전이 바뀐 것만.

        force=True (--replay-deferred)면 버전과 무관하게 TTL 내 전부.
        """
        pv = prompt_version or config.CLASSIFY_PROMPT_VERSION
        lim = config.DEFERRED_DAILY_MAX if limit is None else limit
        cutoff = (date.fromisoformat(today) - timedelta(days=config.DEFERRED_TTL_DAYS)).isoformat()

        out = [r for r in self._latest.values()
               if r.first_seen >= cutoff and (force or r.prompt_version != pv)]
        out.sort(key=lambda r: r.first_seen)      # 오래된 것부터
        return out[:lim]

    def expired(self, today: str) -> list[DeferredRecord]:
        """TTL이 지나 영구 폐기되는 항목. 무증상 누수 방지를 위해 센다."""
        cutoff = (date.fromisoformat(today) - timedelta(days=config.DEFERRED_TTL_DAYS)).isoformat()
        return [r for r in self._latest.values() if r.first_seen < cutoff]
