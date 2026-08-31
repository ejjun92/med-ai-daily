"""관련성·축·카테고리·별점 판정.

LLM은 논문마다 독립적으로 판정만 한다. 쿼터 배분은 selection.py가 맡는다
(계획 D-1) — LLM에게 쿼터를 맡기면 비결정적이고 테스트가 불가능해진다.
"""
from __future__ import annotations

import config
from models import Classification, Paper

SCHEMA = {
    "type": "object",
    "properties": {
        "is_relevant": {"type": "boolean"},
        "axis": {"type": "string", "enum": list(config.AXIS_KEYS)},
        "category_id": {"type": "string", "enum": list(config.CATEGORY_IDS)},
        # 구조화 출력은 minimum/maximum을 스키마에서 제거하므로 enum을 쓴다
        "stars": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "rationale": {"type": "string"},
    },
    "required": ["is_relevant", "axis", "category_id", "stars", "rationale"],
    "additionalProperties": False,
}


def _taxonomy_block() -> str:
    lines = []
    for a in config.AXES:
        lines.append(f"\n■ {a.key} ({a.label})")
        for c in config.CATEGORIES:
            if c.axis == a.key:
                lines.append(f"   - {c.id}: {c.name} — {c.description}")
    return "\n".join(lines)


def _star_block() -> str:
    return "\n".join(f"   {n}점: {d}" for n, d in sorted(config.STAR_RUBRIC.items(), reverse=True))


PROMPT = """당신은 의료 AI 연구자의 논문 큐레이터다. 아래 논문이 이 연구자의
관심 범위에 드는지 판정하고, 든다면 축과 카테고리를 배정하고 중요도를 매겨라.

# 축과 카테고리 (반드시 이 중에서 고른다)
{taxonomy}

# 중요도 기준
{stars}

# 판정 지침
- **재현율을 우선한다.** 애매하면 관련 있음(true)으로 판정하라. 놓치는 것이
  잡음보다 비싸다. 다만 명백히 무관한 분야(예: 자연어처리 일반, 추천시스템,
  금융, 로보틱스 조작)는 false로 한다.
- **dl_methodology는 의료와 무관해도 관련 있음이다.** 일반 컴퓨터비전·멀티모달
  기반모델(open-vocabulary 검출, 분할, VLM, diffusion, 표현학습, 불확실성,
  해석가능성)은 의료 데이터를 안 써도 이 연구자가 추적하는 대상이다.
  (실측: 제외 문구를 늘리자 DINO-X와 YOLO-World가 "의료와 무관"으로 탈락했다.)
- 축은 **논문의 주된 기여**로 정한다. 의료 데이터를 썼다는 이유만으로
  medical_imaging에 넣지 말 것 — 기여가 일반적 방법론이면 dl_methodology다.
- **brain_decoding(뇌신호 AI)의 경계는 "목적"으로 가른다.**
  **판별 기준: 뇌 신호에서 정보를 읽어내거나 뇌 신호의 표현을 배우는 것이
  목적인가? 아니면 질환을 진단하거나 뇌 영상 자체를 처리하는 것이 목적인가?**

  포함한다 — fMRI·EEG·MEG에서 본 이미지·문장·의미를 복원, 인지·운동 의도
  디코딩, BCI·motor imagery·neurofeedback, EEG/fMRI 표현학습·자기지도학습·
  foundation model, 피험자 간 일반화, 뇌 표현과 딥러닝 모델 표현의 정렬·비교.

  제외한다 (이쪽은 medical_imaging이다) —
    · 알츠하이머·자폐·정신질환 등 **질환 진단·예후 예측**
    · 뇌 영상 **분할·정합·합성·초해상·harmonization·화질 개선**
    · 뇌 나이·발달·노화 추정, 커넥톰으로 **환자군을 분류**하는 연구
    · **뇌 네트워크·커넥톰·연결성(connectivity) 연구 일체** — "표현학습",
      "features", "foundation model"이라고 쓰여 있어도 대상이 원신호가 아니라
      거기서 뽑아낸 연결성 그래프면 여기가 아니다
    · **신경영상 메타분석**, 소스 국소화를 제외한 신호 전처리
    · **환자·질환·임상 코호트·다기관 데이터**가 연구 목적에 등장하는 경우
  뇌를 다룬다는 것만으로는 부족하다. 기여가 일반 방법론이면 dl_methodology다.

- 수술 영상·수술 로봇·술기 평가는 surgical_video다. 수술과 무관한 임상 비디오
  (보행 분석, 내시경 진단 등)는 여기 넣지 않는다.
- category_id는 반드시 선택한 axis에 속한 것이어야 한다.
- rationale은 한국어 한 문장.

# 논문
제목: {title}
{venue_line}초록: {abstract}

JSON만 출력하라."""


def build_prompt(paper: Paper) -> str:
    abstract = (paper.abstract or "").strip()
    if not abstract:
        # 초록 미확보 — 제목만으로 판정한다. 일부 소스(Springer LNCS 등)는
        # 초록을 공개 API로 주지 않는다. 감추지 않고 싣는 것이 리스트업 목적에 맞다.
        abstract = "(초록 미확보 — 제목과 학회 정보만으로 판정하라)"
    elif len(abstract) > 2200:
        abstract = abstract[:2200] + " …"
    venue_line = f"학회/저널: {paper.venue.tag}\n" if paper.venue else ""
    return PROMPT.format(taxonomy=_taxonomy_block(), stars=_star_block(),
                         title=paper.title, abstract=abstract, venue_line=venue_line)


def parse(raw: dict | None) -> Classification:
    """판정 불가는 버리지 않고 '관련 있음, 별점 미정'으로 보류한다 (원칙 1)."""
    if not raw:
        return Classification(is_relevant=True, undecided=True)
    axis = raw.get("axis")
    cat = raw.get("category_id")
    # 축과 카테고리가 어긋나면 카테고리 쪽을 신뢰한다 (enum으로 강제된 값)
    if cat in config.CATEGORY_BY_ID:
        axis = config.CATEGORY_BY_ID[cat].axis
    elif axis not in config.AXIS_KEYS:
        return Classification(is_relevant=True, undecided=True)
    stars = raw.get("stars")
    return Classification(
        is_relevant=bool(raw.get("is_relevant")),
        axis=axis, category_id=cat,
        stars=stars if isinstance(stars, int) and 1 <= stars <= 5 else None,
        rationale=str(raw.get("rationale", ""))[:300],
    )


def classify(llm, papers: list[Paper], log=print) -> list[Classification]:
    if not papers:
        return []
    raws = llm.chat_json([build_prompt(p) for p in papers], SCHEMA,
                         config.CLASSIFY_MAX_TOKENS)
    out = [parse(r) for r in raws]
    rel = sum(1 for c in out if c.is_relevant and not c.undecided)
    und = sum(1 for c in out if c.undecided)
    log(f"  [classify] {len(papers)}건 → 관련 {rel} / 무관 {len(out)-rel-und} / 판정불가 {und}")
    return out
