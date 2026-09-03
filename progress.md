# med-ai-daily — 진행 상황

> 대화가 끊겨도 이 문서만 읽으면 이어서 작업할 수 있도록 정리한다.
> 마지막 갱신: 2026-09-03

---

## 1. 이게 뭔가

전은지(ejjun92)의 연구 주제에 맞춘 **매일 자동 갱신 논문 다이제스트**.
`https://gist-ailab.github.io/gail-daily-news/research_latest.html` 를 모델로 삼았다.

- 사이트: <https://ejjun92.github.io/med-ai-daily/>
- 저장소: <https://github.com/ejjun92/med-ai-daily> (public)
- 서버 작업 경로: `/data1/server1/user/ejjun/med-ai-daily`
- 파이썬: `/data1/server1/miniconda3/envs/meddaily/bin/python`

**제품의 핵심은 리스트업 품질이다.** 관련 논문을 놓치지 않는 것, 축·카테고리
분류, 중요도 별점, 학회 라벨이 값어치다. 한국어 요약은 "열어볼지 판단할 정도"면
충분하고, 요약이 실패해도 논문은 싣는다.

---

## 2. ⚠️ 지금 당장 해야 할 일

**자동 실행이 2026-09-02, 09-03 이틀 연속 실패했다. 사이트가 09-01에 멈춰 있다.**

원인: arXiv가 **HTTP 429**(IP 단위 호출 제한)를 냈는데 `arxiv.py`만 예외를 밖으로
던져 파이프라인 전체가 죽었다. PubMed·S2는 실패해도 건너뛰고 진행하는데 arXiv만
달랐다. 전날(09-01) 소속 기관 조사를 하며 arXiv를 너무 많이 호출한 것이 발단이다.

**코드는 고쳤다** (아래 3-6). 남은 일:

```bash
cd /data1/server1/user/ejjun/med-ai-daily
# 1) arXiv 차단이 풀렸는지 확인 (가볍게 한 번만)
/data1/server1/miniconda3/envs/meddaily/bin/python src/pipeline.py run --dry-run

# 2) 풀렸으면 실제 실행 (약 10~20분, 30일 창이지만 대부분 기판정 제외)
./scripts/run_daily.sh
```

`--dry-run`이 또 429면 하루 더 기다린다. 이제는 실패해도 PubMed·S2로 발행되므로
사이트가 멈추지는 않는다.

---

## 3. 지금까지 만든 것

### 3-1. 파이프라인

```
수집 → 교차 중복제거 → 원장 대조 → 분류 → 선별 → 요약 → 렌더 → 원장 기록 → git push
```

| 파일 | 역할 |
|---|---|
| `src/config.py` | 축·카테고리·쿼터·소스·모델 등 모든 설정 (70개 항목) |
| `src/models.py` | `Paper` `Venue` `Classification` `Summary` `Entry`, 직렬화 |
| `src/sources/arxiv.py` | arXiv API. 30일 창, 공백 메우기, 429 백오프 |
| `src/sources/pubmed.py` | PubMed E-utilities. 저널 8종, 30일 창 |
| `src/sources/semanticscholar.py` | S2. 학회 8종 × 최근 2개 연도 |
| `src/sources/frontier.py` | HF Daily Papers. 별도 레인 |
| `src/ingest.py` | 소스 통합, 교차 중복제거, 소스 실패 격리 |
| `src/ledger.py` | published/deferred 원장 (JSONL 월별 샤딩) |
| `src/venue.py` | 학회 라벨 추출·별점 부스트 |
| `src/llm.py` | vLLM 온디맨드, GPU 선택, 깨진 JSON 복구 |
| `src/classify.py` | 관련성·축·카테고리·별점 판정 |
| `src/summarize.py` | 한국어 요약 + 태그, 규격 검증 |
| `src/selection.py` | 축 비율대로 배분 (순수 함수) |
| `src/render.py` | Jinja2 렌더, `PageMeta` |
| `src/pipeline.py` | 전체 연결, `run`/`render` 서브커맨드 |
| `templates/*.j2` | base / digest / archive_index |
| `scripts/run_daily.sh` | cron 진입점 (flock·로그·push) |
| `scripts/install_cron.sh` | crontab 등록 |

### 3-2. 주제 축 (합 100%)

| 축 | 비중 | 목표 | 카테고리 |
|---|---|---|---|
| 🧠 `brain_decoding` 뇌신호 AI | 30% | 15 | 5 |
| 🔬 `surgical_video` 수술영상 | 26% | 13 | 4 |
| ⚙️ `dl_methodology` 딥러닝 방법론 | 17% | 8 | 6 |
| 📈 `ehr_clinical` EHR·임상 시계열 | 14% | 7 | 4 |
| 🩺 `medical_imaging` 의료영상 AI | 13% | 6 | 3 |

하루 40~60건(목표 50). 카테고리 22개. **부족하면 억지로 채우지 않는다.**

별도 레인: **🔭 Frontier AI** 6건 — 축 쿼터와 경쟁하지 않는다.

### 3-3. 수집 범위

| 소스 | 기준 | 창 | 상한 |
|---|---|---|---|
| arXiv | `submittedDate` | 30일 | 12,000 |
| PubMed | `EDAT` | 30일 | 2,000 |
| Semantic Scholar | 발행 **연도** | 최근 2개 연도 | 3,000 (상한 도달 중) |
| HF Daily Papers | `publishedAt` | 30일 | 6 |

- arXiv 카테고리: `cs.CV cs.LG cs.AI eess.IV q-bio.NC`
- PubMed 저널 8종: MedIA, IEEE TMI, TPAMI, Radiology:AI, IJCARS, **JAMIA, J Biomed Inform, npj Digit Med**
- S2 학회 8종: MICCAI, IPCAI, CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR

### 3-4. 추론

- 모델 `Qwen/Qwen2.5-32B-Instruct-AWQ` (로컬 vLLM, API 비용 0)
- **GPU 한 장만** 쓴다. `0 → 1 → 2 → 3` 순으로 여유 24GB 이상인 첫 장을 잡고,
  전부 점유 중이면 5분씩 3번 기다렸다 **그날은 건너뛴다** (연구 작업 우선)
- 점유량 약 46GB (`GPU_MEMORY_UTILIZATION=0.85`), 끝나면 반드시 반납
- `temperature=0` + `seed=42` — 같은 입력이면 같은 페이지
- 분류는 후보 전체, **요약은 선별된 40~60건에만** (비용 30배 차이)
- `CLASSIFY_PROMPT_VERSION = "v5"` — 이 값을 올리면 보류 원장이 전량 재분류된다

### 3-5. 자동화

```
0 12 * * * /data1/server1/user/ejjun/med-ai-daily/scripts/run_daily.sh
```

**12:00 KST.** 처음엔 09:40이었으나 그게 00:40 UTC라 arXiv 발표·재색인 구간과
겹쳐 실제로 arXiv가 0건을 반환했다. 세 시간 뗐다.

`run_daily.sh`가 지키는 것: flock(겹쳐 실행 금지), 실패 시 push 안 함,
`logs/daily-YYYY-MM-DD.log` 30일 보관.

### 3-6. 디자인

원본 사이트의 **뉴모피즘** 시스템을 재현했다.

- Google Fonts: Space Mono(제목) + Noto Sans KR(본문) + JetBrains Mono(메타)
- 청록 `#006666`, 별점 주황 `#FE9900`, 다크 푸터 / 다크모드는 `#0D1117`에서 시작
- 그라데이션 hero + 통계 배지
- **상단** 지난 뉴스 카드 — `<input type="date">` 달력 선택기 + `기록 열기 ↗`
- 접이식 "분류·중요도·수집 기준" 안내 (내용을 `config.py`에서 생성)
- 항목 카드: 소스 배지 + 날짜 + 별점 + `taxonomy-chip` 태그
- 별 `STAR_HIGHLIGHT_MIN`(=5) 이상이면 제목색 테두리 + 그림자 + 붉은 별점
- 다크모드 토글(우하단, 3단: 시스템 자동 / 라이트 / 다크)
- 푸터: **데이터 기준일 + 생성 시각만** (사용자 요청으로 나머지 제거)

### 3-7. 테스트 179개 (모델·네트워크 없이 2초)

| 파일 | 개수 | 지키는 것 |
|---|---|---|
| `test_render.py` | 36 | 이스케이프, 테마 3상태, 아카이브, 날짜 선택기 |
| `test_ledger.py` | 22 | 원장 매칭, 버전 게이트, payload, 샤드 정리 |
| `test_classify_summarize.py` | 18 | 스키마-설정 일치, 축 경계, 요약 검증 |
| `test_pipeline.py` | 15 | 단계 순서, 원장 시점, 저장소 오염 방지 |
| `test_gpu_pick.py` | 10 | GPU 선택·대기·포기 |
| `test_selection.py` | 10 | 쿼터 배분, 결정성 |
| `test_llm_salvage.py` | 9 | 깨진 JSON 복구 (실제 폭주 출력 픽스처) |
| `test_config.py` | 8 | 설정 정합성 |
| `test_crossdedup.py` | 7 | 교차 중복제거 |
| `test_venue.py` | 6 | 학회 라벨 |
| `test_source_failure.py` | 4 | **한 소스가 죽어도 발행** (09-02 사고) |
| `test_arxiv_window.py` | 4 | 창 공백 메우기 |

```bash
/data1/server1/miniconda3/envs/meddaily/bin/python -m pytest tests/ -q
```

---

## 4. 자주 쓰는 명령

```bash
cd /data1/server1/user/ejjun/med-ai-daily
PY=/data1/server1/miniconda3/envs/meddaily/bin/python

$PY src/pipeline.py run                    # 오늘치 생성 → docs/
$PY src/pipeline.py run --dry-run          # 수집까지만, 모델 안 띄움
$PY src/pipeline.py run --date 2026-09-01  # 특정 날짜
$PY src/pipeline.py run --ignore-seen --out-dir /tmp/x   # 원장 안 건드리고 재생성
$PY src/pipeline.py render                 # 페이지만 다시 그림 (GPU 불필요)
$PY src/pipeline.py render --all           # 저장된 모든 날짜 재렌더
./scripts/run_daily.sh                     # cron이 부르는 것과 동일
```

**디자인을 고쳤으면 `render`만 다시 하면 된다.** 선별 결과가
`data/entries/YYYY-MM-DD.json`에 있어 GPU가 필요 없다 (몇 초).

⚠️ `render --all`은 캐시가 있는 날짜만 제대로 그린다. 캐시가 없는 옛 날짜를
덮어쓰지 않도록 주의 (실제로 8-31 아카이브를 가짜 데이터로 덮은 적 있다).

---

## 5. 실측으로 배운 것 (추측하지 말고 이 숫자를 믿을 것)

### 수집·공급

| 사실 | 숫자 |
|---|---|
| arXiv 30일 창 실제 분량 | 9,294건 |
| 진짜 brain decoding 논문 | **arXiv 전체에서 월 5편** (하루 0.17편) |
| 뇌 관련 61건 중 진짜 디코딩 | 1건 (나머지는 알츠하이머 진단·연결성·분할) |
| EHR 계열 30일 공급 | 60~100편 (하루 2~3편) — 한 축을 지탱하기 충분 |
| MICCAI PubMed 색인률 | 6% → S2를 세 번째 소스로 추가한 이유 |
| S2 초록 확보율 | 45% → 초록 미확보도 게시(“초록 미확보” 표시) |
| arXiv venue 라벨 적중 | 신규 수집분에서 1% → 라벨의 실제 출처는 S2 |

### 성능

| 단계 | 시간 |
|---|---|
| 수집 (3소스) | 3~4분 |
| 모델 로드 | 100~110초 |
| 분류 | 약 2건/초 (12,000건이면 100분) |
| 요약·렌더·push | 2분 |
| **첫 30일 실행** | **1시간 40분~2시간** |
| **평시 실행** | **5~20분** (기판정 제외가 대부분을 걸러냄) |

### 품질

| 지표 | 값 |
|---|---|
| 골든셋 분류 재현율 | 20/20 |
| 축 경계 음성 표본 | 8/8 제외 |
| 요약 규격 통과 | 45~46 / 49 |
| 뇌신호 축 정밀도 | v1 1/18 → v3 이후 9/11 |

---

## 6. 실패에서 나온 규칙 (같은 실수를 반복하지 말 것)

1. **테스트를 통과해도 실물을 봐야 한다.** 렌더 테스트가 전부 통과한 상태에서
   실제 데이터로 페이지를 만들었더니 폭주 잔재·`unknown` 날짜·빈 카드가 나왔다.

2. **재현율만 재는 표본은 과잉포섭을 못 잡는다.** 골든셋 20편이 전부 "진짜
   brain decoding"이라, "뇌 신호를 다루면 brain_decoding"이라는 잘못된 규칙으로도
   20/20이 나왔다. 그래서 `data/golden_set.json`에 `negatives`(제외돼야 할 것 8건)와
   `positives_broadened`(포함돼야 할 것 3건)를 넣었다.

3. **비율을 정하기 전에 공급을 재라.** brain decoding에 35%를 배정했다가
   하루 0.2편밖에 없다는 걸 나중에 알았다. EHR 축은 먼저 재고 넣었다.

4. **guided decoding은 안전장치가 아니다.** temperature 0에서 모델이 값을 다 쓴 뒤
   종료 토큰 대신 반복 루프에 빠진다(원문 50만 자). `max_tokens`를 올리는 게 아니라
   **낮추고** 복구·다듬기로 처리한다.

5. **프롬프트의 사소한 한 줄이 결과를 지배한다.** "독자는 의료 AI 연구자"를 지우니
   요약 통과가 15/20 → 19/20이 됐고 날조도 사라졌다. few-shot 예시의 도메인 표현도
   무관한 논문 요약으로 샜다.

6. **한 소스의 실패가 전체를 죽이면 안 된다.** arXiv만 예외를 던져 이틀을 잃었다.

7. **음성 결과를 믿기 전에 양성 대조를 하라.** arXiv 제목 검색이 5건 모두 실패했을 때,
   같은 코드로 ImageBind를 찾아 검색 방법 자체는 정상임을 확인했다.

8. **기억 속 ID를 단정하지 마라.** CheXagent·SKiT의 arXiv ID를 기억으로 짚었는데
   둘 다 틀렸다.

9. **`.gitignore`는 줄 끝 주석을 지원하지 않는다.** `data/entries/  # 주석`으로 썼다가
   규칙이 죽어 테스트 산출물이 커밋됐다.

10. **테스트가 실제 저장소에 쓰지 않게 하라.** `ENTRIES_DIR`을 임시 경로로 바꾸지
    않아 테스트가 만든 "Paper 0" 16건이 8-31 아카이브(44건)를 밀어냈다.

11. **cron은 서버 로컬 시각으로 돈다.** 이 서버는 KST다. UTC로 넘겨짚어 `40 0`을
    쓰면 9시간 어긋난다.

12. **`git diff`는 추적되지 않은 파일을 못 본다.** 원장 샤드가 처음 생기는 달에
    "변경 없음"으로 판정돼 커밋을 건너뛸 뻔했다. `git status --porcelain`을 쓴다.

13. **락 파일을 지우지 마라.** flock은 fd로 잠그므로 파일을 지우면 동시 실행이 된다.

15. **SSH 키 권한이 풀리면 push만 조용히 실패한다.** `0660`이면 git이 키를 무시해
    `Permission denied (publickey)`가 난다. 파이프라인은 성공했는데 사이트만 안
    올라가는 형태라 알아채기 어렵다. `chmod 600 ~/.ssh/id_ed25519_meddaily`.

14. **참조 대상이 있으면 먼저 열어봐라.** 스펙 문서만 보고 디자인을 만들었다가
    전면 재작업했다. `common-ui.js` 같은 외부 스크립트까지 확인해야 한다.

---

## 7. 열려 있는 문제

| 문제 | 상태 |
|---|---|
| **arXiv 429로 이틀 실패** | 코드는 고침. 차단 해제 확인 후 수동 1회 실행 필요 (§2) |
| **Frontier AI가 기관 필터가 아님** | 지금은 HF Daily Papers upvote 기반. 소속 기관으로 거르려 했으나 arXiv `affiliation` 필드와 S2 `authors.affiliations`를 **429 때문에 측정 못 함**. 한도 풀리면 재측정 → 되면 기관 필터 덧대기 |
| **S2가 3,000 상한에 걸림** | 학회 논문 일부 누락. 올릴 수 있으나 대부분 arXiv와 중복 |
| **별점 분포 압축** | v5 눈금으로 고쳤으나 **아직 실제 실행에 적용 안 됨**(v5 이후 성공한 실행이 없음). 다음 실행에서 확인할 것 |
| **골든셋이 비-arXiv 경로를 검증 못 함** | 확정 비-arXiv 표본이 CurConMix 1건뿐. 저널 전용 논문 3~4편 추가 필요 |
| **보류 원장 크기** | 24,790건. TTL 지난 샤드는 자동 삭제되나 git 이력에는 남는다 |
| 디스크 정리 | 사용자가 "나중에" |

---

## 8. 설정을 바꿀 때 알아둘 것

- **축·비율**: `config.AXES`. 합이 1.0이어야 하고 `test_config.py`가 검사한다.
- **카테고리 추가**: `config.CATEGORIES`에 넣고 `test_config.py`·
  `test_classify_summarize.py`의 개수 상수도 함께 고친다.
- **분류 규칙을 고쳤으면 `CLASSIFY_PROMPT_VERSION`을 올린다.** 안 올리면 잘못
  걸러진 논문이 영영 회수되지 않는다. 올리면 다음 실행이 전량 재분류라 2시간 걸린다.
- **디자인**: `templates/*.j2` 고치고 `pipeline.py render`. GPU 불필요.
- **테두리 임계**: `config.STAR_HIGHLIGHT_MIN`.
- 안내문 내용은 `config`에서 생성되므로 축·카테고리를 고치면 자동으로 따라간다.

---

## 9. 환경

```
vllm 0.8.5.post1 · torch 2.6.0+cu124 · transformers 4.51.3
```

버전을 올리지 말 것 — 이유가 `requirements.txt`에 적혀 있다.

- vLLM 상위 버전은 torch cu130을 끌어와 **GPU를 조용히 인식 못 한다**(예외 없음)
- transformers 5.x는 `all_special_tokens_extended`를 제거해 Qwen 토크나이저가 깨진다
- `flashinfer-python`을 설치하면 ABI 불일치로 `OSError`가 나는데 vLLM은
  `ImportError`만 잡는다 — **설치하지 말 것**

GPU 4장 (RTX A6000 48GB). `if __name__ == "__main__":` 가드가 필요하다 —
vLLM v1이 워커를 spawn하며 메인 스크립트를 재import한다.
