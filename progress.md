# med-ai-daily — 진행 상황

> 대화가 끊겨도 이 문서만 읽으면 이어서 작업할 수 있도록 정리한다.
> 마지막 갱신: 2026-09-04 23:50 KST

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

## 2. 현재 상태 (2026-09-04 23:50 기준)

**정상.** 2026-09-04자 35건 + Frontier 6건이 발행돼 있다(커밋 `b113264`). 09-02
사고 이후 처음으로 429 없이 완주했다. 다음 자동 실행은 09-05 12:00 KST.

### 오늘 파낸 것 — 35건은 공급 부족이 아니었다

목표 50건인데 35건만 나갔다. 원장을 세어 보니 원인이 공급이 아니었다.

```
09-04 보류 869건 = not_relevant 606 + quota 263
```

**관련 있다고 판정된 298건 중 35건만 실리고 263건이 자리가 없어 밀렸다.**

`selection.py`의 축 간 재배분이 각 축 `bounds[1]`(최대치)에 막혀 있었다.
공급이 마른 축이 배정받은 칸이 다른 축으로 넘어가지 않는 구조다.

| 축 | 최대치 | 09-04 선별 | |
|---|---|---|---|
| brain_decoding | 18 | **6** | 풀 소진 (진짜 부족) |
| surgical_video | 16 | **3** | 풀 소진 (진짜 부족) |
| dl_methodology | 10 | **10** | 최대치에 걸림 |
| ehr_clinical | 8 | **8** | 최대치에 걸림 |
| medical_imaging | 8 | **8** | 최대치에 걸림 |

brain·surgical이 받은 34칸 중 9칸만 쓰고 **25칸이 유휴로 죽었다.** 9 + 26 = 35.
`DAILY_MIN = 40`은 이 구조에서 도달 자체가 불가능했다.

"부족하면 억지로 채우지 않는다"와는 다른 사안이다. 그 원칙은 무관한 논문으로
패딩하지 말라는 뜻인데, 밀린 263건은 전부 `is_relevant: true`다.

**고친 것** (`test_selection.py`가 지킨다):
- `select()` ②단계를 둘로 나눴다. **②-a**는 기존대로 최대치까지, **②-b**는
  그러고도 하한 미달일 때만 유휴 칸을 아직 공급이 있는 축에 넘긴다
- `_apportion()`이 **최대잉여법**으로 비중대로 나눈다. 라운드로빈이면 13% 축과
  30% 축이 같은 수를 받아 비율 방침이 깨진다
- 무제한이 아니다 — `Axis.ceiling`(= `AXIS_OVERFLOW_FACTOR` 2.5배)에서 멈춘다
- `SelectionResult.overflow_by_axis`로 초과분을 기록하고 로그에 남긴다

### 함께 고친 것 — arXiv 재시도가 조용했다

09-04 arXiv 수집이 **751초**, 09-03은 18초였다. 로그에 429도 에러도 없다.
`_fetch`의 재시도 루프가 아무것도 로깅하지 않고 페이지당 최대 7분을 잤기
때문이다. 09-02 사고 후 "격리 + 백오프"는 넣었으나 **재시도 자체의 관측성은
빠져 있었다.** 재시도·소진·복구 로그와 총 소요시간 계측을 넣었다.

성공 로그가 `ET.fromstring` **앞에** 있던 것도 함께 고쳤다 — arXiv는 429/503일
때 XML 대신 HTML을 돌려주므로, 파싱 실패로 재시도하면서 로그에는 "성공"이 남는
상태였다.

### 함께 고친 것 — `--ignore-seen`이 엔트리 캐시를 덮었다

도움말이 "원장에 쓰지도 않는다"고 하고 이 문서 §4도 "원장 안 건드리고 재생성"
이라고 안내했는데, `save_entries()`가 `if not ignore_seen` 밖에 있어
`data/entries/YYYY-MM-DD.json`을 덮어썼다. 09-04 실물 재현을 돌리려다 발견했다.

발현 형태가 고약하다 — `docs/`는 멀쩡한데 캐시만 재현 결과로 바뀌고, 나중에
`render`가 그걸로 그날 페이지를 다시 그린다. 규칙 10과 같은 계열이다.

### max_tokens 절삭은 실피해가 거의 없다 (조치 안 함)

로그가 "상한 상향 검토"를 반복해 요란하지만 실측하니 손실이 작다.

- 09-04 절삭 11건(분류 6, 요약 5) → **파싱 실패 0건**. `salvage_json`이 전부 흡수
- 게시 35건의 `rationale`: 미완결 문장 **0건**, 최대 227자(상한 300 미만).
  `rationale`이 스키마 마지막 필드라 잘려도 판정값(`axis`/`stars`)은 온전하다
- 요약 5건 절삭 중 4건은 `trim_to_sentence`가 복구, **1건만** 규격 미달
- 09-03은 규격 미달 4/49(8.2%), 09-04는 1/35(2.9%) — 악화 아님

`CLASSIFY_MAX_TOKENS` 상향은 실익이 없고, `SUMMARIZE_MAX_TOKENS`는 올려도 안
줄어든다(`config.py:285` 주석). **경고 문구가 과잉인 쪽이 문제에 가깝다.**

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

### 3-7. 테스트 194개 (모델·네트워크 없이 2초)

| 파일 | 개수 | 지키는 것 |
|---|---|---|
| `test_render.py` | 36 | 이스케이프, 테마 3상태, 아카이브, 날짜 선택기 |
| `test_ledger.py` | 22 | 원장 매칭, 버전 게이트, payload, 샤드 정리 |
| `test_selection.py` | 19 | 쿼터 배분, **유휴 칸 재배분**, 결정성 |
| `test_classify_summarize.py` | 18 | 스키마-설정 일치, 축 경계, 요약 검증 |
| `test_pipeline.py` | 17 | 단계 순서, 원장 시점, **저장소 오염 방지** |
| `test_gpu_pick.py` | 10 | GPU 선택·대기·포기 |
| `test_llm_salvage.py` | 9 | 깨진 JSON 복구 (실제 폭주 출력 픽스처) |
| `test_config.py` | 9 | 설정 정합성, **천장 배수 하한** |
| `test_arxiv_window.py` | 7 | 창 공백 메우기, **재시도 관측성** |
| `test_crossdedup.py` | 7 | 교차 중복제거 |
| `test_venue.py` | 6 | 학회 라벨 |
| `test_source_failure.py` | 4 | **한 소스가 죽어도 발행** (09-02 사고) |

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
$PY src/pipeline.py run --ignore-seen --out-dir /tmp/x   # 원장·캐시 안 건드리고 재생성
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

09-04는 평시인데 **29분**(1,752초) 걸렸다. 수집 751초(09-03은 18초), 모델 로드
370초(평소 100~110초). arXiv 재시도 로그가 없어 원인을 로그로 구분할 수 없었다 —
그래서 재시도 로깅과 소요시간 계측을 넣었다. 09-05부터는 로그로 판별된다.

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

14. **참조 대상이 있으면 먼저 열어봐라.** 스펙 문서만 보고 디자인을 만들었다가
    전면 재작업했다. `common-ui.js` 같은 외부 스크립트까지 확인해야 한다.

15. **SSH 키 권한이 풀리면 push만 조용히 실패한다.** `0660`이면 git이 키를 무시해
    `Permission denied (publickey)`가 난다. 파이프라인은 성공했는데 사이트만 안
    올라가는 형태라 알아채기 어렵다. `chmod 600 ~/.ssh/id_ed25519_meddaily`.

16. **관측되지 않는 백오프는 안전장치가 아니다.** 09-02 사고 후 429 백오프를
    넣었는데 재시도 루프가 아무것도 로깅하지 않아, 수집이 18초에서 751초로 늘어도
    로그로는 원인을 구분할 수 없었다. 조용히 성공하면서 나빠지는 형태는 규칙 15
    (SSH 키)와 같은 계열이다. **재시도·대기·복구는 반드시 흔적을 남긴다.**

17. **"자리가 없어 밀렸다"와 "공급이 없다"는 다르다.** 35건만 나간 것을 공급
    부족으로 읽었는데, 원장을 세어 보니 관련 판정 298건 중 263건이 quota로
    밀린 것이었다. **로그의 결과 수치만 보지 말고 원장의 사유별 분포를 세라.**

18. **상한을 완화하는 것과 제거하는 것은 다르다.** 유휴 칸 재배분을 넣으면서
    축 최대치 검사를 통째로 없앴더니, 최대치 8인 축이 40칸을 전부 먹었다.
    리뷰가 잡았다. 완화에는 반드시 새 천장이 따라와야 한다.

19. **테스트가 통과한다고 그 테스트가 무언가를 지키는 건 아니다.** "비중대로
    나눈다"를 검증한다는 테스트가 부족분 5칸에서는 라운드로빈과 결과가 같아
    아무것도 배제하지 못했다. **두 구현이 갈라지는 입력 크기를 골라야 한다.**

20. **불변식을 바꾸면 그것을 주장하던 옛 테스트를 찾아라.** "각 축 최대치를
    넘지 않는다"가 더 이상 전역 성질이 아닌데 그 테스트는 우연히 계속
    통과하며, 읽는 사람에게 살아 있는 보증인 척했다.

21. **`assert`는 `-O`에서 사라진다.** 루프 종료 보증을 `if ... break`에서
    `assert`로 바꾸면 운영에서 보증이 없어진다. 둘 다 둔다.


---

## 7. 열려 있는 문제

| 문제 | 상태 |
|---|---|
| ~~arXiv 429로 이틀 실패~~ | **해결.** 소스 실패 격리 + 백오프. 09-03 정상 발행 |
| ~~유휴 쿼터로 매일 관련 논문을 버림~~ | **해결(09-04).** ②-b 재배분 + `Axis.ceiling`. 09-05 실행이 첫 실물 검증 |
| ~~arXiv 재시도가 조용함~~ | **해결(09-04).** 재시도·소진·복구 로그 + 소요시간 |
| ~~`--ignore-seen`이 엔트리 캐시를 덮음~~ | **해결(09-04).** 캐시도 원장과 같이 보호 |
| **09-04 선별 변경을 실물로 재현 못 함** | quota 보류분은 payload를 저장하지 않는다(용량 때문, `ledger.py:24-28`). 그래서 그날 밀린 263건의 초록이 없어 재분류가 불가능하다. 합성 풀 재현으로 구 코드가 실제 발행분(35건, 축별 6/3/10/8/8)과 **정확히 일치**함은 확인했다. 실물 확인은 09-05 실행 |
| **`max_tokens` 경고가 과잉** | 실피해는 요약 1건인데 로그가 매 배치 "상한 상향 검토"를 외친다. 임계를 두거나 문구를 낮출 것 |
| **Frontier AI가 기관 필터가 아님** | 지금은 HF Daily Papers upvote 기반. 소속 기관으로 거르려 했으나 arXiv `affiliation` 필드와 S2 `authors.affiliations`를 **429 때문에 측정 못 함**. 한도 풀리면 재측정 → 되면 기관 필터 덧대기 |
| **S2가 3,000 상한에 걸림** | 학회 논문 일부 누락. 올릴 수 있으나 대부분 arXiv와 중복 |
| 별점 분포 압축 | v5 첫 적용 결과 `{5:2, 4:47}`. v4의 5점 0건보다는 나아졌으나 여전히 4점에 몰린다. 상위 49건만 싣는 이상 구조적으로 그렇다 — 더 조이려면 눈금을 한 칸 더 내려야 한다 |
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
- **유휴 칸 재배분 천장**: `config.AXIS_OVERFLOW_FACTOR`(2.5). 공급 축이 3개뿐인
  날에도 40건에 닿게 하는 값이다. 2.0이면 35건에 갇히고, 3.0이면 한 축이 36건까지
  먹어 단일 주제 페이지가 된다. **`DAILY_MAX/DAILY_MIN`(=1.5) 밑으로 내리면
  재배분이 경고 없이 죽는다** — `test_config.py`가 막는다.
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
