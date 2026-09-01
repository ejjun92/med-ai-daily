# med-ai-daily
medical ai daily arxiv

## 매일 자동 실행

```
scripts/run_daily.sh          # cron이 부르는 진입점 (flock + 로그 + push)
scripts/install_cron.sh       # crontab 등록 (12:00 KST)
```

`run_daily.sh`가 지키는 것:

- **겹쳐 돌지 않는다** — `flock -n`. 전날 실행이 늦어져 물려도 GPU를 두 개 잡지 않는다.
- **실패하면 push하지 않는다** — 깨진 페이지를 올리느니 어제 것을 남긴다.
- **로그가 남는다** — `logs/daily-YYYY-MM-DD.log`, 30일치 보관. cron은 조용히 실패하는 것이 기본값이다.

cron은 **서버 로컬 시각**으로 돈다. 이 서버는 KST이므로 crontab에 `0 12`를 적는다.
UTC 서버라고 넘겨짚어 `40 0`을 쓰면 9시간 어긋난 채로 돈다 —
`install_cron.sh`가 타임존을 확인하고 다르면 멈춘다.

## 수동 실행

```bash
python src/pipeline.py run                      # 오늘치 생성 → docs/
python src/pipeline.py run --dry-run            # 수집까지만. 모델을 띄우지 않는다
python src/pipeline.py run --date 2026-08-20    # 특정 날짜
python src/pipeline.py run --ignore-seen --out-dir /tmp/x   # 원장을 건드리지 않고 재생성
python src/pipeline.py run --replay-deferred    # 보류분을 프롬프트 버전 무관하게 재분류
```

`--ignore-seen`은 원장을 **읽지도 쓰지도** 않는다. 실험이 원장을 오염시키면
다음 날 정상 논문이 '기게시'로 걸러져 사라진다.

## 비용이 갈리는 지점

분류는 후보 전체(~800건)에 돌리고, **요약은 선별된 40~60건에만** 돌린다.
요약까지 전체에 돌리면 GPU 점유 시간이 30배가 된다.

## 주제 축

| 축 | 비중 | 내용 |
|---|---|---|
| 🧠 뇌신호 AI | 30% | 자극·의미 복원, BCI, EEG/fMRI 표현학습, 뇌-모델 정렬 |
| 🔬 수술영상 | 26% | 수술 장면 이해, 기구 분할, VLP, 술기 평가 |
| ⚙️ 딥러닝 방법론 | 17% | 기반모델, 표현학습, 불확실성, 해석가능성 |
| 📈 EHR·임상 시계열 | 14% | 코드 시퀀스, ICU 시계열, 임상 LLM, 결과 예측 |
| 🩺 의료영상 AI | 13% | 의료 기반모델, 분할·진단, 재구성 |

EHR과 의료영상은 **입력이 무엇인가**로 가른다 — 코드·수치·생체신호·임상 노트면
EHR, 영상이 주 입력이면 의료영상이다.

## GPU 사용

한 번에 **한 장만** 쓴다. 기동 전에 `0 → 1 → 2 → 3` 순으로 훑어 여유
VRAM이 `GPU_FREE_VRAM_REQUIRED_MB`(24GB) 이상인 첫 장을 잡는다.

- 모두 점유 중이면 5분씩 3번 기다렸다가, 그래도 안 비면 **그날은 건너뛴다.**
  연구 작업을 밀어내지 않는다.
- 가장 여유가 큰 장이 아니라 **선호 순서**를 따른다 — 같은 상황에서 늘 같은
  장을 잡아야 어느 장이 이 작업용인지 예측할 수 있다.
- 점유량은 약 46GB(`GPU_MEMORY_UTILIZATION = 0.85`). 모델 가중치가 18GB,
  나머지는 KV 캐시다. 줄이려면 이 값을 낮춘다.
- 작업이 끝나면 `LocalLLM.close()`가 반드시 반납한다.
