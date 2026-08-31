#!/usr/bin/env bash
# 매일 09:40 KST에 cron이 부른다.
#
# 이 스크립트가 지키는 것 세 가지:
#   1. 겹쳐 돌지 않는다 (flock). 전날 실행이 40분을 넘겨 물려도 안전하다.
#   2. 실패하면 push하지 않는다. 깨진 페이지를 올리느니 어제 것을 남긴다.
#   3. 무슨 일이 있었는지 로그로 남는다. cron은 조용히 실패하는 것이 기본값이다.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MED_AI_PYTHON:-/data1/server1/miniconda3/envs/meddaily/bin/python}"
LOCK="$REPO/logs/.daily.lock"
LOG_DIR="$REPO/logs"
DAY="$(TZ=Asia/Seoul date +%F)"
LOG="$LOG_DIR/daily-$DAY.log"

mkdir -p "$LOG_DIR"

# flock -n: 이미 돌고 있으면 기다리지 않고 즉시 끝낸다. 대기하면 cron이
# 매일 쌓여 결국 GPU를 두 개 잡는다.
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -Is)] 이전 실행이 아직 돌고 있어 건너뜀" >> "$LOG"
  exit 0
fi

exec >> "$LOG" 2>&1
echo "=============================================================="
echo "[$(date -Is)] 시작  repo=$REPO"

cd "$REPO"

# 다른 곳에서 config.py를 고쳐 push했을 수 있다. 원장은 append-only JSONL이라
# rebase가 깔끔하게 된다 (그래서 그 형식을 골랐다).
git pull --rebase --autostash origin main || {
  echo "[$(date -Is)] git pull 실패 — 로컬 상태로 계속 진행"
}

if "$PYTHON" src/pipeline.py run; then
  echo "[$(date -Is)] 파이프라인 성공"
else
  echo "[$(date -Is)] 파이프라인 실패 — push하지 않고 종료"
  exit 1
fi

# 페이지가 실제로 갱신됐는지 본다. 변화가 없으면 빈 커밋을 만들지 않는다.
#
# `git diff`가 아니라 `git status --porcelain`을 쓴다. diff는 **추적되지 않은
# 파일을 보지 못한다** — 원장 샤드(data/published/2026-08.jsonl)가 처음 생기는
# 달에는 "변경 없음"으로 판단해 원장을 영영 커밋하지 않는다.
if [ -z "$(git status --porcelain -- docs data)" ]; then
  echo "[$(date -Is)] 변경 없음 — 커밋 생략"
else
  git add docs data
  git commit -q -m "daily: $DAY"
  # push 실패는 파이프라인 실패와 다르다. 다음 날 함께 올라간다.
  if git push -q origin main; then
    echo "[$(date -Is)] push 완료"
  else
    echo "[$(date -Is)] push 실패 — 커밋은 남았다. 다음 실행에서 재시도된다"
    exit 1
  fi
fi

# 로그는 30일치만 남긴다. 서버 디스크가 넉넉하지 않다.
find "$LOG_DIR" -name 'daily-*.log' -mtime +30 -delete 2>/dev/null || true
echo "[$(date -Is)] 종료"
