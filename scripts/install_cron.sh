#!/usr/bin/env bash
# crontab에 일일 실행을 등록한다. 이미 있으면 갱신한다.
#
# 09:40 KST를 고른 이유: arXiv는 20:00 ET(Sun-Thu)에 발표하고 이는 09~10 KST다.
# 40분 여유를 두면 그날 발표분이 API에 올라온 뒤에 수집한다.
#
# cron은 **서버 로컬 시각**으로 돈다. 이 서버는 이미 KST이므로 그대로 09:40을
# 적는다. UTC 서버라고 넘겨짚어 "40 0"을 쓰면 9시간 어긋난 채로 조용히 돈다.
# 아래에서 실제 타임존을 확인하고 다르면 멈춘다.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TZNAME="$(date +%Z)"
if [ "$TZNAME" != "KST" ]; then
  echo "서버 타임존이 KST가 아니라 $TZNAME 입니다." >&2
  echo "crontab 시각을 직접 환산해서 등록하세요 (목표: 09:40 KST)." >&2
  exit 1
fi

LINE="40 9 * * * $REPO/scripts/run_daily.sh"   # 서버가 KST이므로 그대로 09:40
MARK="# med-ai-daily"

current="$(crontab -l 2>/dev/null || true)"
cleaned="$(printf '%s\n' "$current" | grep -v -F "$MARK" | grep -v -F "run_daily.sh" || true)"
printf '%s\n%s %s\n' "$cleaned" "$LINE" "$MARK" | sed '/^$/d' | crontab -

echo "등록됨:"
crontab -l | grep -F "$MARK"
echo
echo "서버 타임존: $(date +%Z\ %z)   KST 기준 실행 시각: 09:40"
