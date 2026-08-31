"""한국어 3줄 요약 + 태그.

요약은 이 제품의 목적이 아니다 (원칙 2). 제목만 보고 열지 말지 판단하기
어려울 때 결정을 도와주는 정도면 충분하다. 실패하면 제목만 싣는다 —
요약 실패가 리스트업 실패가 되어서는 안 된다.
"""
from __future__ import annotations

import re

import config
from models import Paper, Summary

SCHEMA = {
    "type": "object",
    # 순서가 중요하다: guided decoding은 스키마 property 순서대로 생성한다.
    # 요약이 길어져 절삭되면 뒤 필드가 통째로 날아가므로 짧은 tags를 앞에 둔다.
    # (실측: tags를 뒤에 뒀을 때 SurgX가 태그 없이 통과했다.)
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"},
                 "minItems": config.SUMMARY_TAGS_MIN,
                 "maxItems": config.SUMMARY_TAGS_MAX},
        "korean_summary": {"type": "string"},
    },
    "required": ["tags", "korean_summary"],
    "additionalProperties": False,
}

PROMPT = """다음 논문을 한국어로 요약하라. 독자는 이 요약만 보고 논문을
열어볼지 결정한다.

# 요약 규칙
- **150~250자를 넘기지 마라.** 길면 잘려서 버려진다. 짧고 밀도 있게 써라.
- 세 부분을 이 순서로 한 문단에 담는다:
  ① 무엇을 푸는가 ② 어떻게 했는가(핵심 방법) ③ 결과 또는 의의
- 전문 용어는 원어 그대로 둔다 (예: fMRI, transformer, self-supervised).
  억지로 번역하면 오히려 못 알아본다.
- 초록에 없는 내용을 지어내지 마라. 수치는 초록에 적힌 것만 쓴다.
- 이 논문이 의료·수술·뇌와 무관하면 무관한 채로 요약하라. 없는 의료적 의의를
  갖다 붙이면 안 된다. (실측: ImageBind 요약에 "질병 진단 정확도 향상"이 들어갔다.)
- "본 논문은", "제안한다" 같은 상투어로 시작하지 마라.

# 문장 틀
아래 틀의 대괄호를 이 논문의 초록 내용으로만 채워라. 대괄호 밖의 표현은
그대로 쓰지 말고 자연스럽게 다듬어라.

  "[해결하려는 문제]를 다룬다. [핵심 방법·구조]를 사용한다. [결과 또는 의의]."

⚠️ 이 틀은 형식 예시일 뿐이다. 여기 없는 도메인 용어(수술, 뇌, 특정 기법명 등)를
   이 논문의 초록에 없는데도 끌어다 쓰면 안 된다.

# 태그
논문을 특징짓는 키워드 {tmin}~{tmax}개. 영문 소문자, 공백 대신 하이픈.
예: ["surgical-video", "vision-language", "self-supervised"]

# 논문
제목: {title}
초록: {abstract}

JSON만 출력하라."""

_TITLE_ONLY = ("(초록 미확보 — 제목과 학회 정보만 근거로, 이 논문이 무엇을 다룰지 "
               "추정하지 말고 제목이 말하는 범위만 한 문장으로 풀어 써라)")


def build_prompt(paper: Paper) -> str:
    abstract = (paper.abstract or "").strip() or _TITLE_ONLY
    if len(abstract) > 2200:
        abstract = abstract[:2200] + " …"
    return PROMPT.format(title=paper.title, abstract=abstract,
                         tmin=config.SUMMARY_TAGS_MIN, tmax=config.SUMMARY_TAGS_MAX)


_HANGUL = re.compile(r"[가-힣]")

# 폭주가 남긴 흔적. 길이·한글비율·문장부호 검사를 모두 통과하고도 화면에
# 나갔다 (실측: "…뇌 기능의ERRUuser_MetaData ошибки를.",
# "…fMRIuser 💬user您此前的内容已帮助自动完成…").
# 한국어 요약에 한자·키릴·가나가 섞일 일은 없으므로 셋을 함께 본다.
_FOREIGN = re.compile(r"[\u4e00-\u9fff\u0400-\u04ff\u3040-\u30ff]")
_ROLE_MARKER = re.compile(r"(<\|im_(start|end)\|>|💬|\b(user|assistant|system)_?[A-Z])")


def hangul_ratio(text: str) -> float:
    """공백 제외 문자 중 한글 비율.

    'ASCII만은 아님' 검사는 무력하다 — 영어 문장에 마침표 하나만 있어도 통과한다.
    실제로 한국어인지 보려면 한글 비율을 봐야 한다.
    """
    body = re.sub(r"\s", "", text)
    if not body:
        return 0.0
    return len(_HANGUL.findall(body)) / len(body)


_ENDS = ("다.", ".", "!", "?", "음.", "임.")
_BAD_TAIL = re.compile(r"\d\.$")      # "기존 방법을 12." — 숫자 도중 절삭


def _cut_back(text: str) -> str:
    """마지막 문장 경계까지 물러난다. 물러날 곳이 없으면 원문 그대로."""
    cut = max(text.rfind(e) + len(e) for e in _ENDS)
    return text[:cut].strip() if cut >= config.SUMMARY_MIN_CHARS else text


def trim_to_sentence(text: str, limit: int) -> str:
    """상한을 넘거나 문장 중간에서 끊긴 요약을 마지막 온전한 문장에서 자른다.

    세 가지를 함께 다룬다 — 모델이 길이 지시를 무시해 넘친 경우, max_tokens에
    걸려 어절 중간에서 잘린 뒤 복구된 경우, 그리고 "기존 방법을 12." 처럼
    숫자 도중에 잘려 문장 끝처럼 보이는 경우다(실측: MindLLM).

    길다고 통째로 버리면 내용이 멀쩡한 요약까지 잃는다 — 실측에서 이렇게 13건을
    날렸다. 반대로 문장 경계를 못 찾으면 원문을 그대로 돌려줘 validate()에서
    걸러지게 둔다. 여기서 조용히 이상한 문자열을 만들어 내보내지 않는다.
    """
    out = text[:limit] if len(text) > limit else text
    if len(text) > limit or not out.endswith(_ENDS):
        out = _cut_back(out)
    for _ in range(3):                  # 절삭 꼬리가 겹칠 수 있다
        if not _BAD_TAIL.search(out):
            break
        back = _cut_back(out[:-1])
        if back == out[:-1]:
            break
        out = back
    return out


def validate(raw: dict | None) -> Summary | None:
    """규격 미달이면 None — 호출자는 제목만 싣는다 (AC-9)."""
    if not raw:
        return None
    text = trim_to_sentence(str(raw.get("korean_summary", "")).strip(),
                            config.SUMMARY_MAX_CHARS)
    if not (config.SUMMARY_MIN_CHARS <= len(text) <= config.SUMMARY_MAX_CHARS):
        return None
    if _BAD_TAIL.search(text):
        # 다듬어도 절삭 꼬리가 남았다 — 물러나면 하한 밑으로 떨어지는 경우다.
        # 어중간한 문장을 싣느니 제목만 싣는다.
        return None
    if len(_FOREIGN.findall(text)) >= 3 or _ROLE_MARKER.search(text):
        # 폭주가 문장처럼 보이는 형태로 끝난 경우. 앞부분만 살리려 들지 않는다 —
        # 어디까지가 진짜인지 판정할 근거가 없다.
        return None
    if hangul_ratio(text) < config.SUMMARY_HANGUL_RATIO_MIN:
        return None    # 모델이 영어로 돌아간 경우
    tags = [str(t).strip().lower().replace(" ", "-")
            for t in raw.get("tags", []) if str(t).strip()]
    return Summary(korean_summary=text, tags=tags[:config.SUMMARY_TAGS_MAX])


def summarize(llm, papers: list[Paper], log=print) -> list[Summary | None]:
    if not papers:
        return []
    raws = llm.chat_json([build_prompt(p) for p in papers], SCHEMA,
                         config.SUMMARIZE_MAX_TOKENS)
    out = [validate(r) for r in raws]
    bad = sum(1 for s in out if s is None)
    if bad:
        log(f"  [summarize] {bad}/{len(papers)}건 규격 미달 → 제목만 표시")
    return out
