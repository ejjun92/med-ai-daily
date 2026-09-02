"""단일 진실 원천 (Single source of truth).

축·카테고리·쿼터·소스·모델 설정이 전부 여기 있다. 주제를 조정할 때
다른 파일을 건드릴 필요가 없어야 한다 (계획 R-7).
"""
from __future__ import annotations
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────
# 축 (Axis) — 비율은 골든셋 실측에 맞춘 값이다 (계획 D-20)
#
#   v1~v3은 의료영상 40 / 수술 30 / 방법론 20 / brain 10 이었으나,
#   사용자가 선정한 골든셋 20편의 실제 분포가 brain 35 / 수술 40 /
#   방법론 20 / 의료영상 5 로 정반대였다. 골든셋이 "놓치면 안 되는
#   논문"의 조작적 정의이므로 쿼터를 그쪽에 맞춘다.
# ─────────────────────────────────────────────────────────────────

DAILY_MIN = 40
DAILY_MAX = 60
DAILY_TARGET = 50          # 축별 목표치 계산의 기준

@dataclass(frozen=True)
class Axis:
    key: str
    label: str
    ratio: float           # 0.0~1.0
    emoji: str = ""        # 섹션 헤더용. 훑을 때 자리를 기억하게 해준다

    @property
    def target(self) -> int:
        return round(DAILY_TARGET * self.ratio)

    @property
    def bounds(self) -> tuple[int, int]:
        """일일 총량 범위에 비율을 적용한 하한/상한."""
        return round(DAILY_MIN * self.ratio), round(DAILY_MAX * self.ratio)


AXES: tuple[Axis, ...] = (
    Axis("brain_decoding",  "뇌신호 AI",         0.30, "🧠"),
    Axis("surgical_video",  "수술영상",          0.26, "🔬"),
    Axis("dl_methodology",  "딥러닝 방법론",      0.17, "⚙️"),
    Axis("ehr_clinical",    "EHR·임상 시계열",    0.14, "📈"),
    Axis("medical_imaging", "의료영상 AI",       0.13, "🩺"),
)
AXIS_KEYS = tuple(a.key for a in AXES)
AXIS_BY_KEY = {a.key: a for a in AXES}


# ─────────────────────────────────────────────────────────────────
# 카테고리 (22개) — 각 카테고리는 정확히 한 축에 속한다
#   카테고리 수는 축 비중에 비례하게 배분했다: brain 5 / 수술 4 / EHR 4 /
#   방법론 6 / 의료영상 3. 방법론은 8~12편을 6칸에 나눠 칸당 1~2편으로
#   얇다 — 1주차 결과를 보고 병합 검토 (계획 F-12).
# ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Category:
    id: str
    name: str
    axis: str
    description: str       # 분류 프롬프트에 그대로 들어간다
    emoji: str = ""


CATEGORIES: tuple[Category, ...] = (
    # ■ Brain Decoding (35%)
    Category("fmri_visual_decoding", "fMRI Visual Decoding & Reconstruction", "brain_decoding",
             "fMRI에서 피험자가 본·상상한 이미지·장면을 복원(reconstruction)한다", "👁"),
    Category("eeg_meg_decoding", "EEG/MEG Decoding & BCI", "brain_decoding",
             "EEG·MEG에서 자극 내용이나 인지·운동 의도를 읽어낸다. motor imagery, BCI, neurofeedback 포함", "⚡"),
    Category("brain_to_language", "Brain-to-Language & Multimodal Decoding", "brain_decoding",
             "뇌 신호에서 문장·단어·의미를 생성한다 (speech/text decoding)", "💬"),
    Category("cross_subject_alignment", "Cross-Subject Generalization & Neural Alignment", "brain_decoding",
             "디코딩 모델을 학습에 없던 피험자로 일반화(subject-agnostic)하거나, 뇌 표현과 딥러닝 모델 표현을 정렬·비교한다. 다기관 데이터 조화나 피험자 변이 보정 자체가 목적이면 여기가 아니다", "🔗"),
    Category("brain_representation", "Brain Signal Representation & Foundation Models", "brain_decoding",
             "EEG·fMRI **원신호 자체**(시계열·볼륨)의 표현학습·자기지도학습·foundation model. 연결성 그래프·커넥톰·메타분석처럼 신호에서 뽑아낸 2차 산물을 다루면 여기가 아니다", "🧬"),

    # ■ 수술영상 (30%)
    Category("surgical_scene", "Surgical Scene Understanding", "surgical_video",
             "수술 phase/step recognition, action triplet, 행동 인식", "🎬"),
    Category("surgical_segmentation", "Instrument & Anatomy Segmentation", "surgical_video",
             "수술 도구·해부구조 분할 및 추적", "✂️"),
    Category("surgical_vlp", "Surgical VLP & Foundation Model", "surgical_video",
             "수술 video-language pretraining, 수술 특화 기반모델", "📹"),
    Category("robotic_surgery", "Robotic Surgery & Skill Assessment", "surgical_video",
             "수술 로봇, 술기 평가, 수술 내비게이션", "🦾"),

    # ■ 딥러닝 방법론 (20%)
    Category("uncertainty", "Uncertainty Quantification & Calibration", "dl_methodology",
             "불확실성 정량화, 신뢰도 보정, OOD 탐지", "📊"),
    Category("explainability", "Explainability & Interpretability", "dl_methodology",
             "XAI, saliency, concept 기반 해석, mechanistic interpretability", "🔍"),
    Category("foundation_openvocab", "Foundation Models & Open-Vocabulary Perception", "dl_methodology",
             "open-world/open-vocabulary 검출, 통합 비전 기반모델", "🌐"),
    Category("multimodal_ssl", "Multimodal & Self-Supervised Representation", "dl_methodology",
             "멀티모달 임베딩, contrastive/masked 표현학습", "🧩"),
    Category("generative_editing", "Generative Modeling & Editing", "dl_methodology",
             "diffusion, 이미지 편집, 생성 제어", "🎨"),
    Category("robustness", "Robustness & Domain Generalization", "dl_methodology",
             "도메인 적응·일반화, distribution shift, continual learning", "🛡"),

    # ■ EHR·임상 시계열 (14%)
    Category("ehr_sequence", "EHR Sequence Modeling & Patient Trajectory", "ehr_clinical",
             "진단·처방·검사 코드 시퀀스의 표현학습, 환자 궤적 모델링. transformer·state-space 등 시퀀스 구조가 기여인 연구", "🧾"),
    Category("clinical_timeseries", "Clinical Time Series & Monitoring", "ehr_clinical",
             "ICU 생체신호, 다변량 임상 시계열 예측, 조기경보, 불규칙 샘플링 처리", "📉"),
    Category("clinical_llm", "Clinical LLM & Note Understanding", "ehr_clinical",
             "임상 노트 요약·정보추출, EHR 기반 QA·에이전트, 의무기록 대상 LLM", "📝"),
    Category("clinical_outcome", "Outcome & Risk Prediction", "ehr_clinical",
             "재입원·사망·질병 진행 예측, 생존분석, 치료 반응 예측. 영상이 아닌 표·시계열 기반", "⚕️"),

    # ■ 의료영상 AI (15%)
    Category("medical_foundation", "Medical Foundation Models, VLM & Report Generation", "medical_imaging",
             "의료 특화 기반모델, 판독문 생성, 의료 VQA, 임상 LLM", "🏥"),
    Category("medical_seg_diagnosis", "Segmentation, Detection & Diagnosis", "medical_imaging",
             "장기·병변 분할/검출, 질환 분류, 예후 예측, screening", "🫀"),
    Category("medical_recon_benchmark", "Reconstruction, Generation & Benchmark", "medical_imaging",
             "MRI/CT 재구성, denoising, 의료 데이터셋·평가체계", "🧲"),
)
CATEGORY_IDS = tuple(c.id for c in CATEGORIES)
CATEGORY_BY_ID = {c.id: c for c in CATEGORIES}


# ─────────────────────────────────────────────────────────────────
# 중요도 별점 기준 — 분류 프롬프트에 그대로 들어간다
# ─────────────────────────────────────────────────────────────────
STAR_RUBRIC = {
    # 실측(2026-09-01): 관련 5,883건 중 5점이 0건이었고 게시된 49건이 전부 4점이라
    # '중요도 정렬'이 사실상 동작하지 않았다. 5점을 "패러다임 전환"으로만 좁게
    # 정의하고 4점을 너무 넓게 잡은 탓이다. 눈금을 한 칸씩 내려 분포를 퍼뜨린다.
    5: "패러다임 전환, 주요 SOTA 갱신, 대형 공개 모델·데이터 릴리스",
    4: "의미 있는 기술 혁신, 주요 연구소 결과, 코드가 공개된 우수 논문",
    3: "유용한 개선 — 도구, 데이터셋, 벤치마크, 실용 응용",
    2: "니치하지만 해당 하위분야에서 참고할 만한 결과",
    1: "관련은 있으나 연구 임팩트가 제한적",
}


# ─────────────────────────────────────────────────────────────────
# 수집 소스 (3종) — 역할이 다르다
#   arXiv  : 프리프린트 발견 (초록 확실)
#   PubMed : 저널 5종 (초록 확실)
#   S2     : 학회 proceedings 발견 + 학회 라벨 공급
#
#   MICCAI가 PubMed 화이트리스트에 없는 이유: 2026-08-14 실측에서
#   PubMed 색인이 연 33~70건으로 실제 발행량(~860편)의 6%에 불과했다.
#   S2는 MICCAI 2024를 860편 전량 보유(DOI 99%, arXiv ID 52%)한다.
# ─────────────────────────────────────────────────────────────────

# 모든 날짜 계산의 기준. PubMed EDAT은 NCBI 로컬 기준이므로
# 질의 전 UTC 날짜로 변환한다 (재현 결정성에 필요).
TIMEZONE = "UTC"

ARXIV_CATEGORIES = ("cs.CV", "cs.LG", "cs.AI", "eess.IV", "q-bio.NC")
ARXIV_WINDOW_DAYS = 30       # 한 달. 좁은 창은 발표 지연·API 공백에 취약하다
ARXIV_MAX_WINDOW_DAYS = 45   # 공백 메우기 상한. 며칠 연속 실패해도 되돌아가되,
# 무한정 늘어나 한 번에 수천 건을 분류하는 일은 막는다
ARXIV_REQUEST_DELAY_S = 3.0  # arXiv API 권장 간격
ARXIV_PAGE_SIZE = 2000       # max_results 상한

PUBMED_JOURNALS = (
    "Med Image Anal",                        # Medical Image Analysis
    "IEEE Trans Med Imaging",                # IEEE TMI
    "IEEE Trans Pattern Anal Mach Intell",   # IEEE TPAMI
    "Radiol Artif Intell",                   # Radiology: Artificial Intelligence
    "Int J Comput Assist Radiol Surg",
    # EHR·임상 시계열 축을 위한 임상정보학 저널. 영상 저널만으로는 이 축이 안 찬다.
    "J Am Med Inform Assoc",
    "J Biomed Inform",
    "NPJ Digit Med",       # IJCARS (IPCAI 게재지)
)
PUBMED_WINDOW_DAYS = 30      # 한 달. 저널 색인 지연은 arXiv보다 길다
PUBMED_BATCH_SIZE = 200      # efetch 1회당 PMID 수 (POST 사용)
PUBMED_RETMAX = 1000         # esearch 기본값은 20 — 명시하지 않으면 조용히 잘린다
PUBMED_REQUEST_DELAY_S = 0.34  # 무인증 3 req/s. 키 있으면 10 req/s
# 원논문과 거의 같은 제목을 달아 교차 중복제거를 오염시킨다
PUBMED_EXCLUDED_TYPES = (
    "Published Erratum", "Comment", "Editorial",
    "Retraction of Publication", "Retracted Publication",
)
# NCBI가 모든 E-utility 호출에 요구한다. email은 repo variable로 주입 (원칙 5)
NCBI_TOOL = "med-ai-daily"

S2_VENUES = (
    "Medical Image Computing and Computer-Assisted Intervention",  # MICCAI
    "Information Processing in Computer-Assisted Interventions",   # IPCAI
    "Computer Vision and Pattern Recognition",                     # CVPR
    "International Conference on Computer Vision",                 # ICCV
    "European Conference on Computer Vision",                      # ECCV
    "Neural Information Processing Systems",                       # NeurIPS
    "International Conference on Machine Learning",                # ICML
    "International Conference on Learning Representations",        # ICLR
)
S2_YEARS_BACK = 2            # proceedings는 발표 시점에 일괄 등재된다
S2_REQUEST_DELAY_S = 1.0

# 소스별 상한 — 전역 단일 상한은 대량 색인된 proceedings 볼륨이
# 그날 arXiv를 통째로 밀어낼 수 있다 (계획 D-11)
ARXIV_MAX = 12000      # 30일 창 실측 9,294건(2026-09-01). 여유를 둔다 —
# 상한에 걸리면 조용히 버려지는 게 아니라 푸터에 표시되지만, 애초에 안 걸리는 게 낫다
PUBMED_MAX = 2000      # 30일 창 실측 204건 — 여유가 충분하다
S2_MAX = 3000          # 500이면 MICCAI 한 해(1,008편)도 못 담는다.
# 학회 논문집은 한 번에 등재되므로 상한이 곧 누락이다
DEFERRED_DAILY_MAX = 300     # 보류분은 소스 상한과 경쟁하지 않는 별도 레인

# 절삭 정렬키 — PubMed는 EDAT이 아니라 PDAT를 쓴다.
# 2019년 proceedings가 오늘 대량 색인됐다고 오늘 arXiv보다 앞설 이유가 없다.
TRUNCATION_SORT_KEY = {"arxiv": "submitted_date", "pubmed": "pdat", "s2": "publication_date"}


# ─────────────────────────────────────────────────────────────────
# 원장 (Ledger)
# ─────────────────────────────────────────────────────────────────
DEFERRED_TTL_DAYS = 14
# 이 값이 바뀌어야 보류분이 재진입한다. 같은 프롬프트로 재분류하면
# 같은 판정이 나오므로 순수 낭비다 (계획 D-4).
CLASSIFY_PROMPT_VERSION = "v5"   # v2: brain_decoding 정의를 "복원·해독"으로
# 좁혔다. v1은 "뇌 신호를 다루면 brain_decoding"이라 연결성·질환분류·감정인식이
# 전부 끌려 들어와 51칸 중 18칸을 낭비했다 (2026-08-31 실측).

# arXiv DataCite DOI. 모든 제출물에 발급되며 저널 DOI와 절대 일치하지
# 않는다. 제외하지 않으면 제목 매칭이 도달 불가능한 죽은 코드가 된다.
ARXIV_DOI_PREFIX = "10.48550/arxiv."


# ─────────────────────────────────────────────────────────────────
# 학회 라벨 (D-13) — 부스트는 select() 이후에만 적용된다
# ─────────────────────────────────────────────────────────────────
# 약칭과 정식 명칭을 모두 넣어야 한다. 골든셋 실측:
#   PeskaVLP → "the 38th Conference on Neural Information Processing Systems"
#   MindLLM  → "Forty-Second International Conference on Machine Learning"
# 둘 다 약칭이 없다. 약칭만 찾는 정규식이면 놓친다.
VENUE_BOOST_LIST = {
    # 학회 — 약칭과 정식 명칭 모두. 약칭 뒤 W는 venue.py가 워크숍으로 처리한다.
    "MICCAI":   ("MICCAI", "Medical Image Computing and Computer[- ]?Assisted Intervention"),
    "IPCAI":    ("IPCAI", "Information Processing in Computer[- ]?Assisted Intervention"),
    "CVPR":     ("CVPR", "Computer Vision and Pattern Recognition"),
    "ICCV":     ("ICCV", "International Conference on Computer Vision"),
    "ECCV":     ("ECCV", "European Conference on Computer Vision"),
    "NeurIPS":  ("NeurIPS", "NIPS", "Conference on Neural Information Processing Systems"),
    "ICML":     ("ICML", "International Conference on Machine Learning"),
    "ICLR":     ("ICLR", "International Conference on Learning Representations"),
    # 저널 — PubMed는 NLM 축약형을 반환한다 ("Med Image Anal", "IEEE Trans Med Imaging").
    # 연결어(of/and/on)를 필수로 두면 축약형이 매칭되지 않는다.
    "MedIA":    ("Medical Image Analysis", "Med(ical)? Image Anal(ysis)?"),
    "IEEE TMI": ("IEEE TMI", r"IEEE Trans(actions)?\.? (on )?Med(ical)? Imaging"),
    "TPAMI":    ("TPAMI", r"IEEE Trans(actions)?\.? (on )?Pattern Anal(ysis)?"),
    "Radiology: AI": ("Radiology:? Artificial Intelligence", "Radiol(ogy)?:? Artif(icial)? Intell(igence)?"),
    "IJCARS":   ("IJCARS",
                 r"Int(ernational)?\.? J(ournal)?\.? (of )?Comput(er)?\.? Assist(ed)?\.? Radiol(ogy)?\.?( and)? Surg(ery)?"),
}
# 워크숍은 본회의와 수락 기준이 다르다. CVPRW를 CVPR로 취급하면 과대평가다.
VENUE_WORKSHOP_SUFFIXES = ("W", " Workshop", "-W")
# 게재 확정이 아닌 표현 — 부스트하지 않는다
VENUE_NEGATIVE_CONTEXT = (
    "submitted to", "under review", "in submission",
    "extended version of", "follow-up to", "based on our", "dataset from",
)
VENUE_YEAR_TOLERANCE = 1     # [올해-1, 올해+1] 밖이면 참조로 보고 부스트하지 않는다
VENUE_BOOST = 1              # 상한 5, 중첩 없음
STAR_MAX = 5


# ─────────────────────────────────────────────────────────────────
# 추론 (로컬 vLLM)
# ─────────────────────────────────────────────────────────────────
MODEL_PATH = "Qwen/Qwen2.5-32B-Instruct-AWQ"   # 교체는 이 한 줄 (계획 D-18)
# 앞에서부터 비어 있는 장을 고른다. 0번이 점유돼 있으면 1→2→3으로 넘어간다.
# 한 장만 쓰는 것은 그대로다 — 여러 장을 동시에 잡으면 연구 작업을 밀어낸다 (원칙 5).
GPU_DEVICES = ("0", "1", "2", "3")
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 8192          # 분류 프롬프트가 분류체계 전문을 담아 ~1.3k 토큰
# 같은 입력이면 같은 페이지가 나와야 한다 (원칙 4)
TEMPERATURE = 0.0
SEED = 42
CLASSIFY_BATCH = 200          # 진행 로그를 남기는 단위. vLLM이 내부에서
# 다시 배치하므로 처리량에는 영향이 없고, 40분짜리 작업이 말없이 도는 것을 막는다
CLASSIFY_MAX_TOKENS = 1024    # 512에서 절삭 발생(골든셋 Meta-CoT) — 실측 후 상향
SUMMARIZE_MAX_TOKENS = 512    # 상한을 올려도 절삭이 안 줄어든다. 모델이 길게
# 쓰는 게 아니라, 값을 다 쓴 뒤 종료 토큰 대신 반복 루프에 빠지기 때문이다
# (실측: 4096 상한에서 원문 50만 자, 그중 쓸모 있는 부분은 앞 360자).
# 정상 출력이 ~300토큰이므로 512면 충분하고, 폭주는 여기서 잘라 버린다.
# 잘린 출력은 llm.salvage_json이 되살리고 summarize.trim_to_sentence가 다듬는다.
# 기동 전 대상 GPU 여유 VRAM이 이 값 미만이면 남의 작업으로 보고 대기·건너뛴다
GPU_FREE_VRAM_REQUIRED_MB = 24_000
GPU_WAIT_RETRIES = 3
GPU_WAIT_SECONDS = 300


# ─────────────────────────────────────────────────────────────────
# 요약 규격 (D-7) — 목적은 "읽고 이해"가 아니라 "열어볼지 판단"
# ─────────────────────────────────────────────────────────────────
SUMMARY_MIN_CHARS = 80       # 검증 하한 (규격 150~250자에 여유)
SUMMARY_MAX_CHARS = 400
SUMMARY_HANGUL_RATIO_MIN = 0.30   # "ASCII만은 아님" 검사는 무력하다
SUMMARY_TAGS_MIN = 1
SUMMARY_TAGS_MAX = 5
ITEM_RETRY_LIMIT = 5         # 초과 시 구조적 오류로 보고 크게 실패한다 (D-10)


# ─────────────────────────────────────────────────────────────────
# 출력
# ─────────────────────────────────────────────────────────────────
DOCS_DIR = "docs"
SITE_TITLE = "Medical AI Daily"

RECENT_DAYS_SHOWN = 14       # 본문 하단 '지난 뉴스'에 띄우는 날짜 수

STALENESS_WARN_DAYS = 2      # 데이터가 이보다 오래되면 페이지에 배너를 띄운다
