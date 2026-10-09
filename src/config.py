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

# 공급이 마른 축이 남긴 유휴 칸을 다른 축에 넘길 때의 천장 배수.
# 최대치를 "완화"하는 것이지 "제거"하는 게 아니다 — 없으면 공급이 한 축에만
# 있는 날 그 축이 페이지를 통째로 가져간다 (실측: 최대치 8인 medical_imaging이
# 40칸 전부를 먹었다).
#
# 2.5인 이유 — 공급 축 개수별 도달 상한을 실측해서 골랐다:
#     배수  1축   2축   3축   4축
#     2.0   10    21    35    40      3축만 남는 날 35건에서 멈춘다
#     2.5   13    27    40    40      ← 채택
#     3.0   16    33    40    40      한 축이 36건까지, 단일 주제 페이지가 된다
# brain_decoding 실공급이 월 5편(progress.md §5)이라 그 축이 0인 날이
# 드물지 않다. 2.0이면 그런 날 35건에 갇히는데, 그 35건이 바로 이 재배분을
# 만들게 한 숫자다. 반대로 3.0은 한 축이 페이지의 90%를 먹을 수 있다.
#
# ⚠️ DAILY_MAX/DAILY_MIN(=1.5) 밑으로 내리면 ceiling이 bounds[1]로 눌려
#    유휴 칸 재배분이 경고 없이 죽는다. test_config.py가 지킨다.
AXIS_OVERFLOW_FACTOR = 2.5

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

    @property
    def ceiling(self) -> int:
        """유휴 칸을 넘겨받았을 때 이 축이 도달할 수 있는 절대 상한.

        bounds[1]은 평상시 상한이고 이건 그 위의 하드 스톱이다. 항상
        bounds[1] 이상이므로 천장이 상한보다 낮아지는 일은 없다.
        """
        return max(self.bounds[1],
                   round(DAILY_MIN * self.ratio * AXIS_OVERFLOW_FACTOR))


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
ARXIV_RETRIES = 4
# 429는 IP 단위 차단이라 3초로는 안 풀린다. 실측: 2026-09-02~03 이틀 연속
# 자동 실행이 429로 죽었다 — 전날 수동 조사에서 호출을 너무 많이 했다.
ARXIV_RATE_LIMIT_BACKOFF_S = 60.0
# 실측(2026-10-03)으로 고른 값이다. max_results=2000은 응답이 4.9MB이고
# 소요가 호출마다 34~65초로 요동쳐 아래 타임아웃을 넘나든다 — 그게 로그에
# 찍힌 TimeoutError의 정체다. 500이면 1.2MB·2~11초로 안정권에 들어온다.
# 페이지가 늘어도 총 시간은 비슷하다(큰 페이지가 그만큼 느리다).
ARXIV_PAGE_SIZE = 500

# 한 요청의 상한. 측정된 최악(2000건 65초)의 두 배를 둔다. 60초였을 때는
# 정상 응답이 타임아웃으로 처리되어 재시도를 돌다 소스가 통째로 죽었다.
ARXIV_TIMEOUT_S = 120

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
# arXiv API의 **하드 한계**다. start>=10000이면 질의와 무관하게 HTTP 500을
# 돌려준다 (실측 2026-10-08: start=9500 OK / 10000·10500 모두 500).
# 이 한계는 상한과 별개로 페이징 자체를 끊어야 한다 — 중복제거로 수집 개수가
# 오프셋보다 적게 늘면 상한에 못 닿은 채로 start가 10000을 밟는다
# (실측: 페이지당 중복 1건이면 9,999건에서 start=10000 요청이 나간다).
ARXIV_START_LIMIT = 10_000

# 30일 창의 실제 분량은 13,239건이다(실측 2026-10-09, opensearch:totalResults).
# 상한 10,000은 창의 최신 ~22일만 덮는다 — 남는 꼬리는 절삭된다. 매일 도는
# 동안에는 문제가 없고(오늘 투고분은 오늘 받는다), 절삭이 일어나면 "어디까지
# 덮었는지"를 성공선으로 기록해 다음 실행이 이어받는다.
ARXIV_MAX = 10000
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
# ⚠️ vLLM의 gpu_memory_utilization은 **전체 VRAM 기준**이다. 48GB 장에서
# 0.85는 41.8GB를 요구한다 — 다른 작업이 22GB를 쓰고 있으면 게이트(24GB)는
# 통과하는데 vLLM이 KV 캐시를 못 잡고 죽는다 (실측 2026-10-07·08:
# "여유 27,135MB"로 진입해 ValueError: No available memory for the cache blocks).
# 그래서 이 값을 그대로 쓰지 않고 실제 여유에서 계산한다(gpu_utilization_for).
GPU_MEMORY_UTILIZATION = 0.85     # 장이 통째로 비었을 때의 상한

# 분류 프롬프트가 분류체계 전문을 담아 길다. 내리면 KV 캐시가 넉넉해지지만
# 프롬프트가 안 들어간다 — 실측 2026-10-09 구조적 최악값(초록 절단 상한
# 2,200자 + 긴 제목): 분류 ASCII 3,241토큰·CJK 5,743토큰, 여기에
# CLASSIFY_MAX_TOKENS(1024)가 더 붙는다. 4096은 ASCII에서도 넘친다.
MAX_MODEL_LEN = 8192
# 같은 입력이면 같은 페이지가 나와야 한다 (원칙 4)
TEMPERATURE = 0.0
SEED = 42
CLASSIFY_BATCH = 200          # 진행 로그를 남기는 단위. vLLM이 내부에서
# 다시 배치하므로 처리량에는 영향이 없고, 40분짜리 작업이 말없이 도는 것을 막는다
CLASSIFY_MAX_TOKENS = 1024    # 512에서 절삭 발생(골든셋 Meta-CoT) — 실측 후 상향
SUMMARIZE_MAX_TOKENS = 512    # 상한을 올려도 절삭이 안 줄어든다. 모델이 길게
# 쓰는 게 아니라, 값을 다 쓴 뒤 종료 토큰 대신 반복 루프에 빠지기 때문이다

# 여유에서 util을 계산할 때 남겨 둘 몫. 조회 시점과 실제 할당 사이에 다른
# 작업이 조금 더 잡을 수 있어 꽉 채우면 그 틈에 OOM이 난다.
GPU_FREE_HEADROOM = 0.90

# 게이트는 util 계산과 **정합해야** 한다. 여유에서 util을 계산하므로 여유가
# 적으면 뜨기는 하되 KV 캐시가 말라 동시 시퀀스가 1개로 떨어지고, 분류가
# 몇 배 느려져 GPU를 오래 점유한다. 반대로 게이트를 높이면 아예 못 돈다.
#
# ⚠️ 실측 2026-10-09: 이 서버는 4-way 작업이 장당 ~21.5GB를 잡아 네 장 모두
#    여유가 27GB 선에 고정된다. 게이트를 30,700으로 올렸더니 pick_gpu()가
#    None을 돌려 **매일 100% 건너뛰는** 상태가 됐다. 느린 성공을 영구 미발행과
#    바꾸는 셈이라, 하한은 "돌 수 있는 최소"로 잡고 속도는 포기한다.
#
# 손으로 계산한 상수를 박지 않고 아래에서 유도한다 — MAX_MODEL_LEN을 건드리면
# 게이트가 따라와야 하고, 주석 속 산수는 조용히 낡는다.
GPU_WEIGHTS_MB = 19_000           # Qwen2.5-32B-AWQ 가중치 실측
KV_BYTES_PER_TOKEN = 262_144      # 64층 × KV헤드 8 × head_dim 128 × 2(K,V) × 2B
MIN_CONCURRENT_SEQS = 2           # 이 밑은 사실상 순차 처리다

def _kv_mb_per_seq() -> float:
    return MAX_MODEL_LEN * KV_BYTES_PER_TOKEN / 1e6

GPU_FREE_VRAM_REQUIRED_MB = int(
    (GPU_WEIGHTS_MB + MIN_CONCURRENT_SEQS * _kv_mb_per_seq())
    / GPU_FREE_HEADROOM)
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

# ─────────────────────────────────────────────────────────────────
# Frontier AI — 의료 밖에서 주목받은 AI 논문 (Hugging Face Daily Papers).
# 축 쿼터와 경쟁하지 않는 별도 레인이다. 자기 분야만 보다 큰 흐름을
# 놓치지 않게 하려는 구간이라, 의료 관련성을 따지지 않는다.
# ─────────────────────────────────────────────────────────────────
FRONTIER_WINDOW_DAYS = 30
FRONTIER_PAGES = 6           # 100건씩. 30일이면 6쪽으로 충분하다 (실측 567건)
FRONTIER_MAX = 6             # 별도 상한 — 의료 AI 본문과 별개로 실린다
FRONTIER_FLAGSHIP_UPVOTES = 300
FRONTIER_MAJOR_UPVOTES = 100
FRONTIER_REQUEST_DELAY_S = 0.3

# 이 별점 이상이면 카드에 테두리를 두른다. 5로 두는 이유: 페이지는 상위 49건만
# 싣기 때문에 4로 내리면 전부 테두리를 받아 표시가 무의미해진다 (실측).
STAR_HIGHLIGHT_MIN = 5

RECENT_DAYS_SHOWN = 14       # 본문 하단 '지난 뉴스'에 띄우는 날짜 수

STALENESS_WARN_DAYS = 2      # 데이터가 이보다 오래되면 페이지에 배너를 띄운다
