"""로컬 vLLM 추론 래퍼.

인터페이스를 OpenAI 호환 형태로 통일해둔다 — 1주차 이후 요약만 유료 API로
올릴 때(계획 F-9) 엔드포인트 교체만으로 끝나도록.

⚠️ 이 모듈을 쓰는 **모든 진입점**에 `if __name__ == "__main__":` 가드가
   필요하다. vLLM v1 엔진은 워커를 spawn으로 띄우며 메인 스크립트를 재import
   하는데, 가드가 없으면 자식 프로세스가 파이프라인 전체를 재실행한다.
"""
from __future__ import annotations

import gc
import json
import re
import os
import subprocess
import time
from typing import Any, Iterable, Optional

import config


def gpu_free_mb() -> Optional[dict[str, int]]:
    """장별 여유 VRAM(MB). nvidia-smi가 없거나 실패하면 None."""
    got = _query_gpu("memory.free")
    return {k: v[0] for k, v in got.items()} if got else None


def gpu_total_mb() -> Optional[dict[str, int]]:
    """장별 전체 VRAM(MB). util 계산에 쓴다."""
    got = _query_gpu("memory.total")
    return {k: v[0] for k, v in got.items()} if got else None


def gpu_utilization_for(free_mb: int, total_mb: int) -> float:
    """실제 여유에 맞춘 gpu_memory_utilization.

    vLLM은 이 값을 **전체** VRAM에 곱한 뒤, 거기서 "이미 쓰인 양"을 뺘서 KV
    캐시 예산을 잡는다(vllm/v1/worker/gpu_worker.py determine_available_memory:
    peak_memory += non_torch_allocations, 이 non_torch_allocations가
    (total−free)에서 자기 torch 할당량을 뺀 값이라 **남이 쓰는 몫까지
    품는다**). 그래서 util*total은 "남이 이미 쓰는 양 + 우리가 더 쓸 양"의
    합이어야 한다 — 단순히 여유 비율로 util을 낮추면(실측 2026-11-12,
    tp=2로 처음 이 분기를 실제로 태워서 드러남) 남의 몫을 두 번 빼는 셈이라
    KV 캐시 예산이 음수로 떨어져 죽는다. 장이 반만 비어 있을 때 전역 상수를
    그대로 쓰는 것도 같은 결과다(실측 2026-10-07·08, 그 이틀은 발행이
    0건이었다) — 이번엔 반대 방향(너무 낮음)으로.
    """
    if total_mb <= 0:
        return config.GPU_MEMORY_UTILIZATION
    other_usage_mb = max(0, total_mb - free_mb)
    # 우리가 실제로 쓸 몫: 장이 비었을 때의 상한(GPU_MEMORY_UTILIZATION)과
    # 실제 여유(헤드룸 적용)의 더 작은 쪽. 그래야 비었을 때는 종전과 같은
    # 예산을 쓰고, 좁을 때는 있는 만큼만 요구한다.
    own_budget_mb = min(config.GPU_MEMORY_UTILIZATION * total_mb,
                         free_mb * config.GPU_FREE_HEADROOM)
    return max(0.0, min(1.0, (other_usage_mb + own_budget_mb) / total_mb))


def _query_gpu(*fields: str) -> Optional[dict[str, list[int]]]:
    """nvidia-smi로 장별 수치를 **한 번에** 조회한다. 실패하면 None.

    여러 필드를 한 호출로 받는다 — 나눠 부르면 그 사이에 상황이 바뀌어
    여유와 전체가 서로 다른 시점을 가리키고 util 계산이 어긋난다.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index," + ",".join(fields),
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        got: dict[str, list[int]] = {}
        for line in out.stdout.strip().splitlines():
            idx, *vals = (x.strip() for x in line.split(","))
            got[idx] = [int(v) for v in vals]
        return got or None
    except Exception:       # noqa: BLE001
        return None


def gpu_memory() -> Optional[dict[str, tuple[int, int]]]:
    """장별 (여유, 전체) VRAM(MB). 한 호출로 받아 시점을 일치시킨다."""
    got = _query_gpu("memory.free", "memory.total")
    if not got:
        return None
    return {k: (v[0], v[1]) for k, v in got.items() if len(v) >= 2}


def pick_gpu(log=print) -> Optional[str]:
    """쓸 수 있는 GPU 한 장을 고른다. 없으면 None.

    선호 순서대로(0→1→2→3) 훑어 요구 여유를 처음 만족하는 장을 쓴다.
    가장 여유가 큰 장을 고르지 않는 이유: 순서를 고정해야 같은 상황에서 늘
    같은 장을 잡아, 어느 장이 이 작업용인지 예측 가능해진다.
    """
    free = gpu_free_mb()
    if free is None:
        log("  [llm] nvidia-smi 조회 실패 — 가용성 확인을 건너뛰고 "
            f"GPU{config.GPU_DEVICES[0]}로 진행")
        return config.GPU_DEVICES[0]
    for dev in config.GPU_DEVICES:
        if free.get(dev, 0) >= config.GPU_FREE_VRAM_REQUIRED_MB:
            return dev
    log("  [llm] 여유 " + " / ".join(
        f"GPU{d} {free.get(d, 0):,}MB" for d in config.GPU_DEVICES)
        + f" — 모두 {config.GPU_FREE_VRAM_REQUIRED_MB:,}MB 미만")
    return None


def pick_gpu_plan(log=print) -> Optional[tuple[tuple[str, ...], int]]:
    """쓸 GPU 조합을 고른다 — (사용할 장들, tensor_parallel_size).

    1장 여유로 충분하면 1장만 쓴다 — 가장 적게 건드린다. 그걸로 안 되면
    가중치/KV를 나눠 실을 수 있는 최소 장수까지 넓힌다(config.GPU_TP_CANDIDATES
    순서대로). 다른 사용자가 메모리를 쓰고 있어도 **메모리가 남는 한** 매일
    돌아가게 한다 (결정 2026-11-12).
    """
    dev = pick_gpu(log=log)
    if dev is not None:
        return (dev,), 1

    free = gpu_free_mb()
    if free is None:
        return None     # pick_gpu가 이미 조회 실패 시 GPU0으로 돌려줬다

    for tp in config.GPU_TP_CANDIDATES:
        if tp <= 1 or tp > len(config.GPU_DEVICES):
            continue
        required = config.gpu_required_mb_for(tp)
        usable = [d for d in config.GPU_DEVICES if free.get(d, 0) >= required]
        if len(usable) >= tp:
            return tuple(usable[:tp]), tp

    log("  [llm] 여유 " + " / ".join(
        f"GPU{d} {free.get(d, 0):,}MB" for d in config.GPU_DEVICES)
        + " — 1장으로도, 나눠서도 부족")
    return None


def salvage_json(text: str) -> Optional[dict]:
    """망가진 JSON에서 최대한 많은 필드를 건진다.

    guided decoding + temperature 0 조합에서 모델이 값을 다 쓴 뒤 종료 토큰
    대신 반복 루프에 빠지는 일이 잦다. 실측(골든셋 20편)에서 관찰한 형태:

      ① 문자열이 열린 채 절삭          "요약이 여기서 잘
      ② 문자열을 닫은 뒤 잉여물 폭주   "요약이다. " "  UsageIdUsageId…
      ③ 뒤에 공백 50만 자              (같은 폭주의 다른 형태)

    ①을 먼저 시도한다 — 문자열이 열린 채 끊겼을 때도 ②의 절단 지점(닫힌 tags
    배열)이 유효한 JSON을 만들어내 요약을 잃기 때문이다.

    ②가 전체의 절반이었는데 '마지막 쉼표에서 절단'만으로는 요약 필드가 통째로
    날아갔다. 그래서 한 번 훑으면서 **깊이 1에서 값이 온전히 끝난 지점**을 모아
    두고, 뒤에서부터 닫아 본다. 스키마 property 순서상 뒤 필드일수록 정보가
    많으므로 가장 뒤에서 성공한 것을 쓴다.
    """
    text = (text or "").strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    depth, in_str, esc = 0, False, False
    cuts: list[int] = []
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
                if depth == 1:
                    cuts.append(i + 1)       # 깊이 1의 문자열 값이 닫혔다
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1:
                cuts.append(i + 1)           # 깊이 2의 배열/객체가 닫혔다
        elif ch == "," and depth == 1:
            cuts.append(i)

    # ① 먼저 — 열린 문자열을 그대로 닫는다. 잘린 문장은 호출자가 다듬는다.
    if in_str:
        body = re.sub(r"[\x00-\x1f]", " ", text).rstrip("\\")
        head = body[:body.rfind('"')]
        opened = max(head.rfind("["), head.rfind(","))
        for cand in (body + '"}',
                     head[:opened].rstrip() + "]}" if opened > 0 else None):
            if cand is None:
                continue
            try:
                return json.loads(cand)
            except json.JSONDecodeError:
                pass

    # ② / ③ — 온전히 끝난 지점에서 닫는다
    for cut in reversed(cuts):
        try:
            return json.loads(text[:cut].rstrip().rstrip(",") + "}")
        except json.JSONDecodeError:
            continue

    return None


class LocalLLM:
    """온디맨드 vLLM. 기동 전 GPU 가용성을 확인하고, 끝나면 반드시 반납한다.

    메모리가 남는 한 매일 돌아가는 쪽을 우선한다. 1장 여유로 되면 1장만
    쓰고, 안 되면 가중치/KV를 나눠 실을 수 있는 최소 장수까지 넓힌다
    (결정 2026-11-12) — 끝내 메모리 자체가 부족하면 그날은 건너뛴다.
    """

    def __init__(self, log=print):
        self.log = log
        self._llm = None
        self.device: Optional[tuple[str, ...]] = None
        self.tensor_parallel_size: Optional[int] = None
        self.last_raw: list[str] = []

    # ── 수명주기 ──────────────────────────────────────────────
    def _wait_for_gpu(self) -> Optional[tuple[tuple[str, ...], int]]:
        """쓸 조합이 생길 때까지 기다린다. 끝내 없으면 None."""
        for attempt in range(config.GPU_WAIT_RETRIES):
            plan = pick_gpu_plan(log=self.log)
            if plan is not None:
                devices, tp = plan
                free = gpu_free_mb() or {}
                self.log(f"  [llm] GPU{','.join(devices)} 선택 (tp={tp}, 여유 "
                         + ", ".join(f"{free.get(d, 0):,}MB" for d in devices)
                         + ")")
                return plan
            self.log(f"  [llm] 메모리가 부족해 {config.GPU_WAIT_SECONDS}초 대기 "
                     f"({attempt+1}/{config.GPU_WAIT_RETRIES})")
            time.sleep(config.GPU_WAIT_SECONDS)
        return None

    def __enter__(self) -> "LocalLLM":
        plan = self._wait_for_gpu()
        if plan is None:
            raise RuntimeError(
                f"GPU {', '.join(config.GPU_DEVICES)}번 모두 1장으로도, 나눠서도 "
                "메모리가 부족해 이번 실행을 건너뜁니다.")
        devices, tp = plan
        self.device = devices
        self.tensor_parallel_size = tp
        # vLLM은 CUDA_VISIBLE_DEVICES를 통해서만 장을 고른다. 여러 장을 주면
        # tensor_parallel_size만큼 그 장들에 나눠 올린다.
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)

        # 예산은 전역 상수가 아니라 **이 장들의 실제 여유**에서 계산한다.
        # 상수를 그대로 쓰면 장이 반만 비어 있을 때 없는 메모리를 요구해
        # KV 캐시 단계에서 죽는다 (실측 2026-10-07·08: 이틀 발행 0건).
        mem = gpu_memory() or {}
        pairs = [mem.get(d, (0, 0)) for d in devices]
        if not mem or any(total == 0 for _, total in pairs):
            # 조용히 상수(0.85)로 되돌아가면 10-07·08과 같은 크래시가 나는데
            # 단서가 하나도 안 남는다. 되돌아가더라도 말은 해야 한다.
            util = config.GPU_MEMORY_UTILIZATION
            self.log(f"  [llm] ⚠️  전체 VRAM 조회 실패 — util을 상수 {util:.2f}로 "
                     "쓴다. 장이 비어 있지 않으면 KV 캐시 할당에서 죽을 수 있다")
        else:
            # 여러 장 중 가장 빡빡한 장에 맞춰야 그 장에서 OOM이 안 난다 —
            # vLLM은 모든 장에 같은 gpu_memory_utilization을 적용한다.
            util = min(gpu_utilization_for(free, total) for free, total in pairs)
            if any(free < total for free, total in pairs):
                # 장이 비어 있지 않으면 util이 상수보다 **올라간다** — 남이
                # 이미 쓰는 몫까지 total 기준으로 담아야 하기 때문이다.
                self.log("  [llm] 여유 " + " / ".join(
                    f"{free:,}/{total:,}MB" for free, total in pairs)
                    + f" — 다른 작업 사용량을 반영해 gpu_memory_utilization을 "
                    f"{util:.2f}로 맞춘다")

        from vllm import LLM
        t0 = time.time()
        self._llm = LLM(model=config.MODEL_PATH,
                        tensor_parallel_size=tp,
                        gpu_memory_utilization=util,
                        max_model_len=config.MAX_MODEL_LEN,
                        seed=config.SEED, trust_remote_code=True,
                        # CUDA 그래프 캡처는 배치 크기별로 따로 메모리를 더
                        # 잡는다(실측 2026-11-12: KV 캐시 예산은 통과했는데
                        # compile_or_warm_up_model 단계에서 "CUDA error: out
                        # of memory"). 공유 GPU에서 여유가 좁을 땐 그 추가분을
                        # 예측하기 어려워 꺼둔다 — 배치 작업이라 토큰당 속도
                        # 손실은 감당할 만하다.
                        enforce_eager=True)
        self.log(f"  [llm] 모델 로드 {time.time()-t0:.0f}초 (tp={tp})")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        """GPU를 확실히 반납한다 (AC-14)."""
        if self._llm is None:
            return
        try:
            import torch
            from vllm.distributed import destroy_model_parallel
            try:
                destroy_model_parallel()      # NCCL 리소스 누수 경고 방지
            except Exception:                 # noqa: BLE001
                pass
            del self._llm
            self._llm = None
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:                # noqa: BLE001
            self.log(f"  [llm] 정리 중 경고: {e}")

    # ── 생성 ──────────────────────────────────────────────────
    def chat_json(self, prompts: list[str], schema: dict,
                  max_tokens: int) -> list[Optional[dict]]:
        """스키마를 강제해 JSON을 받는다. 실패 항목은 None으로 돌려준다.

        구조화 출력은 필요조건이지 충분조건이 아니다 — 거부·절삭·파싱 실패
        경로가 남는다. 실패를 버리지 않고 호출자가 '판정 불가'로 다루도록
        None을 돌려준다 (계획 D-6, 원칙 1).
        """
        from vllm import SamplingParams
        kw: dict[str, Any] = dict(temperature=config.TEMPERATURE,
                                  max_tokens=max_tokens, seed=config.SEED)
        try:
            from vllm.sampling_params import GuidedDecodingParams
            kw["guided_decoding"] = GuidedDecodingParams(json=schema)
        except Exception as e:                # noqa: BLE001
            self.log(f"  [llm] guided decoding 불가 ({e}) — 스키마 강제 없이 진행")

        sp = SamplingParams(**kw)
        msgs = [[{"role": "user", "content": p}] for p in prompts]
        outs = self._llm.chat(msgs, sp)

        results: list[Optional[dict]] = []
        self.last_raw = [o.outputs[0].text for o in outs]   # 진단용
        truncated = 0
        for o in outs:
            gen = o.outputs[0]
            if getattr(gen, "finish_reason", None) == "length":
                truncated += 1
            try:
                results.append(json.loads(gen.text.strip()))
            except (json.JSONDecodeError, AttributeError):
                results.append(salvage_json(getattr(gen, "text", "")))
        if truncated:
            # 0이 아니면 max_tokens를 올리라는 신호다. 빈 페이지로 발견하지 말 것.
            self.log(f"  [llm] ⚠️  max_tokens 절삭 {truncated}건 — 상한 상향 검토")
        fail = sum(1 for r in results if r is None)
        if fail:
            self.log(f"  [llm] 파싱 실패 {fail}건 → 판정 불가로 보류 처리")
        return results
