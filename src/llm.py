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

    vLLM은 이 값을 **전체** VRAM에 곱해 예산을 잡는다. 전역 상수를 그대로
    쓰면 장이 반만 비어 있을 때 있지도 않은 메모리를 달라고 해서 KV 캐시
    할당 단계에서 죽는다 (실측 2026-10-07·08, 그 이틀은 발행이 0건이었다).
    여유에서 거꾸로 계산하면 남은 만큼만 쓰고 연구 작업을 밀어내지 않는다.
    """
    if total_mb <= 0:
        return config.GPU_MEMORY_UTILIZATION
    usable = free_mb * config.GPU_FREE_HEADROOM
    return max(0.0, min(config.GPU_MEMORY_UTILIZATION, usable / total_mb))


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

    GPU는 연구용이 우선이다 (원칙 5). 대상 GPU가 점유 중이면 남의 작업을
    밀어내지 않고 대기했다가, 끝내 비지 않으면 그날은 건너뛴다.
    """

    def __init__(self, log=print):
        self.log = log
        self._llm = None
        self.device: Optional[str] = None
        self.last_raw: list[str] = []

    # ── 수명주기 ──────────────────────────────────────────────
    def _wait_for_gpu(self) -> Optional[str]:
        """쓸 장이 생길 때까지 기다린다. 끝내 없으면 None."""
        for attempt in range(config.GPU_WAIT_RETRIES):
            dev = pick_gpu(log=self.log)
            if dev is not None:
                free = (gpu_free_mb() or {}).get(dev)
                self.log(f"  [llm] GPU{dev} 선택"
                         + (f" (여유 {free:,}MB)" if free else ""))
                return dev
            self.log(f"  [llm] 연구 작업 중으로 보고 {config.GPU_WAIT_SECONDS}초 대기 "
                     f"({attempt+1}/{config.GPU_WAIT_RETRIES})")
            time.sleep(config.GPU_WAIT_SECONDS)
        return None

    def __enter__(self) -> "LocalLLM":
        dev = self._wait_for_gpu()
        if dev is None:
            raise RuntimeError(
                f"GPU {', '.join(config.GPU_DEVICES)}번이 모두 점유 중이라 이번 실행을 "
                "건너뜁니다. 연구 작업을 밀어내지 않습니다.")
        self.device = dev
        # vLLM은 CUDA_VISIBLE_DEVICES를 통해서만 장을 고른다. 이 값을 설정하면
        # 프로세스 안에서 선택한 장이 cuda:0이 된다.
        os.environ["CUDA_VISIBLE_DEVICES"] = dev

        # 예산은 전역 상수가 아니라 **이 장의 실제 여유**에서 계산한다.
        # 상수를 그대로 쓰면 장이 반만 비어 있을 때 없는 메모리를 요구해
        # KV 캐시 단계에서 죽는다 (실측 2026-10-07·08: 이틀 발행 0건).
        free, total = (gpu_memory() or {}).get(dev, (0, 0))
        util = gpu_utilization_for(free, total)
        if not total:
            # 조용히 상수(0.85)로 되돌아가면 10-07·08과 같은 크래시가 나는데
            # 단서가 하나도 안 남는다. 되돌아가더라도 말은 해야 한다.
            self.log(f"  [llm] ⚠️  전체 VRAM 조회 실패 — util을 상수 {util:.2f}로 "
                     "쓴다. 장이 비어 있지 않으면 KV 캐시 할당에서 죽을 수 있다")
        elif util < config.GPU_MEMORY_UTILIZATION:
            self.log(f"  [llm] 여유 {free:,}/{total:,}MB — "
                     f"gpu_memory_utilization을 {util:.2f}로 낮춰 띄운다 "
                     "(연구 작업을 밀어내지 않는다)")

        from vllm import LLM
        t0 = time.time()
        self._llm = LLM(model=config.MODEL_PATH,
                        tensor_parallel_size=config.TENSOR_PARALLEL_SIZE,
                        gpu_memory_utilization=util,
                        max_model_len=config.MAX_MODEL_LEN,
                        seed=config.SEED, trust_remote_code=True)
        self.log(f"  [llm] 모델 로드 {time.time()-t0:.0f}초")
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
