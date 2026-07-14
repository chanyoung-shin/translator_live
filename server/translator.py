"""번역 엔진 — 로컬 모델 + 온라인(Google) 폴백.

로컬 엔진 폴백 체인 (위에서부터 시도, 실패하면 다음):
  1. llama.cpp GGUF  — Qwen3-4B-Instruct-2507 Q4_K_M (기본) / Seed-X-PPO-7B (품질 우선)
  2. transformers    — Qwen3-4B 4bit (bitsandbytes)
  3. NLLB-200        — 전용 번역 모델 (가볍고 빠름, 품질은 다소 직역투)
  4. Google 번역     — deep-translator (온라인)

큐 정책:
  - 확정 문장(final)은 반드시 모두 번역 (FIFO)
  - 부분 자막(partial)은 최신 것 하나만 유지 + 시간 스로틀

whisper가 감지한 언어에 따라 en→ko, ko→en 자동 방향 전환.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from . import config

log = logging.getLogger("livebridge.mt")

LANG_NAME = {"ko": "Korean", "en": "English", "ja": "Japanese", "zh": "Chinese"}
NLLB_CODE = {
    "en": "eng_Latn", "ko": "kor_Hang", "ja": "jpn_Jpan", "zh": "zho_Hans",
    "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn", "vi": "vie_Latn",
}
SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter for business meetings. "
    "The source line comes from real-time speech recognition, so it may contain "
    "mis-recognized words (replaced by similar-sounding ones) or broken grammar. "
    "Use the conversation context to infer what the speaker actually meant, "
    "silently fix likely recognition errors, and translate the line into natural, "
    "fluent {target}. Output ONLY the translation of the [Line to translate] — "
    "no quotes, no notes, no explanations. "
    "Keep technical terms, product names and numbers as-is when appropriate. "
    "For Korean, use polite spoken style (해요체)."
)


def build_user_content(text: str, context) -> str:
    """번역 요청 본문: 최근 대화 맥락 + 번역 대상 문장."""
    ctx = [c for c in (context or []) if c][-6:]
    if not ctx:
        return text
    return ("[Conversation so far]\n" + "\n".join(ctx)
            + "\n\n[Line to translate]\n" + text)


def target_for(src_lang: str) -> str:
    return config.TARGET_LANG_FOR.get(src_lang, config.DEFAULT_TARGET)


# ============================================================
# 백엔드들
# ============================================================
class GoogleBackend:
    """deep-translator 의 Google 웹 번역. 온라인 필요, 즉시 사용 가능."""
    name = "Google 번역 (온라인)"

    def __init__(self):
        from deep_translator import GoogleTranslator
        self._cls = GoogleTranslator

    def translate(self, text: str, src: str, dst: str, context=None) -> str:
        return self._cls(source=src if src in ("en", "ko", "ja") else "auto",
                         target=dst).translate(text)


def _auto_gpu_layers() -> int:
    """남은 VRAM에 맞춰 GPU에 올릴 레이어 수 결정 (부족한데 전부 올리면 네이티브 크래시).
    Qwen3-4B Q4_K_M 기준: 전체 오프로드에 약 3.2GB 필요."""
    setting = str(config.GGUF_GPU_LAYERS).strip().lower()
    if setting not in ("auto", ""):
        return int(setting)
    try:
        import torch
        free = torch.cuda.mem_get_info()[0] / 2**30  # GB
    except Exception:
        return 0  # 확인 불가 → 안전하게 CPU
    if free >= 3.4:
        return -1   # 전부 GPU
    if free >= 2.2:
        return 24
    if free >= 1.5:
        return 12
    return 0        # CPU (32GB RAM이면 문장 단위 번역은 감당 가능)


class LlamaCppBackend:
    """llama.cpp GGUF — 기본 로컬 백엔드. 짧은 문장 번역에 지연이 가장 낮다.

    llama.cpp는 CUDA 오류 시 프로세스를 네이티브 abort로 죽일 수 있어,
    서버 본체를 보호하기 위해 별도 워커 프로세스(server/llm_worker.py)로 돌린다.
    """
    READY_TIMEOUT = 300   # 첫 로드(디스크 캐시 미스 포함) 대기
    REPLY_TIMEOUT = 60

    def __init__(self):
        import llama_cpp  # noqa: F401 — 미설치면 여기서 빠르게 실패해 다음 백엔드로
        from huggingface_hub import hf_hub_download
        preset = config.GGUF_PRESETS[config.GGUF_PRESET]
        log.info("GGUF 다운로드/확인: %s / %s", preset["repo"], preset["file"])
        path = hf_hub_download(preset["repo"], preset["file"])
        self._lock = threading.Lock()
        self._start_worker(path, preset)

    def _start_worker(self, path: str, preset: dict):
        import subprocess
        import sys as _sys
        gpu_layers = _auto_gpu_layers()
        log.info("llama.cpp 워커 시작: %s (gpu_layers=%s)", path, gpu_layers)
        self.proc = subprocess.Popen(
            [_sys.executable, "-m", "server.llm_worker",
             "--model-path", path,
             "--n-gpu-layers", str(gpu_layers),
             "--n-ctx", "2048",
             "--mode", preset["mode"]],
            cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,  # stderr는 콘솔로
            text=True, encoding="utf-8", bufsize=1,
        )
        self._replies: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        try:
            line = self._replies.get(timeout=self.READY_TIMEOUT)
        except queue.Empty:
            self._kill()
            raise RuntimeError("llama.cpp 워커가 제한시간 안에 준비되지 않음")
        if line != "READY":
            self._kill()
            raise RuntimeError(f"llama.cpp 워커 기동 실패: {line}")
        where = "GPU" if gpu_layers != 0 else "CPU"
        self.name = f"로컬 LLM ({preset['file'].split('.gguf')[0]}, llama.cpp/{where})"

    def _reader(self):
        try:
            for raw in self.proc.stdout:
                raw = raw.strip()
                if raw.startswith("@@"):
                    self._replies.put(raw[2:])
        except Exception:
            pass
        self._replies.put('{"ok": false, "error": "워커 프로세스 종료됨"}')

    def _kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def translate(self, text: str, src: str, dst: str, context=None) -> str:
        import json as _json
        with self._lock:
            if self.proc.poll() is not None:
                raise RuntimeError("llama.cpp 워커가 죽어 있음 (VRAM 부족 가능성)")
            self.proc.stdin.write(_json.dumps(
                {"text": text, "src": src, "dst": dst, "context": context or []},
                ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            try:
                reply = _json.loads(self._replies.get(timeout=self.REPLY_TIMEOUT))
            except queue.Empty:
                self._kill()
                raise RuntimeError("llama.cpp 워커 응답 시간 초과")
            if not reply.get("ok"):
                raise RuntimeError(f"워커 번역 실패: {reply.get('error')}")
            return reply["text"]


class HfLlmBackend:
    """transformers + bitsandbytes 4bit — llama.cpp가 없을 때의 폴백."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = config.LLM_MODEL
        log.info("번역 LLM 로드(transformers): %s (4bit=%s)", model_id, config.LLM_4BIT)
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs = {}
        if torch.cuda.is_available():
            if config.LLM_4BIT:
                from transformers import BitsAndBytesConfig
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
                kwargs["device_map"] = "cuda"
            else:
                kwargs["dtype"] = torch.bfloat16
                kwargs["device_map"] = "cuda"
        else:
            kwargs["dtype"] = torch.float32
            kwargs["device_map"] = "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.name = f"로컬 LLM ({model_id.split('/')[-1]}, transformers)"

    def translate(self, text: str, src: str, dst: str, context=None) -> str:
        torch = self._torch
        messages = [
            {"role": "system",
             "content": SYSTEM_PROMPT.format(target=LANG_NAME.get(dst, "Korean"))},
            {"role": "user", "content": build_user_content(text, context)},
        ]
        # transformers 5.x: apply_chat_template 기본이 return_dict=True (BatchEncoding 반환)
        enc = self.tok.apply_chat_template(
            messages, add_generation_prompt=True,
            return_tensors="pt", return_dict=True).to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=min(512, max(48, len(text) * 2)),
                do_sample=False,
                pad_token_id=self.tok.eos_token_id,
            )
        result = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if "</think>" in result:
            result = result.split("</think>")[-1]
        return result.strip().strip('"')


class NllbBackend:
    """facebook/nllb-200 — 전용 번역 모델. 가볍고 빠르지만 다소 직역투."""

    def __init__(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model_id = config.NLLB_MODEL
        log.info("NLLB 로드: %s", model_id)
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=dtype).to(device)
        self.device = device
        self.name = f"NLLB-200 ({model_id.split('/')[-1]}, {device})"

    def translate(self, text: str, src: str, dst: str, context=None) -> str:
        torch = self._torch
        self.tok.src_lang = NLLB_CODE.get(src, "eng_Latn")
        inputs = self.tok(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        bos = self.tok.convert_tokens_to_ids(NLLB_CODE.get(dst, "kor_Hang"))
        with torch.inference_mode():
            out = self.model.generate(**inputs, forced_bos_token_id=bos,
                                      max_new_tokens=512, num_beams=4)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


LOCAL_CHAIN = [
    ("llama.cpp GGUF", LlamaCppBackend),
    ("transformers 4bit", HfLlmBackend),
    ("NLLB", NllbBackend),
    ("Google", GoogleBackend),
]

# 백엔드는 전역 캐시로 공유한다 — 파이프라인 재시작(소스 변경 등)마다 새 Translator가
# 만들어지더라도 무거운 모델을 다시 로드하거나 VRAM에 이중 적재하지 않도록.
_BACKEND_CACHE: dict = {}
_BACKEND_LOCK = threading.Lock()


# ============================================================
# 번역 매니저 (워커 스레드 + 큐)
# ============================================================
@dataclass
class Job:
    id: str
    text: str
    src: str
    is_final: bool
    context: list = None


class Translator:
    def __init__(self, on_result: Callable[[str, str, bool], None],
                 on_status: Optional[Callable[[str], None]] = None):
        """on_result(job_id, translated_text, is_final)"""
        self.on_result = on_result
        self.on_status = on_status or (lambda s: None)
        self._final_q: "queue.Queue[Job]" = queue.Queue()
        self._partial_lock = threading.Lock()
        self._partial_job: Optional[Job] = None
        self._last_partial_t = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend = None
        self._backend_name = "미로드"
        self._engine = config.TRANSLATE_ENGINE_DEFAULT

    # ---------- 백엔드 로드 ----------
    def set_engine(self, engine: str):
        if engine in ("local", "google") and engine != self._engine:
            self._engine = engine
            self._backend = None  # 다음 번역 때 재로드

    def _ensure_backend(self):
        if self._backend is not None:
            return self._backend
        with _BACKEND_LOCK:  # 전역 락: 동시 이중 로드 방지 (VRAM 보호)
            cached = _BACKEND_CACHE.get(self._engine)
            if cached is not None:
                self._backend = cached
            elif self._engine == "google":
                self.on_status("번역 엔진 준비 중 (Google)…")
                self._backend = GoogleBackend()
            else:
                # 같은 프로세스 안의 무거운 CUDA 로드(HfLlm/NLLB)는 whisper 추론과
                # 겹치지 않게 직렬화. llama.cpp는 별도 워커 프로세스라 제외
                # (락을 잡으면 로드 동안 자막이 멎는다).
                from .transcriber import _infer_lock as _asr_lock
                import contextlib
                for label, cls in LOCAL_CHAIN:
                    if self._stop.is_set():
                        raise RuntimeError("중지됨")
                    try:
                        self.on_status(f"번역 모델 준비 중 ({label})… 첫 실행은 다운로드로 오래 걸릴 수 있어요")
                        guard = contextlib.nullcontext() if cls is LlamaCppBackend else _asr_lock
                        with guard:
                            self._backend = cls()
                        break
                    except Exception as e:
                        log.warning("번역 백엔드 %s 사용 불가: %s", label, e)
                if self._backend is None:
                    raise RuntimeError("사용 가능한 번역 백엔드가 없습니다")
            _BACKEND_CACHE[self._engine] = self._backend
        self._backend_name = self._backend.name
        self.on_status(f"번역 준비 완료: {self._backend_name}")
        return self._backend

    def _demote_to_google(self):
        """번역이 반복 실패하면 세션이 통째로 죽지 않게 Google(온라인)로 강등."""
        try:
            google = GoogleBackend()
        except Exception:
            return
        with _BACKEND_LOCK:
            _BACKEND_CACHE[self._engine] = google
        self._backend = google
        self._backend_name = google.name
        log.error("로컬 번역이 반복 실패해 Google 번역으로 전환합니다")
        self.on_status("로컬 번역 오류가 반복돼 Google 번역(온라인)으로 전환했어요")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def preload(self):
        """시작 시 미리 로드 (블로킹)."""
        try:
            self._ensure_backend()
        except Exception as e:
            log.error("번역 백엔드 프리로드 실패: %s", e)

    # ---------- 작업 제출 ----------
    def submit_final(self, job_id: str, text: str, src_lang: str, context: list = None):
        self._final_q.put(Job(job_id, text, src_lang, True, context))

    def submit_partial(self, job_id: str, text: str, src_lang: str, context: list = None):
        with self._partial_lock:
            self._partial_job = Job(job_id, text, src_lang, False, context)

    # ---------- 워커 ----------
    def _worker(self):
        fails = 0
        while True:
            # 중지되어도 확정 문장 큐는 끝까지 비운다 (마지막 문장 번역 유실 방지)
            if self._stop.is_set() and self._final_q.empty():
                break
            job: Optional[Job] = None
            try:
                job = self._final_q.get(timeout=0.15)
            except queue.Empty:
                if self._stop.is_set():
                    continue
                now = time.monotonic()
                if now - self._last_partial_t >= config.PARTIAL_TRANSLATE_INTERVAL:
                    with self._partial_lock:
                        job, self._partial_job = self._partial_job, None
                    if job is not None:
                        self._last_partial_t = now
            if job is None:
                continue
            try:
                backend = self._ensure_backend()
                dst = target_for(job.src)
                if job.src == dst:
                    result = job.text
                else:
                    t0 = time.monotonic()
                    result = backend.translate(job.text, job.src, dst, context=job.context)
                    log.debug("번역 %.2fs [%s→%s] %s", time.monotonic() - t0, job.src, dst, result[:40])
                fails = 0
                self.on_result(job.id, result, job.is_final)
            except Exception as e:
                log.error("번역 오류: %s", e)
                fails += 1
                if fails >= 2 and not isinstance(self._backend, GoogleBackend):
                    self._demote_to_google()
                    if job.is_final:
                        self._final_q.put(job)  # 새 백엔드로 재시도
                        continue
                if job.is_final:
                    self.on_result(job.id, "(번역 실패 — 엔진을 확인해 주세요)", True)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="translator", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=15)  # 남은 확정 번역 드레인 대기
            self._thread = None
