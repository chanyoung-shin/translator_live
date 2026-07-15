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


def build_user_content(text: str, context, target: str = None) -> str:
    """번역 요청 본문: 최근 대화 맥락 + 번역 대상 문장 (+ 명시적 지시).

    지시문을 문장 바로 뒤에 붙이는 이유: 시스템 프롬프트만으로는 일부 모델
    (특히 nothink 프리필 경로의 Qwen3-8B)이 번역하지 않고 원문을 복사한다 (실측).
    """
    ctx = [c for c in (context or []) if c][-6:]
    parts = []
    if ctx:
        parts.append("[Conversation so far]\n" + "\n".join(ctx))
    parts.append("[Line to translate]\n" + text)
    if target:
        parts.append(f"Translate the [Line to translate] into natural {target}. "
                     "Fix likely speech-recognition errors (similar-sounding wrong words) "
                     "using the context. Output only the translation.")
    return "\n\n".join(parts)


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


def _auto_gpu_layers(preset: dict) -> int:
    """남은 VRAM에 맞춰 GPU에 올릴 레이어 수 결정 (부족한데 전부 올리면 네이티브 크래시)."""
    setting = str(config.GGUF_GPU_LAYERS).strip().lower()
    if setting not in ("auto", ""):
        return int(setting)
    need = preset.get("vram_full_gb", 3.4)
    layers = preset.get("layers", 36)
    try:
        import torch
        free = torch.cuda.mem_get_info()[0] / 2**30  # GB
    except Exception:
        return 0  # 확인 불가 → 안전하게 CPU
    if free >= need:
        return -1                    # 전부 GPU
    if free >= need * 0.66:
        return int(layers * 2 / 3)
    if free >= need * 0.45:
        return int(layers / 3)
    return 0        # CPU (32GB RAM이면 문장 단위 번역은 감당 가능)


class _GgufWorker:
    """llm_worker.py 서브프로세스 1개 관리 (기동/요청/정리).

    llama.cpp는 CUDA 오류 시 프로세스를 네이티브 abort로 죽일 수 있어,
    서버 본체를 보호하기 위해 반드시 별도 프로세스로 돌린다.
    """
    READY_TIMEOUT = 600   # 첫 로드(다운로드 포함 가능) 대기

    def __init__(self, preset: dict):
        import subprocess
        import sys as _sys
        if "path" in preset:  # 커스텀 GGUF (models/ 폴더)
            path = str(preset["path"])
        else:
            from huggingface_hub import hf_hub_download
            log.info("GGUF 다운로드/확인: %s / %s", preset["repo"], preset["file"])
            path = hf_hub_download(preset["repo"], preset["file"])
        self.gpu_layers = _auto_gpu_layers(preset)
        log.info("llama.cpp 워커 시작: %s (mode=%s, gpu_layers=%s)",
                 preset["label"], preset["mode"], self.gpu_layers)
        cmd = [_sys.executable, "-m", "server.llm_worker",
               "--model-path", path,
               "--n-gpu-layers", str(self.gpu_layers),
               "--n-ctx", str(preset.get("n_ctx", 2048)),
               "--mode", preset["mode"]]
        if preset.get("nothink"):
            cmd.append("--nothink")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(__import__("pathlib").Path(__file__).resolve().parent.parent),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,  # stderr는 콘솔로
            text=True, encoding="utf-8", bufsize=1,
        )
        self._replies: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()
        try:
            line = self._replies.get(timeout=self.READY_TIMEOUT)
        except queue.Empty:
            self.kill()
            raise RuntimeError("llama.cpp 워커가 제한시간 안에 준비되지 않음")
        if line != "READY":
            self.kill()
            raise RuntimeError(f"llama.cpp 워커 기동 실패: {line}")

    def _reader(self):
        try:
            for raw in self.proc.stdout:
                raw = raw.strip()
                if raw.startswith("@@"):
                    self._replies.put(raw[2:])
        except Exception:
            pass
        self._replies.put('{"ok": false, "error": "워커 프로세스 종료됨"}')

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def request(self, payload: dict, timeout: float = 60) -> dict:
        import json as _json
        if self.proc.poll() is not None:
            raise RuntimeError("llama.cpp 워커가 죽어 있음 (VRAM 부족 가능성)")
        self.proc.stdin.write(_json.dumps(payload, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()
        try:
            reply = _json.loads(self._replies.get(timeout=timeout))
        except queue.Empty:
            self.kill()
            raise RuntimeError("llama.cpp 워커 응답 시간 초과")
        if not reply.get("ok"):
            raise RuntimeError(f"워커 처리 실패: {reply.get('error')}")
        return reply


def resolve_preset(key: str) -> dict:
    """프리셋 키 → 설정 딕셔너리. "file:이름.gguf" 는 models/ 폴더의 커스텀 모델."""
    if key.startswith("file:"):
        name = key[5:]
        p = config.MODELS_DIR / name
        if not p.is_file():
            raise FileNotFoundError(f"models 폴더에 {name} 파일이 없습니다")
        size_gb = p.stat().st_size / 2**30
        return {
            "path": p,
            "mode": "chat",     # GGUF에 내장된 채팅 템플릿 사용
            "n_ctx": 4096,
            "layers": 999,      # 레이어 수 미상 → 사실상 전부 GPU 또는 CPU 이분
            "vram_full_gb": size_gb * 1.25 + 0.5,
            "label": name,
        }
    return config.GGUF_PRESETS[key]


class LlamaCppBackend:
    """llama.cpp GGUF — 기본 로컬 백엔드. 짧은 문장 번역에 지연이 가장 낮다.

    Seed-X(번역 특화, 채팅 불가) 선택 시에는 맥락 기반 오전사 보정을 대신할
    보정 워커를 앞단에 함께 띄운다:
      ASR 문장 → [보정 워커: 맥락으로 오인식 수정] → [Seed-X: 번역]
    채팅형 모델(Qwen 등)은 보정+번역을 한 호출에 처리 (실시간성 우선).
    """

    def __init__(self, preset_key: Optional[str] = None):
        import llama_cpp  # noqa: F401 — 미설치면 여기서 빠르게 실패해 다음 백엔드로
        self.preset_key = preset_key or config.GGUF_PRESET
        preset = resolve_preset(self.preset_key)
        self._preset = preset  # 워커 사망 시 재시작용
        self._lock = threading.Lock()
        self._trans = _GgufWorker(preset)
        self._corr: Optional[_GgufWorker] = None
        if preset["mode"] == "seedx":
            try:
                self._corr = _GgufWorker(config.CORRECTOR_GGUF)
            except Exception as e:
                log.warning("보정 모델 로드 실패(%s) — 보정 없이 번역만 합니다", e)
        where = "GPU" if self._trans.gpu_layers != 0 else "CPU"
        corr_tag = " + 보정AI" if self._corr else ""
        self.name = f"로컬 LLM ({preset['label']}{corr_tag}, llama.cpp/{where})"

    _PROMOTE_CHECK_SEC = 60.0  # CPU로 밀려난 워커의 GPU 복귀 확인 주기

    def _ensure_alive(self):
        """워커가 죽어 있으면(다른 앱의 VRAM 점유 등) 현재 GPU 여유에 맞춰 재시작.
        재시작 시 _auto_gpu_layers가 다시 계산되므로, VRAM이 부족해졌으면
        자동으로 일부/전부 CPU로 내려간 채 살아난다."""
        if self._trans.proc.poll() is not None:
            log.warning("번역 워커 사망 감지 → 재시작 (GPU 여유에 맞춰 재배치)")
            self._trans = _GgufWorker(self._preset)
            where = "GPU" if self._trans.gpu_layers != 0 else "CPU"
            log.info("번역 워커 재시작 완료 (%s)", where)
        elif self._trans.gpu_layers != -1:
            # VRAM 부족으로 CPU/부분 오프로드에 머물러 있으면 번역이 계속 느리다
            # → 주기적으로 확인해서 GPU 여유가 돌아왔으면 GPU로 재승격
            now = time.monotonic()
            if now - getattr(self, "_last_promote_check", 0) >= self._PROMOTE_CHECK_SEC:
                self._last_promote_check = now
                if _auto_gpu_layers(self._preset) == -1:
                    log.info("GPU 여유 회복 감지 → 번역 워커를 GPU로 재승격")
                    self._trans.kill()
                    self._trans = _GgufWorker(self._preset)
        if self._corr is not None and self._corr.proc.poll() is not None:
            self._corr = None  # 보정기는 없어도 동작하므로 조용히 비활성화

    def close(self):
        """모델 전환 시 워커를 내려 VRAM 회수."""
        self._trans.kill()
        if self._corr:
            self._corr.kill()

    def translate(self, text: str, src: str, dst: str, context=None) -> str:
        with self._lock:
            self._ensure_alive()
            if self._corr is not None and context:
                try:
                    fixed = self._corr.request(
                        {"text": text, "context": context, "src": src, "dst": dst},
                        timeout=30)["text"].strip()
                    if fixed:
                        if fixed != text:
                            log.debug("보정: %r → %r", text[:40], fixed[:40])
                        text = fixed
                except Exception as e:
                    log.warning("오전사 보정 실패(원문으로 진행): %s", e)
            reply = self._trans.request(
                {"text": text, "src": src, "dst": dst, "context": context or []})
            return reply["text"]

    def summarize(self, text: str, combine: bool = False) -> str:
        """회의 전사 요약 (한국어). combine=True면 부분 요약들을 병합."""
        with self._lock:
            self._ensure_alive()
            reply = self._trans.request(
                {"task": "summarize", "text": text, "combine": combine}, timeout=300)
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
            {"role": "user",
             "content": build_user_content(text, context, LANG_NAME.get(dst, "Korean"))},
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
    # (표시명, 팩토리(preset_key), ASR 추론 락 필요 여부 — 같은 프로세스 CUDA 로드만 True)
    ("llama.cpp GGUF", lambda preset: LlamaCppBackend(preset), False),
    ("transformers 4bit", lambda preset: HfLlmBackend(), True),
    ("NLLB", lambda preset: NllbBackend(), True),
    ("Google", lambda preset: GoogleBackend(), False),
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
        self._model_preset = config.GGUF_PRESET

    # ---------- 백엔드 로드 ----------
    @property
    def engine(self) -> str:
        return self._engine

    def set_engine(self, engine: str):
        if engine in ("local", "google") and engine != self._engine:
            self._engine = engine
            self._backend = None  # 다음 번역 때 재로드

    def set_model(self, preset: str):
        """로컬 번역 모델 전환 (프리셋 키 또는 "file:이름.gguf")."""
        if preset == self._model_preset:
            return
        try:
            label = resolve_preset(preset)["label"]
        except Exception as e:
            log.warning("모델 전환 불가 (%s): %s", preset, e)
            return
        self._model_preset = preset
        self._backend = None  # 다음 번역 때 새 모델 로드
        self.on_status(f"번역 모델을 {label}(으)로 전환 — 다음 문장부터 적용돼요")

    def summarize(self, text: str, combine: bool = False) -> str:
        backend = self._ensure_backend()
        if not hasattr(backend, "summarize"):
            raise RuntimeError("현재 번역 엔진은 요약을 지원하지 않습니다 — 설정에서 로컬 AI 모델을 선택해 주세요")
        return backend.summarize(text, combine=combine)

    def _cache_key(self) -> str:
        return "google" if self._engine == "google" else f"local:{self._model_preset}"

    def _evict_other_local(self, keep_key: str):
        """다른 프리셋의 llama.cpp 워커를 내려 VRAM 회수 (8GB에 7B+4B 동시 적재 방지)."""
        for key in list(_BACKEND_CACHE):
            if key.startswith("local:") and key != keep_key:
                backend = _BACKEND_CACHE.pop(key)
                if hasattr(backend, "close"):
                    try:
                        backend.close()
                        log.info("이전 번역 워커 종료: %s", key)
                    except Exception:
                        pass

    def _ensure_backend(self):
        if self._backend is not None:
            return self._backend
        key = self._cache_key()
        with _BACKEND_LOCK:  # 전역 락: 동시 이중 로드 방지 (VRAM 보호)
            cached = _BACKEND_CACHE.get(key)
            if cached is not None:
                self._backend = cached
            elif self._engine == "google":
                self.on_status("번역 엔진 준비 중 (Google)…")
                self._backend = GoogleBackend()
            else:
                self._evict_other_local(key)
                # 같은 프로세스 안의 무거운 CUDA 로드(HfLlm/NLLB)는 whisper 추론과
                # 겹치지 않게 직렬화. llama.cpp는 별도 워커 프로세스라 제외
                # (락을 잡으면 로드 동안 자막이 멎는다).
                from .transcriber import _infer_lock as _asr_lock
                import contextlib
                for label, factory, needs_asr_lock in LOCAL_CHAIN:
                    if self._stop.is_set():
                        raise RuntimeError("중지됨")
                    try:
                        self.on_status(f"번역 모델 준비 중 ({label})… 첫 실행은 다운로드로 오래 걸릴 수 있어요")
                        guard = _asr_lock if needs_asr_lock else contextlib.nullcontext()
                        with guard:
                            self._backend = factory(self._model_preset)
                        break
                    except Exception as e:
                        log.warning("번역 백엔드 %s 사용 불가: %s", label, e)
                if self._backend is None:
                    raise RuntimeError("사용 가능한 번역 백엔드가 없습니다")
            _BACKEND_CACHE[key] = self._backend
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
            _BACKEND_CACHE[self._cache_key()] = google
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
                # 워커가 CPU로 밀려나 있으면(문장당 수 초) 부분 번역은 생략하고
                # 확정 번역에 자원 집중 — 큐가 밀려 몰아서 출력되는 것 방지
                if not job.is_final:
                    trans = getattr(backend, "_trans", None)
                    if trans is not None and trans.gpu_layers == 0:
                        continue
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
