"""스트리밍 음성 인식 — faster-whisper + Silero VAD + LocalAgreement 안정화.

whisper는 스트리밍 모델이 아니므로:
  1. '지금 말하는 중인 구간'을 주기적으로 다시 인식해 부분(파티셜) 자막을 낸다
  2. 연속된 부분 결과의 공통 접두어(LocalAgreement)만 '안정된 텍스트'로 취급한다
  3. 침묵(600ms)이 감지되면 beam=5 + 직전 확정문 프롬프트로 최종 인식해 확정한다

주의: WASAPI 루프백은 무음 중 오디오 콜백이 아예 오지 않으므로,
침묵 시간은 tick()을 통해 벽시계 기준으로도 흘러가야 한다.

무거운 whisper 모델은 전역 1개만 로드해 락으로 공유한다 (8GB VRAM 절약).
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import config

log = logging.getLogger("livebridge.asr")

_model = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def get_model():
    """whisper 모델 전역 싱글턴 (지연 로드)."""
    global _model
    with _model_lock:
        if _model is None:
            # Windows: torch 임포트가 CUDA/cuDNN DLL 경로를 등록해 준다
            # (ctranslate2가 cudnn_ops64_9.dll 등을 torch\lib 에서 찾게 됨)
            try:
                import torch  # noqa: F401
            except ImportError:
                pass
            from faster_whisper import WhisperModel
            log.info("faster-whisper 로드: %s (%s/%s)",
                     config.ASR_MODEL, config.ASR_DEVICE, config.ASR_COMPUTE)
            try:
                _model = WhisperModel(config.ASR_MODEL, device=config.ASR_DEVICE,
                                      compute_type=config.ASR_COMPUTE)
            except Exception as e:
                # 내장그래픽/무GPU 환경: large 모델은 CPU에서 실시간을 못 따라가므로
                # (사용자가 모델을 명시하지 않았다면) small로 자동 축소 + 갱신 주기 완화
                model_name = config.ASR_MODEL
                if config.ASR_MODEL_IS_DEFAULT:
                    model_name = "small"
                    config.PARTIAL_INTERVAL_SEC = max(config.PARTIAL_INTERVAL_SEC, 1.5)
                    log.warning("GPU 사용 불가(%s) — CPU 모드: ASR을 small로 자동 전환, "
                                "부분자막 주기 1.5초", e)
                else:
                    log.warning("GPU 로드 실패(%s) — CPU(int8)로 %s 유지", e, model_name)
                _model = WhisperModel(model_name, device="cpu", compute_type="int8")
        return _model


# ---------- VAD (Silero, faster-whisper 내장 onnx — torch 불필요) ----------
class Vad:
    def __init__(self):
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        self._get_ts = get_speech_timestamps
        self._opts = VadOptions(threshold=config.VAD_THRESHOLD,
                                min_speech_duration_ms=100,
                                min_silence_duration_ms=100)

    def has_speech(self, chunk: np.ndarray) -> bool:
        try:
            return len(self._get_ts(chunk, self._opts)) > 0
        except Exception:
            # VAD 실패 시 에너지 기반 폴백
            return float(np.sqrt(np.mean(chunk ** 2))) > 0.01


# ---------- 환각/노이즈 필터 ----------
_norm_re = re.compile(r"[\s​]+")


def _norm(text: str) -> str:
    return _norm_re.sub(" ", text).strip().lower()


def looks_hallucinated(text: str) -> bool:
    t = _norm(text)
    if not t or t in {p.lower() for p in config.HALLUCINATION_PHRASES}:
        return True
    # 같은 단어의 과도한 반복 = 전형적 whisper 환각
    words = t.split()
    if len(words) >= 8 and len(set(words)) <= max(1, len(words) // 8):
        return True
    return False


def _common_prefix_len(a: list, b: list) -> int:
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i


_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def effective_lang(text: str, detected: Optional[str]) -> str:
    """whisper 언어 감지는 짧은 구간에서 자주 틀린다 — 실제 문자로 보정.
    (E2E 테스트에서 영어 문장이 ko로 감지되는 사례 확인됨)"""
    if _HANGUL.search(text):
        return "ko"
    if detected == "ko" and _LATIN.search(text):
        return "en"  # 한글이 전혀 없는데 ko 감지 → 오탐
    return detected or "en"


@dataclass
class Utterance:
    id: str
    source: str          # "system" | "mic"
    text: str            # 전체 텍스트
    stable: str          # LocalAgreement로 안정된 접두어 (partial일 때만 의미)
    lang: str
    time: float          # epoch
    is_final: bool


@dataclass
class SegmentAssembler:
    """오디오 청크를 발화 단위로 모으고, 부분/확정 인식 결과를 콜백으로 전달.

    feed(chunk) : 16kHz float32 오디오 공급 (약 BLOCK_SEC 단위)
    tick(dt)    : 오디오가 안 들어올 때 시간 경과 알림 (루프백 무음 대응)
    """
    source: str
    on_partial: Callable[[Utterance], None]
    on_final: Callable[[Utterance], None]

    _buf: list = field(default_factory=list)
    _pre_roll: list = field(default_factory=list)
    _active: bool = False
    _silence_sec: float = 0.0          # 확정 트리거용: 버퍼 침묵 + 벽시계(tick) 침묵
    _buffered_silence_sec: float = 0.0  # 트림용: 실제로 버퍼에 들어간 침묵만
    _speech_sec: float = 0.0
    _last_partial_t: float = 0.0
    _prev_partial_words: list = field(default_factory=list)
    _confirmed_tail: str = ""      # 직전 확정 문장들 (final 패스 프롬프트용)
    _counter: int = 0
    _seg_seq: int = 0              # 세그먼트 일련번호 (부분 자막/번역 식별용)
    _seg_lang: Optional[str] = None  # 세그먼트 내 고정 언어 (부분 자막 흔들림 방지)

    def __post_init__(self):
        self._vad = Vad()

    # ----- 시간 경과 (오디오 없음 = 무음) -----
    def tick(self, dt: float):
        if not self._active:
            return
        self._silence_sec += dt
        if self._silence_sec * 1000 >= config.SILENCE_FINALIZE_MS:
            self._finalize()

    # ----- 청크 공급 -----
    def feed(self, chunk: np.ndarray):
        speech = self._vad.has_speech(chunk)
        dur = len(chunk) / config.TARGET_SR

        if not self._active:
            if speech:
                self._active = True
                self._seg_seq += 1
                self._seg_lang = None
                self._buf = list(self._pre_roll) + [chunk]
                self._pre_roll = []
                self._speech_sec = dur
                self._silence_sec = 0.0
                self._buffered_silence_sec = 0.0
                self._last_partial_t = time.monotonic()
                self._prev_partial_words = []
            else:
                self._pre_roll.append(chunk)
                keep = int(config.PRE_ROLL_SEC / config.BLOCK_SEC) + 1
                self._pre_roll = self._pre_roll[-keep:]
            return

        self._buf.append(chunk)
        if speech:
            self._speech_sec += dur
            self._silence_sec = 0.0
            self._buffered_silence_sec = 0.0
        else:
            self._silence_sec += dur
            self._buffered_silence_sec += dur

        total = sum(len(c) for c in self._buf) / config.TARGET_SR
        if self._silence_sec * 1000 >= config.SILENCE_FINALIZE_MS or total >= config.MAX_SEGMENT_SEC:
            self._finalize()
        elif time.monotonic() - self._last_partial_t >= config.PARTIAL_INTERVAL_SEC:
            self._last_partial_t = time.monotonic()
            self._emit_partial()

    # ----- 인식 -----
    def _transcribe(self, audio: np.ndarray, final: bool):
        model = get_model()
        # 부분 자막: 세그먼트 내 고정 언어 사용 (짧은 오디오의 감지 흔들림으로
        # 자막이 zh/ko/en 사이를 오가는 것 방지). 확정은 전체 오디오로 재감지.
        kwargs = dict(
            language=None if final else self._seg_lang,
            task="transcribe",
            without_timestamps=True,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            no_speech_threshold=config.NO_SPEECH_THRESHOLD,
            log_prob_threshold=config.LOG_PROB_THRESHOLD,
            compression_ratio_threshold=config.COMPRESSION_RATIO_THRESHOLD,
        )
        if final:
            # 직전 확정 문장 + (설정 시) 도메인 용어를 프롬프트로 → 인식 정확도 향상
            prompt = self._confirmed_tail[-200:]
            if config.VOCAB:
                prompt = (config.VOCAB + ". " + prompt)[:320]
            kwargs.update(beam_size=config.ASR_BEAM_FINAL,
                          condition_on_previous_text=True,
                          initial_prompt=prompt or None)
        else:
            kwargs.update(beam_size=config.ASR_BEAM_PARTIAL, best_of=1,
                          condition_on_previous_text=False)
        with _infer_lock:
            segments, info = model.transcribe(audio, **kwargs)
            segs = list(segments)
        # 확신도 높은 en/ko 감지가 나오면 이 세그먼트의 언어로 고정
        if (self._seg_lang is None and info.language in ("en", "ko")
                and (info.language_probability or 0) >= 0.6):
            self._seg_lang = info.language
        text = " ".join(s.text.strip() for s in segs).strip()
        if not text or looks_hallucinated(text):
            return None, info.language
        return text, info.language

    def _emit_partial(self):
        audio = np.concatenate(self._buf)
        if len(audio) / config.TARGET_SR < max(config.MIN_SPEECH_SEC, 0.5):
            return
        try:
            text, lang = self._transcribe(audio, final=False)
        except Exception as e:
            log.warning("부분 인식 오류: %s", e)
            return
        if not text:
            return
        # LocalAgreement: 직전 부분 결과와의 공통 접두어만 '안정'
        words = text.split()
        stable_n = _common_prefix_len(self._prev_partial_words, words)
        self._prev_partial_words = words
        self.on_partial(Utterance(
            id=f"{self.source}-s{self._seg_seq}", source=self.source, text=text,
            stable=" ".join(words[:stable_n]),
            lang=effective_lang(text, lang), time=time.time(), is_final=False))

    def _emit_segment_end(self):
        """확정 없이 세그먼트가 버려질 때(짧음/환각/오류) 라이브 자막을 지우도록 빈 부분자막 전송."""
        self.on_partial(Utterance(
            id=f"{self.source}-s{self._seg_seq}", source=self.source,
            text="", stable="", lang="en", time=time.time(), is_final=False))

    def _finalize(self):
        buf, self._buf = self._buf, []
        self._active = False
        self._pre_roll = []
        self._silence_sec = 0.0
        buffered_silence, self._buffered_silence_sec = self._buffered_silence_sec, 0.0
        speech_sec, self._speech_sec = self._speech_sec, 0.0
        self._prev_partial_words = []

        if speech_sec < config.MIN_SPEECH_SEC or not buf:
            self._emit_segment_end()
            return
        audio = np.concatenate(buf)
        # 끝의 침묵은 잘라냄 — 단, 실제로 버퍼에 들어간 침묵만.
        # (tick으로 흐른 벽시계 침묵은 버퍼에 없으므로 잘라내면 실제 음성이 깎인다)
        trim = int(max(0.0, buffered_silence - 0.2) * config.TARGET_SR)
        if trim > 0 and len(audio) > trim:
            audio = audio[:-trim]
        try:
            text, lang = self._transcribe(audio, final=True)
        except Exception as e:
            log.error("확정 인식 오류: %s", e)
            self._emit_segment_end()
            return
        if not text:
            self._emit_segment_end()
            return
        self._confirmed_tail = (self._confirmed_tail + " " + text)[-400:]
        self._counter += 1
        self.on_final(Utterance(
            id=f"{self.source}-{int(time.time()*1000)}-{self._counter}",
            source=self.source, text=text, stable=text,
            lang=effective_lang(text, lang), time=time.time(), is_final=True))

    def flush(self):
        """중지 시 남아있는 발화를 확정."""
        if self._active and self._buf:
            self._finalize()
