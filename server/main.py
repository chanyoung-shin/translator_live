"""LiveBridge 서버 — FastAPI + WebSocket 오케스트레이션.

오디오 캡처(콜백→큐) → 펌프 스레드(VAD/조립/tick) → whisper → 번역 → WS 브로드캐스트
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .audio_capture import AudioSource, list_devices
from .transcriber import SegmentAssembler, Utterance, get_model, wait_asr_idle
from .translator import Translator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("livebridge")

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
RECORD_DIR = ROOT / "recordings"

app = FastAPI(title="LiveBridge")
app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

PUMP_TIMEOUT = 0.2  # 큐 대기 시간 = 무음 시 tick 간격


class _WavRecorder:
    """캡처 오디오(16kHz 모노)를 WAV로 저장. 무음 구간은 0으로 채워 시간 정렬 유지."""

    def __init__(self, kind: str):
        RECORD_DIR.mkdir(exist_ok=True)
        tag = "마이크" if kind == "mic" else "시스템"
        self.path = RECORD_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.wav"
        self.wf = wave.open(str(self.path), "wb")
        self.wf.setnchannels(1)
        self.wf.setsampwidth(2)
        self.wf.setframerate(config.TARGET_SR)

    def write(self, chunk: np.ndarray):
        self.wf.writeframes((np.clip(chunk, -1.0, 1.0) * 32767).astype("<i2").tobytes())

    def write_silence(self, sec: float):
        self.wf.writeframes(b"\x00\x00" * int(config.TARGET_SR * sec))

    def close(self) -> Path:
        try:
            self.wf.close()
        except Exception:
            pass
        return self.path


class Pipeline:
    """소스 1개(system 또는 mic)에 대한 캡처→조립 파이프라인."""

    def __init__(self, kind: str, session: "Session"):
        self.kind = kind
        self.session = session
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.asm = SegmentAssembler(source=kind,
                                    on_partial=session.on_partial,
                                    on_final=session.on_final)
        self.src = AudioSource(kind, self.q,
                               on_error=lambda m: session.broadcast(
                                   {"type": "error_toast", "detail": m}))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._acc: list[np.ndarray] = []
        self._acc_len = 0

    def _pump(self):
        """큐에서 오디오를 모아 BLOCK_SEC 단위로 조립기에 공급.
        루프백은 무음 중 콜백이 안 오므로, 큐 타임아웃을 침묵 시간으로 흘린다."""
        block = int(config.TARGET_SR * config.BLOCK_SEC)
        rec: Optional[_WavRecorder] = None
        try:
            while not self._stop.is_set():
                try:
                    chunk = self.q.get(timeout=PUMP_TIMEOUT)
                except queue.Empty:
                    # 콜백이 끊겼다 = 무음 시작. 블록 미만으로 남은 오디오(마지막 음절)를
                    # 먼저 흘려보내야 tick-확정 때 잘리지 않고, 다음 발화에 섞이지도 않는다.
                    if self._acc_len > 0:
                        rest = np.concatenate(self._acc)
                        self._acc, self._acc_len = [], 0
                        if len(rest) < 512:  # silero VAD 최소 창 크기 확보
                            rest = np.pad(rest, (0, 512 - len(rest)))
                        self.asm.feed(rest)
                    self.asm.tick(PUMP_TIMEOUT)
                    if rec is not None:
                        rec.write_silence(PUMP_TIMEOUT)  # 무음도 채워 시간 정렬 유지
                    continue
                # ---- 녹음 (원본 16k 모노 그대로) ----
                if self.session.recording:
                    if rec is None:
                        rec = _WavRecorder(self.kind)
                        self.session.notice(f"🔴 녹음 시작 ({self.kind})")
                    rec.write(chunk)
                elif rec is not None:
                    self.session.notice(f"💾 녹음 저장: {rec.close().name}")
                    rec = None
                self._acc.append(chunk)
                self._acc_len += len(chunk)
                while self._acc_len >= block:
                    buf = np.concatenate(self._acc)
                    self._acc = [buf[block:]] if len(buf) > block else []
                    self._acc_len = len(buf) - block
                    self.asm.feed(buf[:block])
        finally:
            if rec is not None:
                self.session.notice(f"💾 녹음 저장: {rec.close().name}")

    def start(self):
        self.src.start()
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, name=f"pump-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self):
        self.src.stop()
        self._stop.set()
        joined = True
        if self._thread:
            self._thread.join(timeout=5)
            joined = not self._thread.is_alive()
        if joined:  # 펌프가 아직 돌고 있으면 flush가 feed와 경합하므로 건너뜀
            try:
                self.asm.flush()
            except Exception:
                pass


class Session:
    def __init__(self):
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.clients: set[WebSocket] = set()
        self.state = "idle"          # idle | loading | listening | error
        self.pipelines: list[Pipeline] = []
        self.translator: Optional[Translator] = None
        self.transcript: list[dict] = []
        self.summary: Optional[str] = None
        self._summarizing = False
        self._summary_tr: Optional[Translator] = None  # Google 엔진일 때 요약용 로컬 백엔드
        self.live_translation = True
        self.recording = False
        self._last_mt_words: dict[str, int] = {}   # 소스별 마지막 부분번역 안정단어 수
        self._last_seg_id: dict[str, str] = {}     # 소스별 현재 세그먼트 id
        self._start_lock = threading.Lock()

    # ---------- 브로드캐스트 (임의 스레드에서 호출 가능) ----------
    def broadcast(self, msg: dict):
        if self.loop is None:
            return
        data = json.dumps(msg, ensure_ascii=False)

        def _send():
            for ws in list(self.clients):
                task = asyncio.ensure_future(ws.send_text(data))
                task.add_done_callback(lambda t: t.exception())  # 예외 소비
        self.loop.call_soon_threadsafe(_send)

    def set_state(self, state: str, detail: str = "", sysinfo: str = ""):
        self.state = state
        msg = {"type": "status", "state": state, "detail": detail}
        if sysinfo:
            msg["sysinfo"] = sysinfo
        self.broadcast(msg)

    def notice(self, detail: str):
        """상태 필에 잠깐 표시되는 안내 (녹음 시작/저장 등)."""
        self.broadcast({"type": "notice", "detail": detail})

    # ---------- 파이프라인 콜백 ----------
    def on_partial(self, u: Utterance):
        tentative = u.text[len(u.stable):].strip() if u.text.startswith(u.stable) else u.text
        self.broadcast({"type": "partial", "id": u.id, "stable": u.stable,
                        "tentative": tentative, "source": u.source})
        # 새 세그먼트가 시작되면 부분번역 카운터 리셋
        # (확정이 환각 필터 등으로 억제돼 on_final이 안 와도 다음 발화가 굶지 않도록)
        if self._last_seg_id.get(u.source) != u.id:
            self._last_seg_id[u.source] = u.id
            self._last_mt_words[u.source] = 0
        # 안정된(LocalAgreement) 텍스트만, 새 단어가 충분히 쌓였을 때만 부분 번역
        if self.live_translation and self.translator and u.stable:
            n = len(u.stable.split())
            if n - self._last_mt_words.get(u.source, 0) >= config.PARTIAL_TRANSLATE_MIN_NEW_WORDS:
                self._last_mt_words[u.source] = n
                self.translator.submit_partial(u.id, u.stable, u.lang,
                                               context=self._recent_context())

    def _recent_context(self, exclude_last: bool = False) -> list:
        """번역 LLM에 줄 최근 대화 맥락 — 오전사 보정과 대명사 해석에 사용."""
        entries = self.transcript[:-1] if exclude_last else self.transcript
        return [e["src_text"] for e in entries[-config.CONTEXT_LINES:]]

    _TERMINAL_PUNCT = ".?!…。」”\"'"

    def _is_continuation(self, prev: dict, u: Utterance) -> bool:
        """새 확정이 직전 문장의 이어짐인지 — 긴 발화가 강제 컷으로 쪼개진 경우
        따로 번역하면 어색하므로 합쳐서 재번역한다."""
        if prev["source"] != u.source or prev["lang"] != u.lang:
            return False
        if len(prev["src_text"]) > 600:   # 무한 병합 방지
            return False
        gap = (u.time - u.dur) - prev["time"]  # 앞 문장 끝 ~ 새 문장 시작 사이 침묵
        if gap > 2.0:
            return False
        tail = prev["src_text"].rstrip()
        return bool(tail) and tail[-1] not in self._TERMINAL_PUNCT  # 문장이 안 끝났음

    def on_final(self, u: Utterance):
        self._last_mt_words[u.source] = 0
        prev = self.transcript[-1] if self.transcript else None
        if prev is not None and self._is_continuation(prev, u):
            # 직전 카드에 이어붙이고 합쳐진 전체 문장을 재번역
            prev["src_text"] = (prev["src_text"] + " " + u.text).strip()
            prev["dst_text"] = None
            prev["time"] = u.time
            self.broadcast({"type": "final_update", "id": prev["id"],
                            "text": prev["src_text"]})
            if self.translator:
                context = [e["src_text"] for e in self.transcript[:-1][-config.CONTEXT_LINES:]]
                self.translator.submit_final(prev["id"], prev["src_text"], prev["lang"],
                                             context=context)
            return
        context = self._recent_context()  # 현재 문장 추가 전 = 이전 문장들
        entry = {"id": u.id, "source": u.source, "time": u.time,
                 "src_text": u.text, "dst_text": None, "lang": u.lang}
        self.transcript.append(entry)
        self.broadcast({"type": "final", "id": u.id, "text": u.text,
                        "source": u.source, "time": u.time, "lang": u.lang})
        if self.translator:
            self.translator.submit_final(u.id, u.text, u.lang, context=context)

    def on_translation(self, job_id: str, text: str, is_final: bool):
        if is_final:
            for entry in reversed(self.transcript):
                if entry["id"] == job_id:
                    entry["dst_text"] = text
                    break
            self.broadcast({"type": "translation", "id": job_id, "text": text})
        else:
            # id를 함께 보내 클라이언트가 '지금 표시 중인 세그먼트'의 번역인지 확인
            self.broadcast({"type": "partial_translation", "id": job_id, "text": text})

    # ---------- 시작/중지 ----------
    def start_pipeline(self, source_kind: str, engine: str, model: str = None):
        with self._start_lock:
            self._stop_pipeline_locked()
            self.set_state("loading", "음성 인식 모델 로드 중… (최초 실행은 다운로드로 몇 분 걸려요)")
            try:
                get_model()  # whisper 지연 로드 (블로킹)
            except Exception as e:
                log.exception("ASR 모델 로드 실패")
                self.set_state("error", f"음성 인식 모델 로드 실패: {e}")
                return

            self.translator = Translator(
                on_result=self.on_translation,
                on_status=lambda s: self.broadcast({"type": "status", "state": self.state,
                                                    "detail": s}),
            )
            self.translator.set_engine(engine)
            if model:
                self.translator.set_model(model)
            self.translator.start()
            # 번역 모델은 백그라운드 프리로드 (자막은 그동안에도 나오도록)
            threading.Thread(target=self.translator.preload, daemon=True).start()

            kinds = ["system", "mic"] if source_kind == "both" else [source_kind]
            for kind in kinds:
                pl = Pipeline(kind, self)
                pl.start()
                self.pipelines.append(pl)

            time.sleep(0.6)  # 장치 오픈 대기
            self.set_state("listening", "듣는 중", sysinfo=self._sysinfo())
            log.info("파이프라인 시작: %s", source_kind)

    def _stop_pipeline_locked(self):
        for pl in self.pipelines:
            pl.stop()
        self.pipelines = []
        wait_asr_idle(10)  # 마지막 확정 인식이 끝나야 번역 큐에 들어간다
        if self.translator:
            self.translator.stop()
        self._last_mt_words = {}
        self._last_seg_id = {}

    def stop_pipeline(self):
        with self._start_lock:
            self._stop_pipeline_locked()
            self.set_state("idle", "대기 중")

    # ---------- 회의 요약 ----------
    def summarize(self):
        if self._summarizing:
            return
        if not self.transcript:
            self.notice("요약할 내용이 아직 없어요")
            return
        self._summarizing = True
        try:
            self.notice("📝 회의 요약 생성 중… (수십 초 걸릴 수 있어요)")
            # 요약은 로컬 LLM 필요 — 번역 엔진이 Google이거나 시작 전이면 전용 로컬 백엔드 사용
            tr = self.translator
            if tr is None or tr.engine != "local":
                if self._summary_tr is None:
                    self._summary_tr = Translator(on_result=lambda *a: None,
                                                  on_status=lambda s: None)
                    self._summary_tr.set_engine("local")
                tr = self._summary_tr
            lines = []
            for e in self.transcript:
                who = "나" if e["source"] == "mic" else "상대방"
                lines.append(f"[{who}] {e['src_text']}")
            text = "\n".join(lines)
            # 긴 회의: 나눠 요약(map) 후 병합(reduce)
            if len(text) > config.SUMMARY_CHUNK_CHARS:
                parts = []
                for i in range(0, len(text), config.SUMMARY_CHUNK_CHARS):
                    chunk = text[i:i + config.SUMMARY_CHUNK_CHARS]
                    self.notice(f"📝 요약 중… ({i // config.SUMMARY_CHUNK_CHARS + 1}"
                                f"/{-(-len(text) // config.SUMMARY_CHUNK_CHARS)} 구간)")
                    parts.append(tr.summarize(chunk))
                result = tr.summarize("\n\n---\n\n".join(parts), combine=True)
            else:
                result = tr.summarize(text)
            self.summary = result
            self.broadcast({"type": "summary", "text": result})
        except Exception as e:
            log.error("요약 실패: %s", e)
            self.broadcast({"type": "error_toast", "detail": f"요약 실패: {e}"})
        finally:
            self._summarizing = False

    def _sysinfo(self) -> str:
        lines = [f"음성 인식: faster-whisper {config.ASR_MODEL} "
                 f"({config.ASR_DEVICE}/{config.ASR_COMPUTE})"]
        if self.translator:
            lines.append(f"번역: {self.translator.backend_name}")
        try:
            lines.append(list_devices())
        except Exception:
            pass
        return "\n".join(lines)


session = Session()


# ============================================================
# 라우트
# ============================================================
@app.get("/")
async def index():
    return FileResponse(str(WEB / "index.html"))


@app.get("/export")
async def export():
    if not session.transcript:
        return PlainTextResponse("내용 없음", status_code=404)
    lines = ["# 회의록 (LiveBridge)", "",
             f"- 내보낸 시각: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if session.summary:
        lines += ["## 요약", "", session.summary, "", "## 전체 기록", ""]
    for e in session.transcript:
        t = time.strftime("%H:%M:%S", time.localtime(e["time"]))
        who = "나" if e["source"] == "mic" else "상대방"
        lines.append(f"**[{t}] {who}**")
        if e.get("dst_text"):
            lines.append(f"> {e['dst_text']}")
        lines.append(f"> _{e['src_text']}_")
        lines.append("")
    return PlainTextResponse("\n".join(lines), media_type="text/markdown; charset=utf-8",
                             headers={"Content-Disposition": "attachment; filename=meeting.md"})


def _available_models() -> list:
    """UI 모델 선택칸 목록: 기본 프리셋 + models/ 폴더의 커스텀 GGUF."""
    options = [
        {"value": "qwen3-4b", "label": "Qwen3-4B — 기본 · 권장 (VRAM 2.7GB)"},
        {"value": "qwen3-8b", "label": "Qwen3-8B — 품질 우선 (VRAM ~5.3GB)"},
        {"value": "qwen3-1.7b", "label": "Qwen3-1.7B — 초경량 · CPU/저사양용 (품질 낮음)"},
    ]
    try:
        if config.MODELS_DIR.is_dir():
            for p in sorted(config.MODELS_DIR.glob("*.gguf")):
                size = p.stat().st_size / 2**30
                options.append({"value": f"file:{p.name}",
                                "label": f"📁 {p.name} ({size:.1f}GB)"})
    except Exception:
        pass
    return options


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session.loop = asyncio.get_running_loop()
    session.clients.add(ws)
    await ws.send_text(json.dumps(
        {"type": "status", "state": session.state,
         "detail": "듣는 중" if session.state == "listening" else "대기 중"},
        ensure_ascii=False))
    await ws.send_text(json.dumps({"type": "models", "options": _available_models()},
                                  ensure_ascii=False))
    if session.summary:
        await ws.send_text(json.dumps({"type": "summary", "text": session.summary},
                                      ensure_ascii=False))
    # 새로고침 대응: 기존 기록 재전송
    for e in session.transcript[-200:]:
        await ws.send_text(json.dumps({"type": "final", "id": e["id"], "text": e["src_text"],
                                       "source": e["source"], "time": e["time"], "lang": e["lang"]},
                                      ensure_ascii=False))
        if e.get("dst_text"):
            await ws.send_text(json.dumps({"type": "translation", "id": e["id"],
                                           "text": e["dst_text"]}, ensure_ascii=False))
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            t = msg.get("type")
            if t == "start":
                source = msg.get("source", "system")
                engine = msg.get("engine", config.TRANSLATE_ENGINE_DEFAULT)
                model = msg.get("model")
                if "recording" in msg:
                    session.recording = bool(msg["recording"])
                threading.Thread(target=session.start_pipeline,
                                 args=(source, engine, model), daemon=True).start()
            elif t == "stop":
                threading.Thread(target=session.stop_pipeline, daemon=True).start()
            elif t == "options":
                if "live_translation" in msg:
                    session.live_translation = bool(msg["live_translation"])
                if "recording" in msg:
                    session.recording = bool(msg["recording"])
                if "engine" in msg and session.translator:
                    session.translator.set_engine(msg["engine"])
                if "model" in msg and session.translator:
                    session.translator.set_model(msg["model"])
            elif t == "summarize":
                threading.Thread(target=session.summarize, daemon=True).start()
            elif t == "clear":
                session.transcript = []
                session.summary = None
    except WebSocketDisconnect:
        pass
    finally:
        session.clients.discard(ws)


def main():
    import uvicorn
    log.info("LiveBridge 시작: http://%s:%d", config.HOST, config.PORT)
    uvicorn.run(app, host=config.HOST, port=config.PORT, log_level="warning")


if __name__ == "__main__":
    main()
