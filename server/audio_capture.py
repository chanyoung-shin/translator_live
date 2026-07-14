"""오디오 캡처 — WASAPI 루프백(시스템 소리) / 마이크.

PyAudioWPatch 로 Windows WASAPI 루프백 장치를 열어, 회의 앱(Zoom/Teams/Meet)이
스피커로 내보내는 소리를 캡처한다. 임의 샘플레이트/채널을 soxr 스트리밍
리샘플러로 16kHz 모노 float32 로 변환해 큐에 넣는다.

핵심 제약 (PyAudioWPatch/PortAudio 검증된 동작):
  - 루프백에서 blocking read()는 무음 시 무한 대기 → 반드시 콜백 방식 사용
  - 무음 중에는 콜백 자체가 오지 않음 → 침묵 감지는 소비자 쪽에서 시간 기반으로
  - 스트림 열기/닫기는 PyAudio()를 만든 스레드에서만
  - PortAudio 장치 테이블은 Pa_Initialize 시점에 고정 → 살아있는 인스턴스로는
    기본 장치 변경(블루투스 전환 등)을 감지할 수 없음. 대신 '콜백이 한동안
    안 온다 = 무음이거나 장치가 죽었다'로 보고 주기적으로 전체 재오픈한다.
    (재오픈 시 PyAudio를 terminate→재생성해야 새 장치 테이블을 읽는다.
     주의: 'both' 모드에서는 다른 소스의 인스턴스가 살아있어 refcount가 0이
     안 되므로 테이블 갱신이 안 될 수 있음 — 그 경우 중지/시작으로 복구)
  - 장치 고유 샘플레이트로만 열 수 있음 → 16kHz 강제 금지, 이후 리샘플
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from . import config

log = logging.getLogger("livebridge.audio")

try:
    import pyaudiowpatch as pyaudio
except ImportError:  # pragma: no cover
    pyaudio = None

try:
    import soxr
except ImportError:  # pragma: no cover
    soxr = None

FRAMES_PER_BUFFER = 1024
STARVATION_REOPEN_SEC = 10.0  # 이 시간 동안 콜백이 없으면 스트림 재오픈 (장치 변경 대응)

# Pa_Initialize/Pa_Terminate는 스레드 안전하지 않다 — '둘 다' 모드에서 시스템/마이크
# 스레드가 동시에 PyAudio를 만들고 지우면 PortAudio 내부 assertion으로 프로세스가
# 통째로 죽는다 (pa_front.c:233, 실제 재현됨). 생성~스트림 오픈과 정리를 전역 직렬화.
_pa_lock = threading.Lock()


class AudioSource:
    """단일 오디오 소스(루프백 또는 마이크)를 전용 스레드로 캡처.

    16kHz 모노 float32 numpy 조각이 out_q 로 들어간다 (크기는 가변).
    """

    def __init__(self, kind: str, out_q: "queue.Queue[np.ndarray]",
                 on_error: Optional[Callable[[str], None]] = None):
        assert kind in ("system", "mic")
        self.kind = kind
        self.out_q = out_q
        self.on_error = on_error or (lambda msg: None)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.device_name = "?"
        self._resampler = None
        self._channels = 1
        self._last_cb = 0.0  # 마지막 콜백 시각 (monotonic)

    # ---------- 장치 선택 ----------
    def _pick_device(self, p) -> dict:
        if self.kind == "mic":
            try:
                return p.get_default_wasapi_device(d_in=True)
            except Exception:
                return p.get_default_input_device_info()
        try:
            return p.get_default_wasapi_loopback()
        except Exception:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_out = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
            for lb in p.get_loopback_device_info_generator():
                if default_out["name"] in lb["name"]:
                    return lb
            for lb in p.get_loopback_device_info_generator():
                return lb
            raise RuntimeError("루프백(시스템 오디오) 장치를 찾을 수 없습니다")

    # ---------- PortAudio 콜백 (PortAudio 자체 스레드에서 실행) ----------
    def _callback(self, in_data, frame_count, time_info, status):
        try:
            self._last_cb = time.monotonic()
            pcm = np.frombuffer(in_data, dtype=np.float32)
            if self._channels > 1:
                pcm = pcm.reshape(-1, self._channels).mean(axis=1)
            mono = np.ascontiguousarray(pcm, dtype=np.float32)
            if self._resampler is not None:
                mono = self._resampler.resample_chunk(mono, last=False)
            if mono.size:
                self.out_q.put(mono)
        except Exception:
            pass  # 콜백에서 예외가 새면 스트림이 죽으므로 삼킨다
        return (None, pyaudio.paContinue)

    # ---------- 캡처 스레드 ----------
    def _run(self):
        if pyaudio is None:
            self.on_error("PyAudioWPatch가 설치되지 않았습니다. setup.bat을 다시 실행해 주세요.")
            return
        # WASAPI는 스레드별 COM 초기화가 필요하다. PortAudio는 '최초 Pa_Initialize를
        # 수행한 스레드'만 COM을 세팅하므로, '둘 다' 모드의 두 번째 소스 스레드는
        # COM 없이 장치를 열다 -9999 Unanticipated host error가 난다 (실측 재현/검증됨).
        import ctypes
        co_ok = False
        try:
            hr = ctypes.windll.ole32.CoInitializeEx(None, 0x0)  # COINIT_MULTITHREADED
            co_ok = hr in (0, 1)  # S_OK / S_FALSE(이미 초기화됨)
        except Exception:
            pass
        fails = 0
        while not self._stop.is_set():
            p = None
            stream = None
            try:
                with _pa_lock:  # 초기화+장치조회+오픈을 다른 소스와 직렬화
                    p = pyaudio.PyAudio()
                    dev = self._pick_device(p)
                    self.device_name = dev["name"]
                    dev_index, dev_name = dev["index"], dev["name"]
                    src_sr = int(dev["defaultSampleRate"])
                    self._channels = max(1, int(dev["maxInputChannels"]))
                    self._resampler = None
                    if src_sr != config.TARGET_SR:
                        if soxr is None:
                            raise RuntimeError("soxr가 설치되지 않았습니다 (리샘플 불가)")
                        self._resampler = soxr.ResampleStream(src_sr, config.TARGET_SR, 1, dtype="float32")
                    log.info("[%s] 캡처 시작: %s (%dHz, %dch)", self.kind, dev_name, src_sr, self._channels)

                    stream = p.open(
                        format=pyaudio.paFloat32,
                        channels=self._channels,
                        rate=src_sr,
                        input=True,
                        input_device_index=dev_index,
                        frames_per_buffer=FRAMES_PER_BUFFER,
                        stream_callback=self._callback,
                    )
                    stream.start_stream()
                fails = 0
                self._last_cb = time.monotonic()

                # 감시: PortAudio 장치 테이블은 갱신되지 않으므로 장치 비교는 무의미.
                # 콜백이 STARVATION_REOPEN_SEC 동안 없으면(긴 무음 또는 기본 장치가
                # 다른 곳으로 넘어감) 전체 재오픈해 새 기본 장치를 따라간다.
                # 진짜 무음일 때의 재오픈은 어차피 소리가 없으므로 무해하다.
                while not self._stop.is_set():
                    time.sleep(0.25)
                    if not stream.is_active():
                        log.warning("[%s] 스트림 비활성 → 재연결", self.kind)
                        break
                    if time.monotonic() - self._last_cb >= STARVATION_REOPEN_SEC:
                        log.debug("[%s] %.0f초간 오디오 없음 → 장치 재확인 재오픈",
                                  self.kind, STARVATION_REOPEN_SEC)
                        break

            except Exception as e:
                if self._stop.is_set():
                    break
                fails += 1
                log.warning("[%s] 캡처 오류 (%s) — %d번째 재시도", self.kind, e, fails)
                if fails == 3:
                    self.on_error(f"오디오 장치 오류: {e}")
                if fails > 40:
                    self.on_error("오디오 장치를 계속 열 수 없어 캡처를 중단합니다.")
                    break
                time.sleep(min(0.5 * fails, 3.0))
            finally:
                with _pa_lock:  # 정리(Pa_Terminate)도 다른 소스의 초기화와 겹치면 안 됨
                    try:
                        if stream is not None:
                            stream.stop_stream()
                            stream.close()
                    except Exception:
                        pass
                    try:
                        if p is not None:
                            p.terminate()
                    except Exception:
                        pass
        if co_ok:
            try:
                ctypes.windll.ole32.CoUninitialize()
            except Exception:
                pass
        log.info("[%s] 캡처 종료", self.kind)

    # ---------- 제어 ----------
    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"audio-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=4)
            self._thread = None


def list_devices() -> str:
    """디버깅/설정 화면용: 사용 가능한 장치 요약 문자열."""
    if pyaudio is None:
        return "PyAudioWPatch 미설치"
    lines = []
    with _pa_lock:
        p = pyaudio.PyAudio()
        try:
            try:
                lb = p.get_default_wasapi_loopback()
                lines.append(f"시스템 오디오: {lb['name']}")
            except Exception as e:
                lines.append(f"루프백 조회 실패: {e}")
            try:
                mic = p.get_default_wasapi_device(d_in=True)
                lines.append(f"기본 마이크: {mic['name']}")
            except Exception:
                lines.append("기본 마이크: 없음")
        finally:
            p.terminate()
    return "\n".join(lines)
