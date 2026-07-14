"""E2E 스모크 테스트 — 마이크 없이 파이프라인 검증.

Windows SAPI TTS로 영어 음성 WAV를 합성한 뒤, 캡처 단계를 건너뛰고
SegmentAssembler(VAD→whisper) → Translator 로 직접 흘려보낸다.

사용:  python -m tests.test_pipeline [--engine google|local]
"""
import argparse
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import config  # noqa: E402
from server.transcriber import SegmentAssembler  # noqa: E402
from server.translator import Translator  # noqa: E402

WAV = Path(__file__).parent / "sample_en.wav"
SENTENCE = ("Good morning everyone. Let's review the quarterly results "
            "and discuss the roadmap for the next release.")


def synth_wav():
    """SAPI TTS로 16kHz 모노 WAV 생성."""
    ps = f'''
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile("{WAV}", $fmt)
$s.Speak("{SENTENCE}")
$s.Dispose()
'''
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
    assert WAV.exists(), "TTS WAV 생성 실패"


def load_wav() -> np.ndarray:
    with wave.open(str(WAV), "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.astype(np.float32) / 32768.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="google", choices=["google", "local"])
    args = ap.parse_args()

    if not WAV.exists():
        print("1) TTS로 테스트 음성 합성 중...")
        synth_wav()
    audio = load_wav()
    print(f"   음성 길이: {len(audio)/16000:.1f}초")

    results = {"partials": [], "finals": [], "translations": []}

    print("2) whisper 모델 로드 + 스트리밍 인식...")
    t0 = time.time()
    asm = SegmentAssembler(
        source="system",
        on_partial=lambda u: (results["partials"].append(u.text),
                              print(f"   [부분] {u.text}")),
        on_final=lambda u: (results["finals"].append(u),
                            print(f"   [확정] ({u.lang}) {u.text}")),
    )
    # 실제 캡처처럼 BLOCK_SEC 단위로 공급 + 끝에 침묵 붙여 확정 유도
    block = int(config.TARGET_SR * config.BLOCK_SEC)
    padded = np.concatenate([audio, np.zeros(int(16000 * 1.5), dtype=np.float32)])
    for i in range(0, len(padded), block):
        asm.feed(padded[i:i + block])
    asm.flush()
    print(f"   인식 완료 ({time.time()-t0:.1f}초)")

    assert results["finals"], "확정 인식 결과가 없습니다!"
    joined = " ".join(u.text for u in results["finals"]).lower()
    assert "quarterly" in joined or "roadmap" in joined, f"인식 내용이 이상함: {joined}"

    print(f"3) 번역 ({args.engine})...")
    done = []
    tr = Translator(on_result=lambda jid, text, fin: (done.append(text),
                                                      print(f"   [번역] {text}")))
    tr.set_engine(args.engine)
    tr.start()
    for u in results["finals"]:
        tr.submit_final(u.id, u.text, u.lang)
    deadline = time.time() + 300
    while len(done) < len(results["finals"]) and time.time() < deadline:
        time.sleep(0.3)
    tr.stop()

    assert len(done) == len(results["finals"]), "번역이 완료되지 않았습니다"
    assert any("가" <= ch <= "힣" for t in done for ch in t), "한국어 출력이 아닙니다"
    print("\n✅ E2E 파이프라인 검증 성공")


if __name__ == "__main__":
    main()
