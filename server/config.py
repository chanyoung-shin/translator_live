"""LiveBridge 설정 — 모델/파이프라인 기본값.

값들은 2026-07 리서치(오픈소스 스트리밍 자막 프로젝트들의 검증된 기본값)에서 가져옴.
대부분 환경변수로 덮어쓸 수 있다.
"""
import os

# ---------- 서버 ----------
HOST = os.environ.get("LB_HOST", "127.0.0.1")
PORT = int(os.environ.get("LB_PORT", "8765"))

# ---------- 오디오 ----------
TARGET_SR = 16000          # ASR 입력 샘플레이트
BLOCK_SEC = 0.2            # 파이프라인 처리 블록 길이 (초) — VAD 판정 단위
MAX_SEGMENT_SEC = 22.0     # whisper 30초 창 아래에서 강제 문장 확정
PRE_ROLL_SEC = 0.4         # 발화 시작 직전 오디오 포함 (첫 음절 잘림 방지)

# ---------- VAD / 세그먼트 (RealtimeSTT/whisper_streaming 검증 기본값 기반) ----------
VAD_THRESHOLD = 0.4            # silero 발화 확률 임계값 (0.5는 멀리서 나는 소리를 놓침)
SILENCE_FINALIZE_MS = 600      # 이 시간 이상 조용하면 문장 확정 (UI '문장 나누기'로 조절)
PARTIAL_INTERVAL_SEC = 0.7     # 부분(파티셜) 자막 갱신 주기
MIN_SPEECH_SEC = 0.25          # 이보다 짧은 발화는 무시

# ---------- ASR (faster-whisper) ----------
# 기본은 가벼운 small (VRAM ~0.5GB, 반응 빠름). 인식 정확도를 올리려면:
#   set LB_ASR_MODEL=large-v3-turbo   (VRAM ~1.6GB, 한국어/전문용어에 훨씬 강함)
ASR_MODEL = os.environ.get("LB_ASR_MODEL", "small")
ASR_MODEL_IS_DEFAULT = "LB_ASR_MODEL" not in os.environ  # CPU 폴백 시 자동 축소 허용 여부
ASR_DEVICE = os.environ.get("LB_ASR_DEVICE", "cuda")
ASR_COMPUTE = os.environ.get("LB_ASR_COMPUTE", "int8_float16")  # ~1.6GB VRAM
ASR_BEAM_PARTIAL = 1           # 부분: greedy (지연 최소)
ASR_BEAM_FINAL = 5             # 확정: 품질 우선
NO_SPEECH_THRESHOLD = 0.6
LOG_PROB_THRESHOLD = -1.0
COMPRESSION_RATIO_THRESHOLD = 2.4

# ---------- 번역 ----------
# "local"  : 로컬 모델 (폴백 체인: llama.cpp GGUF → transformers 4bit → NLLB → Google)
# "google" : deep-translator 의 Google 웹 번역 (온라인, 즉시 사용 가능)
TRANSLATE_ENGINE_DEFAULT = os.environ.get("LB_ENGINE", "local")

# 로컬 GGUF 프리셋 (llama-cpp-python)
#   qwen3-4b : Apache-2.0, Q4_K_M ≈2.5GB VRAM — 기본값 (Zoom과 GPU 공유해도 여유)
#   seedx-7b : ByteDance Seed-X-PPO-7B, 번역 특화 최고 품질, Q4_K_M ≈4.6GB — VRAM 빠듯
GGUF_PRESET = os.environ.get("LB_GGUF", "qwen3-4b")
GGUF_PRESETS = {
    "qwen3-1.7b": {
        "repo": "unsloth/Qwen3-1.7B-GGUF",
        "file": "Qwen3-1.7B-Q4_K_M.gguf",
        "mode": "chat",
        "nothink": True,      # 하이브리드 추론 모델 — 빈 <think> 프리필로 즉답
        "layers": 28,
        "vram_full_gb": 2.2,
        "n_ctx": 4096,
        "label": "Qwen3-1.7B",
    },
    "qwen3-4b": {
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
        "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        "mode": "chat",       # 채팅형 → 대화 맥락 기반 오전사 보정 지원
        "layers": 36,
        "vram_full_gb": 3.7,  # 전체 GPU 오프로드에 필요한 여유 VRAM (n_ctx 4096 포함)
        "n_ctx": 4096,        # 요약 기능을 위해 넉넉히
        "label": "Qwen3-4B",
    },
    "qwen3-8b": {
        "repo": "unsloth/Qwen3-8B-GGUF",
        "file": "Qwen3-8B-Q4_K_M.gguf",
        "mode": "chat",
        "nothink": True,      # 하이브리드 추론 모델 — 빈 <think> 프리필로 즉답 유도
        "layers": 36,
        "vram_full_gb": 6.3,
        "n_ctx": 4096,
        "label": "Qwen3-8B",
    },
    # 실험용 — UI에는 노출 안 함: Seed-X GGUF는 언어 태그(<ko>)가 특수 토큰으로
    # 변환되지 않아 en→ko 출력이 깨지는 것을 실측 확인 (조기 EOS/환각 반복).
    # vLLM 등 원본 가중치 배포에서는 우수하나 llama.cpp 경로에서는 비권장.
    "seedx-7b": {
        "repo": "mradermacher/Seed-X-PPO-7B-GGUF",
        "file": "Seed-X-PPO-7B.Q4_K_M.gguf",
        "mode": "seedx",      # 번역 특화 고정 프롬프트 → 별도 보정 워커 동반
        "layers": 32,
        "vram_full_gb": 5.6,
        "label": "Seed-X-7B",
    },
}
GGUF_GPU_LAYERS = os.environ.get("LB_GGUF_GPU_LAYERS", "auto")  # auto = 남은 VRAM 보고 결정

# 오전사 보정용 모델 — 번역 특화 모델(Seed-X)은 채팅형이 아니라 맥락 보정을
# 못 하므로, seedx 선택 시 이 모델이 보정 단계로 함께 뜬다 (기본 Qwen 경로는 불필요).
# 1.7B는 실험 결과 보정 능력이 부족해(원문 복사만 함) 4B를 사용 — VRAM이 부족하면
# auto-layers가 일부/전부 CPU로 내리며, 보정은 확정 문장에만 돌아서 감당 가능.
CORRECTOR_GGUF = {
    "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF",
    "file": "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "mode": "correct",
    "layers": 36,
    "vram_full_gb": 3.4,
    "label": "보정AI(Qwen3-4B)",
}

# transformers 폴백용 LLM
LLM_MODEL = os.environ.get("LB_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
LLM_4BIT = os.environ.get("LB_LLM_4BIT", "1") == "1"
# NLLB 폴백 (CC-BY-4.0 파인튜닝 — en→ko 전용)
NLLB_MODEL = os.environ.get("LB_NLLB_MODEL", "facebook/nllb-200-distilled-600M")

# 부분 자막 실시간 번역 (안정화된 텍스트만, 스로틀)
PARTIAL_TRANSLATE_INTERVAL = 1.2   # 초
PARTIAL_TRANSLATE_MIN_NEW_WORDS = 3  # 새 안정 단어가 이만큼 쌓여야 번역

# 맥락 기반 번역/오전사 보정: 최근 확정 문장 N개를 번역 LLM에 함께 전달
# → 음성인식이 비슷한 발음으로 잘못 받아적어도 맥락으로 의도를 살려 번역
# (많을수록 보정은 좋아지지만 문장마다 프롬프트 처리 비용 증가 — 4가 균형점)
CONTEXT_LINES = 4

# 회의 도메인 용어 (음성인식 정확도 힌트) — 예: set LB_VOCAB=LiveBridge, Kubernetes, 쿼터
VOCAB = os.environ.get("LB_VOCAB", "")

# 커스텀 번역 모델: 이 폴더에 .gguf 파일을 넣으면 UI 모델 선택칸에 자동으로 나타남
import pathlib
MODELS_DIR = pathlib.Path(__file__).resolve().parent.parent / "models"

# 요약: 전사가 이 길이(자)를 넘으면 나눠 요약 후 합침 (n_ctx 4096 안에 들어가게)
SUMMARY_CHUNK_CHARS = 6000

# 번역 방향: whisper가 감지한 언어 → 목표 언어
TARGET_LANG_FOR = {
    "en": "ko",
    "ko": "en",
}
DEFAULT_TARGET = "ko"

# ---------- 환각(hallucination) 필터 ----------
HALLUCINATION_PHRASES = {
    "thank you.", "thank you", "thanks for watching.", "thanks for watching",
    "thank you for watching.", "thank you for watching", "you", "bye.", "bye-bye.",
    "please subscribe.", "subtitles by", ".", "*", "음악", "[음악]", "(음악)",
    "시청해주셔서 감사합니다.", "시청해 주셔서 감사합니다.", "구독과 좋아요 부탁드립니다.",
    "감사합니다.", "감사합니다", "mbc 뉴스 이덕영입니다.",
}
