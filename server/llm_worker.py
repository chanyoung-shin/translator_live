"""llama.cpp 번역 워커 — 별도 프로세스로 실행.

llama.cpp는 VRAM 부족 등 CUDA 오류 시 파이썬 예외가 아니라 네이티브 abort로
프로세스를 통째로 죽일 수 있다. 서버 본체가 같이 죽지 않도록 이 워커를
서브프로세스로 띄우고, stdin/stdout JSON 라인으로 통신한다.

프로토콜 (stdout, 응답 라인은 "@@" 접두어 — llama.cpp 로그와 구분):
  기동 완료:  @@READY
  요청(stdin): {"text": "...", "src": "en", "dst": "ko"}
  응답:        @@{"ok": true, "text": "..."}  또는  @@{"ok": false, "error": "..."}
stdin EOF 시 종료 (부모 프로세스가 죽으면 자동 종료).
"""
import argparse
import json
import re
import sys

from server.translator import LANG_NAME, SYSTEM_PROMPT, build_user_content

# 오전사 보정 모드 프롬프트 + few-shot (작은 모델일수록 예시가 결정적)
CORRECT_PROMPT = (
    "You fix speech-to-text transcription errors from a live meeting. "
    "Given recent conversation lines and the last transcribed [Line], "
    "output ONLY the corrected version of the [Line], in the SAME language it is written in. "
    "Words may have been replaced by similar-SOUNDING wrong words - find and fix them using the context. "
    "If the line already looks correct, output it unchanged. "
    "Never translate. Never add explanations or quotes."
)
CORRECT_FEWSHOT = [
    ("[Context]\nWe should merge the branch today.\nThe build passed.\n\n"
     "[Line]\nLet's dip Roy the new version tonight.",
     "Let's deploy the new version tonight."),
    ("[Context]\n요즘 GPU 가격이 너무 올랐어\n\n[Line]\n그래서 새 그래픽 가드를 못 사겠어",
     "그래서 새 그래픽 카드를 못 사겠어"),
    ("[Context]\n내일 회의 몇 시야?\n\n[Line]\n오후 세 시에 시작해",
     "오후 세 시에 시작해"),
]

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--mode", choices=["chat", "seedx", "correct"], default="chat")
    ap.add_argument("--nothink", action="store_true",
                    help="하이브리드 추론 모델(Qwen3 비-2507)의 생각 모드를 빈 <think> 프리필로 끔")
    args = ap.parse_args()

    from llama_cpp import Llama
    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        verbose=False,
    )
    print("@@READY", flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            text, src, dst = req["text"], req["src"], req["dst"]
            context = req.get("context") or []
            if args.mode == "seedx":
                # Seed-X는 고정 NMT 프롬프트 형식 — 맥락 미지원
                prompt = (f"Translate the following {LANG_NAME.get(src, 'English')} sentence into "
                          f"{LANG_NAME.get(dst, 'Korean')}:\n{text} <{dst}>")
                out = llm(prompt, max_tokens=256, temperature=0.0,
                          stop=["\n", "Translate the following"])
                result = out["choices"][0]["text"].strip()
            elif args.mode == "correct":
                ctx = "\n".join(c for c in context[-6:] if c)
                messages = [{"role": "system", "content": CORRECT_PROMPT}]
                for u, a in CORRECT_FEWSHOT:
                    messages.append({"role": "user", "content": u})
                    messages.append({"role": "assistant", "content": a})
                messages.append({"role": "user",
                                 "content": (f"[Context]\n{ctx}\n\n[Line]\n{text}" if ctx
                                             else f"[Line]\n{text}")})
                out = llm.create_chat_completion(
                    messages=messages, temperature=0.0,
                    max_tokens=max(64, len(text) * 2))
                result = out["choices"][0]["message"]["content"]
                result = _THINK_RE.sub("", result).strip().strip('"')
                if not result:
                    result = text  # 보정 결과가 비면 원문 유지
            elif args.nothink:
                # ChatML 수동 구성 + 빈 <think> 프리필 — 하이브리드 모델이 생각을
                # 건너뛰고 즉답하게 한다 (create_chat_completion으로는 제어 불가)
                target = LANG_NAME.get(dst, "Korean")
                system = SYSTEM_PROMPT.format(target=target)
                user = build_user_content(text, context, target)
                prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
                          f"<|im_start|>user\n{user}<|im_end|>\n"
                          f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
                out = llm(prompt, max_tokens=min(512, max(48, len(text) * 2)),
                          temperature=0.0, stop=["<|im_end|>"])
                result = out["choices"][0]["text"].strip().strip('"')
            else:
                target = LANG_NAME.get(dst, "Korean")
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT.format(target=target)},
                    {"role": "user", "content": build_user_content(text, context, target)},
                ]
                out = llm.create_chat_completion(
                    messages=messages, temperature=0.0,
                    max_tokens=min(512, max(48, len(text) * 2)))
                result = out["choices"][0]["message"]["content"].strip().strip('"')
            print("@@" + json.dumps({"ok": True, "text": result}, ensure_ascii=False), flush=True)
        except Exception as e:  # 요청 단위 오류는 프로세스를 죽이지 않는다
            print("@@" + json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
