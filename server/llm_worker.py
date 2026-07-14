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

# 오전사 보정 모드 프롬프트 (Qwen3-1.7B — /no_think로 추론 모드 끔)
CORRECT_PROMPT = (
    "You clean up real-time speech-recognition transcripts. "
    "You are given recent conversation lines and the last transcribed [Line]. "
    "Output ONLY the corrected version of the [Line], in the SAME language it is written in. "
    "Fix words that were likely mis-heard (replaced by similar-sounding words) using the context. "
    "If the line already looks correct, output it unchanged. "
    "Never translate. Never add explanations or quotes. /no_think"
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--n-ctx", type=int, default=1024)
    ap.add_argument("--mode", choices=["chat", "seedx", "correct"], default="chat")
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
                user = (f"[Conversation so far]\n{ctx}\n\n[Line]\n{text}" if ctx
                        else f"[Line]\n{text}")
                out = llm.create_chat_completion(
                    messages=[{"role": "system", "content": CORRECT_PROMPT},
                              {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=max(64, len(text) * 2))
                result = out["choices"][0]["message"]["content"]
                result = _THINK_RE.sub("", result).strip().strip('"')
                if not result:
                    result = text  # 보정 결과가 비면 원문 유지
            else:
                messages = [
                    {"role": "system",
                     "content": SYSTEM_PROMPT.format(target=LANG_NAME.get(dst, "Korean"))},
                    {"role": "user", "content": build_user_content(text, context)},
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
