import argparse
import json
import sys
import requests

from qquark.context_builder import detect_project_context
from qquark.prompt import build_prompt, clean_output

def call_llama_completion(
    server: str,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    repeat_penalty: float,
    max_tokens: int,
) -> str:
    url = server.rstrip("/") + "/completion"

    payload = {
        "prompt": prompt,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "min_p": 0,
        "repeat_penalty": repeat_penalty,
        "n_predict": max_tokens,
        "stop": ["<|im_end|>"],
    }

    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()

    if "content" in data:
        return data["content"]

    if "choices" in data and data["choices"]:
        return data["choices"][0].get("text", "")

    return json.dumps(data, ensure_ascii=False, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="qquark 435M local prompt enhancer")
    parser.add_argument("request", nargs="+", help="Raw user request to improve")
    parser.add_argument("--server", default="http://127.0.0.1:8088", help="llama.cpp server URL")
    parser.add_argument("--context", default=".", help="Project root for context detection")
    parser.add_argument("--no-context", action="store_true", help="Disable automatic project context")
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--repeat-penalty", type=float, default=1.18)
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    user_request = " ".join(args.request)

    context = None
    if not args.no_context:
        context = detect_project_context(args.context)

    prompt = build_prompt(user_request, context=context)

    try:
        raw = call_llama_completion(
            server=args.server,
            prompt=prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repeat_penalty=args.repeat_penalty,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        print(f"qquark error: {e}", file=sys.stderr)
        sys.exit(1)

    print(clean_output(raw))

if __name__ == "__main__":
    main()
