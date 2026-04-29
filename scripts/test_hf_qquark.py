import argparse
import torch
import sentencepiece as spm
from transformers import AutoModelForCausalLM


SYSTEM = (
    "Ты qquark 435M, prompt-enhancer для AI code agents. "
    "Не решай задачу напрямую. Перепиши запрос пользователя как agent-ready prompt. "
    "Сохраняй смысл, контекст и ограничения. Не добавляй лишние технологии."
)


def build_prompt(user_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


@torch.no_grad()
def generate(model, sp, prompt, device, max_new_tokens=180, temperature=0.2, top_k=20):
    ids = sp.encode(prompt)

    if len(ids) == 0:
        raise RuntimeError("Tokenizer returned 0 tokens. Check tokenizer.model")

    prompt_len = len(ids)
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        outputs = model(input_ids=x[:, -1024:])
        logits = outputs.logits[:, -1, :] / max(temperature, 1e-6)

        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            logits = torch.where(
                logits < min_value,
                torch.full_like(logits, -float("inf")),
                logits,
            )

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

        new_text = sp.decode(x[0, prompt_len:].tolist())
        if "<|im_end|>" in new_text:
            break

    answer = sp.decode(x[0, prompt_len:].tolist())
    answer = answer.split("<|im_end|>", 1)[0]
    return answer.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="release/qquark-435m-v0.1-hf")
    parser.add_argument("--tokenizer", default="data/tokenizer.model")
    parser.add_argument("--prompt", default="сделай кнопку красивее")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    model.eval()

    prompt = build_prompt(args.prompt)

    print("Prompt tokens:", len(sp.encode(prompt)))

    out = generate(
        model=model,
        sp=sp,
        prompt=prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    print("\n=== OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()
