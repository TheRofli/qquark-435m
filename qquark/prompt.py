SYSTEM_PROMPT = (
    "Ты qquark 435M, prompt-enhancer для AI code agents. "
    "Не решай задачу напрямую. Перепиши запрос пользователя как agent-ready prompt. "
    "Сохраняй смысл, контекст и ограничения. Не добавляй лишние технологии."
)

def build_prompt(user_request: str, context: str | None = None) -> str:
    context_text = ""
    if context:
        context_text = (
            "Автоматически собранный контекст проекта:\n"
            f"{context}\n\n"
            "Важно: сохраняй стек проекта, не добавляй технологии, которых нет в контексте.\n\n"
        )

    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{context_text}"
        f"{user_request}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def clean_output(text: str) -> str:
    text = text.split("<|im_end|>", 1)[0]
    text = text.split("<|im_start|>", 1)[0]

    bad_markers = [
        "source=",
        "url-status=",
        "<|page=",
    ]

    for marker in bad_markers:
        if marker in text:
            text = text.split(marker, 1)[0]

    return text.strip()
