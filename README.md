# qquark 435M

**qquark 435M** is a small local prompt-enhancer model for AI code agents.

It rewrites rough user requests into clearer agent-ready prompts before sending them to larger coding agents such as Codex, Roo, Cline, OpenCode, Claude Code, local Qwen/OmniCoder pipelines, or other agentic coding tools.

> Status: early alpha / research prototype.

---

## Why qquark exists

Large coding agents often perform better when the user request is clear, scoped, and explicit.

qquark 435M acts as a lightweight local preprocessing model:

```text
rough user request
        ↓
qquark 435M
        ↓
agent-ready prompt
        ↓
larger code agent
```

The goal is not to replace a coding model.  
The goal is to improve the prompt before the coding model receives it.

---

## Example

Input:

```text
сделай кнопку красивее
```

Output:

```text
Переработай задачу как prompt для AI code agent: улучшить внешний вид существующей кнопки, сохранив текущую логику клика и поведение компонента. Сфокусироваться на отступах, размере, цвете, hover/focus состояниях, контрасте и соответствии текущему стилю интерфейса. Не менять backend/API.
```

---

## Model

| Field | Value |
|---|---|
| Name | qquark 435M |
| Version | v0.1 |
| Parameters | ~435M |
| Format | GGUF |
| Recommended quant | Q4_K_M |
| File size | ~277 MB |
| Runtime | llama.cpp |
| Purpose | local prompt enhancement for AI code agents |
| Status | early alpha |

---

## Recommended inference settings

```text
temperature: 0.15
top_k: 10
top_p: 0.85
min_p: 0
repeat_penalty: 1.18
max_tokens: 80–120
stop: <|im_end|>
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/qquark-435m.git
cd qquark-435m
```

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install llama.cpp

Build llama.cpp with CUDA if you want GPU acceleration:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llm/llama.cpp
cd ~/llm/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
```

Add llama.cpp binaries to PATH, or use the full binary path.

---

## Download the model

Download:

```text
qquark-435m-v0.1-byte-Q4_K_M.gguf
```

Place it here:

```text
release/qquark-435m-v0.1-byte-Q4_K_M.gguf
```

Recommended distribution:
- GitHub Releases
- Hugging Face model repository

The `.gguf` file should not be committed directly to git.

---

## Run with llama.cpp

Start the server:

```bash
./scripts/run_llama_cpp.sh release/qquark-435m-v0.1-byte-Q4_K_M.gguf
```

Or manually:

```bash
llama-server \
  -m release/qquark-435m-v0.1-byte-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8088 \
  --ctx-size 2048 \
  --n-gpu-layers 999 \
  --flash-attn on
```

---

## Use qquark CLI

In another terminal:

```bash
python -m qquark.cli "сделай кнопку красивее"
```

With automatic project context:

```bash
python -m qquark.cli --context /path/to/project "кнопку круглой сделай"
```

Without context:

```bash
python -m qquark.cli --no-context "почему training killed"
```

Custom llama.cpp server:

```bash
python -m qquark.cli \
  --server http://127.0.0.1:8088 \
  "сделай задачу для агента чтобы он улучшил UI"
```

---

## Context Builder

qquark works best when it receives project context.

The CLI includes a simple Context Builder that detects common project markers:

| Marker | Meaning |
|---|---|
| `package.json` | JS/TS project |
| `vite.config.ts` | Vite frontend |
| `next.config.js` | Next.js |
| `*.uproject` | Unreal Engine |
| `CMakeLists.txt` | C/C++ |
| `pyproject.toml` | Python |
| `Cargo.toml` | Rust |
| `__manifest__.py` | Odoo |
| `docker-compose.yml` | Docker Compose |

Example:

```bash
python -m qquark.cli --context ~/my-unreal-game "кнопку круглой сделай"
```

Internally, qquark receives a prompt with detected context and constraints, so it is less likely to suggest the wrong stack.

---

## How to integrate into an app

Recommended app architecture:

```text
User input
   ↓
Context Builder
   ↓
qquark 435M via llama.cpp
   ↓
cleaned agent-ready prompt
   ↓
larger coding agent
```

Important runtime behavior:

1. Use llama.cpp `/completion`, not chat mode, for v0.1.
2. Build the prompt manually using qquark's prompt template.
3. Stop on `<|im_end|>`.
4. Also post-process by cutting everything after `<|im_end|>`.
5. Use low temperature.

Minimal post-processing:

```python
def clean_qquark_output(text: str) -> str:
    text = text.split("<|im_end|>", 1)[0]
    text = text.split("<|im_start|>", 1)[0]
    return text.strip()
```

---

## Limitations

qquark 435M v0.1 is an early prototype.

Known limitations:

- Not a general assistant.
- Not a coding model.
- It may hallucinate technologies if no context is provided.
- Git safety is not perfect.
- It can sometimes repeat phrases.
- It should be used with low temperature.
- Outputs should be post-processed after `<|im_end|>`.
- Context-aware behavior is experimental.

---

## Training summary

- Architecture: decoder-only Transformer
- Parameters: ~435M
- Vocabulary: 32k + byte fallback for llama.cpp GGUF compatibility
- Trained from scratch
- SFT on synthetic prompt-enhancement data
- Masked SFT: loss applied only to assistant output
- Exported to HF/LLaMA-compatible format
- Converted to GGUF for llama.cpp runtime
- Hardware used: RTX 4070 SUPER 12GB, WSL2

---

## Roadmap

- [ ] Improve context-aware behavior
- [ ] Improve Git safety
- [ ] Add more high-quality gold correction data
- [ ] Add OpenCode/Roo/Cline examples
- [ ] Add app integration example
- [ ] Add model card on Hugging Face
- [ ] Improve GGUF chat-template metadata
- [ ] Add Q5_K_M and Q8_0 releases

---

## License

Choose a license before release. Recommended for code: MIT.

For model weights, choose a separate model license if needed.
