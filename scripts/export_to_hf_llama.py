import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import qquarkConfig

from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer


def clean_state_dict(state):
    cleaned = {}
    for k, v in state.items():
        if "rope.inv_freq" in k:
            continue
        if "rope.cos_cache" in k or "rope.sin_cache" in k:
            continue
        cleaned[k] = v
    return cleaned


def convert_qquark_to_hf(ckpt_path: str, tokenizer_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = clean_state_dict(state)

    vocab_size = state["tok_emb.weight"].shape[0]
    print("Detected vocab_size:", vocab_size)

    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=2048,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )

    print("Creating HF Llama model...")
    model = LlamaForCausalLM(config)
    hf = model.state_dict()

    # embeddings
    hf["model.embed_tokens.weight"] = state["tok_emb.weight"].clone()

    # transformer blocks
    for i in range(config.num_hidden_layers):
        prefix_q = f"blocks.{i}"
        prefix_h = f"model.layers.{i}"

        hf[f"{prefix_h}.input_layernorm.weight"] = state[f"{prefix_q}.norm1.weight"].clone()
        hf[f"{prefix_h}.post_attention_layernorm.weight"] = state[f"{prefix_q}.norm2.weight"].clone()

        qkv = state[f"{prefix_q}.attn.qkv.weight"]
        q, k, v = torch.chunk(qkv, 3, dim=0)

        hf[f"{prefix_h}.self_attn.q_proj.weight"] = q.clone()
        hf[f"{prefix_h}.self_attn.k_proj.weight"] = k.clone()
        hf[f"{prefix_h}.self_attn.v_proj.weight"] = v.clone()
        hf[f"{prefix_h}.self_attn.o_proj.weight"] = state[f"{prefix_q}.attn.out.weight"].clone()

        hf[f"{prefix_h}.mlp.gate_proj.weight"] = state[f"{prefix_q}.ffn.gate.weight"].clone()
        hf[f"{prefix_h}.mlp.up_proj.weight"] = state[f"{prefix_q}.ffn.up.weight"].clone()
        hf[f"{prefix_h}.mlp.down_proj.weight"] = state[f"{prefix_q}.ffn.down.weight"].clone()

    # final norm + lm head
    if "norm.weight" in state:
        hf["model.norm.weight"] = state["norm.weight"].clone()
    elif "ln_f.weight" in state:
        hf["model.norm.weight"] = state["ln_f.weight"].clone()
    else:
        raise KeyError("Could not find final norm weight. Expected norm.weight")

    if "lm_head.weight" in state:
        hf["lm_head.weight"] = state["lm_head.weight"].clone()
    else:
        # fallback if tied
        hf["lm_head.weight"] = state["tok_emb.weight"].clone()

    missing = [k for k in model.state_dict().keys() if k not in hf]
    if missing:
        print("Missing keys:", missing[:20], "...")
        raise RuntimeError(f"Missing {len(missing)} keys")

    print("Loading converted weights into HF model...")
    model.load_state_dict(hf, strict=True)

    print(f"Saving HF model to: {out}")
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    model.config.pad_token_id = 3
    model.config.unk_token_id = 0
    model.generation_config.bos_token_id = 1
    model.generation_config.eos_token_id = 2
    model.generation_config.pad_token_id = 3
    model.save_pretrained(out, safe_serialization=True)

    print("Saving tokenizer...")
    tok = LlamaTokenizer(
        vocab_file=tokenizer_path,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        add_bos_token=False,
        add_eos_token=False,
        legacy=True,
    )

    tok.model_max_length = 2048
    tok.padding_side = "right"
    tok.save_pretrained(out)

    # Force correct tokenizer files for llama.cpp-compatible tokenizer:
    # 0=<unk>, 1=<s>, 2=</s>, 3=<pad>
    shutil.copy2(tokenizer_path, out / "tokenizer.model")

    tokenizer_config = {
        "tokenizer_class": "LlamaTokenizer",
        "model_max_length": 2048,
        "padding_side": "right",
        "truncation_side": "right",
        "add_bos_token": False,
        "add_eos_token": False,
        "legacy": True,
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>"
    }

    special_tokens_map = {
        "unk_token": "<unk>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<pad>"
    }

    with open(out / "tokenizer_config.json", "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)

    with open(out / "special_tokens_map.json", "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, indent=2, ensure_ascii=False)

    # extra metadata
    meta = {
        "model_name": "qquark 435M",
        "version": "v0.1",
        "source_checkpoint": ckpt_path,
        "purpose": "local prompt-enhancer for AI code agents",
        "recommended_temperature": 0.2,
        "recommended_top_k": 20,
        "recommended_max_new_tokens": 160,
    }

    with open(out / "qquark_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="data/tokenizer.model")
    parser.add_argument("--out", default="release/qquark-435m-v0.1-hf")
    args = parser.parse_args()

    convert_qquark_to_hf(args.checkpoint, args.tokenizer, args.out)


if __name__ == "__main__":
    main()
