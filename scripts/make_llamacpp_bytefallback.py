import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.transformer import qquarkConfig

import torch
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2


def make_tokenizer(src_tokenizer: str, dst_tokenizer: str):
    src = Path(src_tokenizer)
    dst = Path(dst_tokenizer)
    dst.parent.mkdir(parents=True, exist_ok=True)

    model = sp_pb2.ModelProto()
    model.ParseFromString(src.read_bytes())

    pieces = list(model.pieces)

    # Original tokenizer:
    # 0 <pad>, 1 <unk>, 2 <s>, 3 </s>
    #
    # llama.cpp-compatible tokenizer:
    # 0 <unk>, 1 <s>, 2 </s>, 3 <pad>
    order = [1, 2, 3, 0] + list(range(4, len(pieces)))

    new_pieces = [pieces[i] for i in order]
    existing = {p.piece for p in new_pieces}

    # Add byte fallback tokens: <0x00> ... <0xFF>
    for b in range(256):
        piece = f"<0x{b:02X}>"
        if piece not in existing:
            p = sp_pb2.ModelProto().SentencePiece()
            p.piece = piece
            p.score = 0.0
            p.type = sp_pb2.ModelProto.SentencePiece.Type.BYTE
            new_pieces.append(p)

    del model.pieces[:]
    for p in new_pieces:
        model.pieces.append(p)

    model.trainer_spec.vocab_size = len(model.pieces)
    model.trainer_spec.unk_id = 0
    model.trainer_spec.bos_id = 1
    model.trainer_spec.eos_id = 2
    model.trainer_spec.pad_id = 3
    model.trainer_spec.byte_fallback = True

    dst.write_bytes(model.SerializeToString())

    sp = spm.SentencePieceProcessor(model_file=str(dst))

    print("Saved tokenizer:", dst)
    print("piece_size:", sp.get_piece_size())
    print("unk_id:", sp.unk_id(), sp.id_to_piece(sp.unk_id()))
    print("bos_id:", sp.bos_id(), sp.id_to_piece(sp.bos_id()))
    print("eos_id:", sp.eos_id(), sp.id_to_piece(sp.eos_id()))
    print("pad_id:", sp.pad_id(), sp.id_to_piece(sp.pad_id()))
    print("<0x0A> id:", sp.piece_to_id("<0x0A>"))

    ids = sp.encode("сделай кнопку красивее\nновая строка")
    print("test tokens:", len(ids), ids[:40])
    print("decoded:", sp.decode(ids))


def make_checkpoint(src_ckpt: str, dst_ckpt: str, old_vocab=32000, new_vocab=32256):
    src = Path(src_ckpt)
    dst = Path(dst_ckpt)
    dst.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt

    # New token ids:
    # 0 <unk>  <- old 1
    # 1 <s>    <- old 2
    # 2 </s>   <- old 3
    # 3 <pad>  <- old 0
    order = torch.tensor([1, 2, 3, 0] + list(range(4, old_vocab)), dtype=torch.long)

    def reorder_and_expand(w: torch.Tensor, name: str):
        print("Reorder + expand:", name, tuple(w.shape))

        w = w[order].contiguous()

        extra_rows = new_vocab - old_vocab
        extra = torch.zeros((extra_rows, w.shape[1]), dtype=w.dtype)

        # Tiny random init for byte fallback rows.
        std = w.float().std().item()
        if std > 0:
            extra.normal_(mean=0.0, std=std * 0.02)

        return torch.cat([w, extra], dim=0).contiguous()

    if "tok_emb.weight" not in state:
        raise KeyError("tok_emb.weight not found")

    state["tok_emb.weight"] = reorder_and_expand(
        state["tok_emb.weight"],
        "tok_emb.weight",
    )

    if "lm_head.weight" in state:
        state["lm_head.weight"] = reorder_and_expand(
            state["lm_head.weight"],
            "lm_head.weight",
        )
    else:
        print("lm_head.weight not found; skipping")

    # Patch config vocab size if possible.
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg = ckpt["config"]
        if isinstance(cfg, dict):
            cfg["vocab_size"] = new_vocab
        else:
            try:
                cfg.vocab_size = new_vocab
            except Exception:
                pass

    torch.save(ckpt, dst)
    print("Saved checkpoint:", dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/qquark_435m_sft_v0_1.pt",
    )
    parser.add_argument(
        "--tokenizer",
        default="data/tokenizer.model",
    )
    parser.add_argument(
        "--out-checkpoint",
        default="checkpoints/qquark_435m_sft_v0_1_llamacpp_byte.pt",
    )
    parser.add_argument(
        "--out-tokenizer",
        default="release/qquark-435m-v0.1-llamacpp-byte-tokenizer.model",
    )
    args = parser.parse_args()

    make_tokenizer(args.tokenizer, args.out_tokenizer)
    make_checkpoint(args.checkpoint, args.out_checkpoint)


if __name__ == "__main__":
    main()
