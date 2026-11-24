"""The MLA block against transformers' own, on a tiny random checkpoint. No GPU, no weights needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM  # noqa: E402

from flide.backbones.deepseek.mla import attention, rotary_freqs  # noqa: E402


def tiny(q_lora=None, seed=0):
    torch.manual_seed(seed)
    cfg = DeepseekV2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, kv_lora_rank=16, q_lora_rank=q_lora, qk_nope_head_dim=8, qk_rope_head_dim=4,
        v_head_dim=8, n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, moe_intermediate_size=32,
        first_k_dense_replace=1, max_position_embeddings=128, rms_norm_eps=1e-6, tie_word_embeddings=False,
    )
    return cfg, DeepseekV2ForCausalLM(cfg).eval()


def layer_weights(attn, cfg, q_lora):
    w = {
        "kv_a_proj_with_mqa": attn.kv_a_proj_with_mqa,
        "kv_a_layernorm": attn.kv_a_layernorm,
        "kv_b_proj": attn.kv_b_proj,
        "o_proj": attn.o_proj,
        "kv_lora_rank": cfg.kv_lora_rank,
    }
    if q_lora:
        w.update(q_a_proj=attn.q_a_proj, q_a_layernorm=attn.q_a_layernorm, q_b_proj=attn.q_b_proj)
    else:
        w["q_proj"] = attn.q_proj
    return w


def check(q_lora=None):
    cfg, model = tiny(q_lora)
    attn = model.model.layers[0].self_attn
    b, t = 2, 7
    hidden = torch.randn(b, t, cfg.hidden_size)
    pos = torch.arange(t)[None, :].expand(b, t)
    with torch.no_grad():
        freqs = rotary_freqs(pos, cfg.qk_rope_head_dim)
        want, _ = attn(hidden, position_embeddings=model.model.rotary_emb(hidden, pos))
        got = attention(hidden, layer_weights(attn, cfg, q_lora), freqs, cfg.num_attention_heads,
                        cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.rms_norm_eps,
                        q_lora=q_lora)
    gap = (got - want).abs().max().item()
    print("%-28s max abs difference %.3e" % ("q_lora" if q_lora else "no q_lora", gap))
    return gap


def main():
    fails = sum(gap > 1e-6 for gap in (check(None), check(24)))
    print("ALL PASS" if not fails else "%d FAILED" % fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
