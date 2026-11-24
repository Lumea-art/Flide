"""llama3 rope against transformers, including positions past the original context."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import LlamaConfig  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding  # noqa: E402

from flide.backbones.llama.rope import cos_sin  # noqa: E402

DIR = Path(__file__).resolve().parents[1] / "models/Llama-3.2-1B-Instruct"


def main():
    if not DIR.exists():
        print("SKIP: %s is not downloaded" % DIR)
        return
    cfg = json.loads((DIR / "config.json").read_text())
    pos = torch.arange(0, 9000)
    cos, sin = cos_sin(cfg["head_dim"], cfg["rope_theta"], pos, cfg["rope_scaling"])
    ref = LlamaRotaryEmbedding(config=LlamaConfig.from_pretrained(str(DIR)))
    zeros = torch.zeros(1, len(pos), cfg["num_attention_heads"], cfg["head_dim"])
    with torch.no_grad():
        ref_cos, ref_sin = ref(zeros, pos[None])
    gap = max((cos - ref_cos).abs().max().item(), (sin - ref_sin).abs().max().item())
    print("llama3 rope, positions 0..9000 (original context %d): max abs difference %.3e"
          % (cfg["rope_scaling"]["original_max_position_embeddings"], gap))
    print("ALL PASS" if gap < 1e-5 else "FAILED")
    sys.exit(0 if gap < 1e-5 else 1)


if __name__ == "__main__":
    main()
