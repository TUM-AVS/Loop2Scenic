"""Load ONE 8B model the way the eval does and report peak VRAM. Fresh process => clean number.
Usage: python scripts/measure_one_model_vram.py {embedder|reranker}
"""
import sys, torch, subprocess

def smi_used():
    out = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    return int(out.strip())

which = sys.argv[1]
base = smi_used()
torch.cuda.reset_peak_memory_stats()

if which == "embedder":
    from src.services.embedder.providers.qwen3_vl_embedding import Qwen3VLForEmbedding
    dtype = torch.bfloat16
    m = Qwen3VLForEmbedding.from_pretrained("./models/Qwen3-VL-Embedding-8B",
            torch_dtype=dtype, trust_remote_code=True).to("cuda").eval()
    label = "Qwen3-VL-Embedding-8B"
elif which == "reranker":
    from transformers import Qwen3VLForConditionalGeneration
    lm = Qwen3VLForConditionalGeneration.from_pretrained("./models/Qwen3-VL-Reranker-8B",
            trust_remote_code=True).to("cuda")
    m = lm.model  # eval keeps only lm.model (drops lm_head)
    m.eval()
    label = "Qwen3-VL-Reranker-8B (lm.model only)"
else:
    print("arg must be embedder|reranker"); sys.exit(2)

torch.cuda.synchronize()
peak_alloc = torch.cuda.max_memory_allocated()/1024**3
reserved   = torch.cuda.memory_reserved()/1024**3
smi_now    = smi_used()
print(f"MODEL: {label}")
print(f"  torch.max_memory_allocated = {peak_alloc:.2f} GiB")
print(f"  torch.memory_reserved      = {reserved:.2f} GiB")
print(f"  nvidia-smi delta           = {(smi_now-base)/1024:.2f} GiB  (base {base} -> {smi_now} MiB)")
