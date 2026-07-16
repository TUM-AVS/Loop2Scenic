# Loop2Scenic — Retrieval (RAG) Ablation Study

This branch contains the **Level‑1 whole‑scenario multimodal‑RAG retrieval ablation** for
Loop2Scenic: the code, configs, run scripts, and results that establish *which embedder, which
query modality, and which retrieval mechanism* best retrieve a base driving scenario to seed
scenario generation.

- **Corpus:** `N = 245` curated scenarios, each a folder of *(text description, BEV video,
  `code.scenic`)*. Retrieval is **query‑by‑example** (a scenario's ground truth is itself),
  **zero‑shot** throughout.
- **Design:** exactly **one database per embedder** in its natural deployed form (multimodal
  embedders index text+video; text embedders index text). We **vary the query** (text /
  video / text+video), never rebuild the DB to match the query.
- **Full methodology & every number:** [`eval/results/rag/METHODS.md`](eval/results/rag/METHODS.md).
  Canonical aggregated results: [`eval/results/rag/summaries/summary.csv`](eval/results/rag/summaries/summary.csv).

---

## Headline findings

| Ablation | Finding |
|---|---|
| **Backbone** (7 embedders × 3 modalities) | Native **VL embedders win video & text+video decisively** (gemini t+v **96.7** HIT@1, Qwen3‑VL‑8B 88.6, 2B 75.9); **text encoders win text‑only** (Qwen3‑Emb‑8B **68.2**). Fusion is **super‑additive** (t+v ≫ either channel alone). |
| **Mechanism** (dense / BM25 / hybrid) | **Dense wins** for strong retrievers; BM25 and RRF‑hybrid help only weak ones — a **"helps‑weak, hurts‑strong"** rule. |
| **Cross‑modal fusion** (early/joint vs late) | **Joint encoding ≫ late RRF fusion** of unimodal rankings (+36 to +57 HIT@1) — true multimodal embedding cannot be faked by fusing separate searches. |
| **Reranking** (Qwen3‑VL‑Reranker 2B/8B) | Reranking **hurts** the deployed text+video query for every embedder, and a **bigger reranker hurts more** (2B 75.9→73.5→72.2; 8B 88.6→78.0→75.5; gemini 96.7→84.1→80.8) ⇒ **reranker dropped**. |
| **Text‑reranker symmetry** (Qwen3‑Reranker 0.6B/4B/8B) | Reranking **helps the weak text encoders** (+3 to +9) and is flat on the strong one (best fair number **70.2**). |
| **Modality‑gap analysis** | Matched cross‑modal cosine (0.40–0.51) is **below** within‑modal cosine (0.69–0.84) — this single fact explains video‑only near‑chance, text‑query underperformance, and the fusion advantage. |

**Deployed configuration** (ablation‑chosen): a native VL embedder — **Qwen3‑VL‑Embedding‑2B** (light/fast,
0.35 s) — with **dense retrieval, top‑k=3, no reranker**. Stronger embedders (8B, gemini) trade compute
for accuracy; reranking is dropped because it hurts the multimodal query.

---

## Tables (all in `METHODS.md`)

| Table | Content |
|---|---|
| **Table 1** | Query‑modality backbone: 7 embedders × {text, video, text+video} |
| **Table 1r / 1f** | Mechanism (dense/BM25/hybrid) · early‑vs‑late cross‑modal fusion |
| **Table 2** | Reranking on/off, full 3‑modality × 3‑VL‑embedder grid |
| **Table 2t** | Text‑encoder baselines × text reranker (symmetry) |
| **Table 3** | Modality‑gap analysis (2B / 8B / gemini) |
| **Table 4** | Reranker capacity (2B vs 8B) |

All numbers are **full bf16**, `N = 245`, HIT@1 / HIT@3 / MRR@3.

---

## Repository layout (this branch)

```
eval/
  retrieval_eval.py          # main harness: build queries, retrieve, score (HIT@1/@3, MRR@3, latency)
  aggregate_rag_results.py   # fold all run CSVs -> summary.csv (variant dedup, keep-latest)
  check_results.py           # sanity gate (NaN / range / HIT@1<=HIT@3 / MRR in [HIT@1,HIT@3] / dups)
  rerank_from_run.py         # TWO-PASS offline rerank: re-score a saved dense top-3 (VL or text reranker)
  text_reranker.py           # Qwen3-Reranker (text cross-encoder) yes/no-logit scorer
  modality_gap.py            # cross-modal vs within-modal cosine analysis (Table 3)
  bm25_baseline.py           # lexical BM25 baseline
  rrf_fusion.py              # dense+BM25 RRF hybrid + early-vs-late fusion
  video_only_captioner.py    # video->text caption bridge (qwen3.6-plus)
config/                      # per-embedder x reranker x query-modality ablation configs
scripts/                     # run scripts (backbone, reranker grids, capacity, VRAM, ingest)
eval/results/rag/METHODS.md  # full methodology, motivation, evidence trail, all tables
```

### The two‑pass reranking trick (`rerank_from_run.py`)
An 8B embedder + 8B reranker cannot co‑reside on a 32 GB GPU. But reranking only reorders the dense
**top‑3** and never needs the embedder in memory, so we run it **two‑pass**: (a) load the embedder alone,
save the dense top‑3; (b) load the reranker alone, re‑score. This is *mathematically identical* to the
co‑resident pipeline (HIT@3 invariant) and was **validated against live runs to the decimal** — at full
bf16, no quantization.

---

## Reproduce

```bash
# 1. Environment (RTX 50xx / Blackwell needs cu128)
conda activate loop2scenic          # torch==2.8.0+cu128, requirements.txt + rank_bm25, python-dotenv
export PYTHONPATH=.
set -a; source .env; set +a          # QWEN_API_KEY, GEMINI_API_KEY (gitignored)

# 2. Milvus (standalone) must be running on localhost:19530; ingest one DB per embedder:
python scripts/ingest_local_scenarios.py --config config/config_2b_videoonly.yaml --doc-modality text_video --reset

# 3. Run a backbone cell (dense, no reranker), e.g. Qwen3-VL-2B, text+video query:
python -m eval.retrieval_eval --mode text-video --config config/config_norerank.yaml

# 4. Reranking (two-pass, from a saved no-rerank run):
python -m eval.rerank_from_run --config config/config_2b_rerank8b.yaml \
       --out-suffix 8brr --source-csv eval/results/rag/runs/<embedder>/<...norerank...>.csv
#    text reranker: add  --reranker-kind text --reranker-path models/Qwen3-Reranker-8B

# 5. Aggregate + gate:
python eval/aggregate_rag_results.py          # -> summary.csv (deduped)
python eval/check_results.py                  # PASS/FAIL sanity gate
```

Batch run scripts (all long jobs run in named tmux sessions, survive disconnect):
`scripts/build_backbone_symmetric.sh`, `scripts/table2_rerank.sh`, `scripts/reranker_capacity.sh`,
`scripts/rerank_grid_phase2.sh`, `scripts/text_reranker_grid.sh`.

---

## Notes

- **Two retrieval levels are distinct.** This ablation is **Level‑1 whole‑scenario** retrieval (245
  scenarios). The Coder's **Level‑2 component‑snippet RAG** (codeICL) is a separate 922‑snippet library
  decomposed from a 145‑scenario subset (see `scripts/decompose_full.py` for an optional 245‑covering
  build into a separate collection).
- **Secrets** (`.env`, `.env.bak`) and **model weights** (`models/`) are gitignored and never committed.
- The retrieval numbers here are the paper's *retrieval‑design* results; the closed‑loop *generation*
  evaluation lives on a separate branch.
