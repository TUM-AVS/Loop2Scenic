# RAG ablation — methods, motivation, and evidence trail

Companion to `summaries/summary.csv` (canonical numbers; aggregated **2026-07-20**) and
`METHODS-OLD.md` (prior N=245 write-up). This document records methodology and results for the
**re-run on this workstation** so every number can be traced to a stamped CSV under
`eval/results/rag/runs/`.

Machine: local workstation (Mifcom2). Corpus: **N = 200** driving scenarios, each a folder with
`new_description.txt` (indexed doc text), `text_query_description.txt` (fair MLLM caption query),
`video_only_description.txt` (pure-video caption via qwen3.6-plus), `BEV.mp4`, `code.scenic`.
Query-by-example: GT for a query = the scenario folder name.

## Section & table status index


| Section   | What                                       | Table(s)                          | Status (this re-run)                                      |
| --------- | ------------------------------------------ | --------------------------------- | --------------------------------------------------------- |
| §4 A0     | query-modality backbone (embedder × query) | **Table 1**                       | ✅ done                                                    |
| §5 A1     | rerank on/off, all query modalities        | **Table 2**                       | ✅ done (prefer HIT@3-invariant / phase-2 cells)           |
| §5 A1     | text-encoder × text reranker               | **Table 2t**                      | ⚠️ on **combined** (bridge) queries                        |
| §5 A1     | reranker capacity (2B vs 8B)               | **Table 4**                       | ✅ done                                                    |
| §6 B1/B1b | dense / BM25 / hybrid                      | **Table 1r**                      | ✅ done (14 dense×BM25 fusions)                            |
| §6        | early vs late cross-modal fusion           | **Table 1f**                      | ✅ done (3 VL late fusions)                               |
| §6b       | caption-bridge                             | (rows in Table 1)                 | ✅ done                                                    |
| §9        | modality-gap analysis                      | **Table 3**                       | ✅ done (N=200)                                            |


Metrics: HIT@1, HIT@3, MRR@3, latency(s), avg top-1 cosine. All retained rows have
`no_error_rate = 100` on **N = 200**. Absolute HIT@1 is **not byte-comparable** to OLD (N=245);
design conclusions are.

---

## 0 · Design: one database per embedder, ablate the query

- **Multimodal embedders** (Qwen3-VL-2B/8B, gemini-embedding-2): DB = **text+video**. Ablate query:
  text / video / text+video.
- **Text embedders** (bge-m3, Qwen3-Embedding-0.6B/4B/8B): DB = **text**. Video enters only via the
  qwen3.6-plus **caption bridge** (§0b / §6b).

Video-matched DBs are discarded (same `BEV.mp4` on both sides → self-copy leak). Text has an
independent paraphrase (`text_query_description.txt` ≠ `new_description.txt`); video does not.

## 0b · Captioners & zero-shot

Everything is zero-shot. Captioners:

- Fair text query → `text_query_description.txt`
- Video → text (bridge) → `video_only_description.txt` (`eval/video_only_captioner.py`)
- Bridge text+video → concatenate caption + video-only caption (`--query-text-source combined`)

## 1 · Environment

- Conda `ads-mrag`, `PYTHONPATH=.`, `.env` for API keys
- Milvus via `docker compose` (`localhost:19530`)
- Ingest: `scripts/ingest_local_scenarios.py --doc-modality {text_video,text_only,video_only}`
- Eval: `python -m eval.retrieval_eval` (**dot**, not slash)
- Offline rerank: `python -m eval.rerank_from_run`
- BM25 / RRF: `eval/bm25_baseline.py`, `eval/rrf_fusion.py` (unique per-dense filenames)
- Aggregate: `python eval/aggregate_rag_results.py` → `summaries/summary.csv`
- Logs: `logs/ads_mrag_*.log`

## 2 · Provenance

Default query = `text_query_description.txt` (fair). Indexed text = `new_description.txt`.
Self-retrieval with indexed text is diagnostic only, not a system result.

## 3 · Mechanism sketch

**Table 1** = Stage 1 dense bi-encoder + Milvus cosine (`IVF_FLAT`, top_k=3).
**Tables 2/4** = Stage 1 + Stage 2 cross-encoder on the top-3 shortlist.
HIT@3 is invariant under rerank when the shortlist is fixed. Live runs whose HIT@3 ≠ dense parent
used a different index and are footnoted / omitted from the primary Table 2 grid.

Level-1 (this doc) retrieves a whole scenario → folder-join to `code.scenic`. Level-2 snippet RAG
is out of scope.

---

## 4 · A0 — Backbone · Table 1

Dense, no reranker. VL = text+video DB; text encoders = text DB. † = caption bridge.
Best HIT@1 per query group in **bold**.


| Embedding model              | dim  | DB         | HIT@1    | HIT@3 | MRR@3 | lat(s) | sim   |
| ---------------------------- | ---- | ---------- | -------- | ----- | ----- | ------ | ----- |
| **══ Query = text ══**       |      |            |          |       |       |        |       |
| Qwen3-VL-2B                  | 2048 | text+video | 36.0     | 56.5  | 0.447 | 0.021  | 0.826 |
| Qwen3-VL-8B                  | 4096 | text+video | 43.5     | 66.5  | 0.541 | 0.037  | 0.753 |
| gemini-emb-2                 | 3072 | text+video | 57.0     | 85.0  | 0.693 | 0.289  | 0.567 |
| bge-m3                       | 1024 | text       | 52.0     | 71.5  | 0.605 | 0.020  | 0.947 |
| Qwen3-Emb-0.6B               | 1024 | text       | 44.0     | 66.0  | 0.539 | 0.019  | 0.851 |
| Qwen3-Emb-4B                 | 2560 | text       | 49.5     | 76.0  | 0.612 | 0.031  | 0.846 |
| Qwen3-Emb-8B                 | 4096 | text       | **67.0** | 87.0  | 0.762 | 0.045  | 0.877 |
| **══ Query = video ══**      |      |            |          |       |       |        |       |
| Qwen3-VL-2B                  | 2048 | text+video | 18.5     | 32.5  | 0.249 | 0.369  | 0.769 |
| Qwen3-VL-8B                  | 4096 | text+video | 45.5     | 65.0  | 0.538 | 0.608  | 0.720 |
| gemini-emb-2                 | 3072 | text+video | **51.0** | 70.5  | 0.595 | 2.604  | 0.802 |
| bge-m3 †                     | 1024 | text       | 4.0      | 9.0   | 0.061 | 0.019  | 0.918 |
| Qwen3-Emb-0.6B †             | 1024 | text       | 3.0      | 6.0   | 0.044 | 0.018  | 0.860 |
| Qwen3-Emb-4B †               | 2560 | text       | 4.0      | 8.5   | 0.060 | 0.030  | 0.838 |
| Qwen3-Emb-8B †               | 4096 | text       | 5.0      | 11.5  | 0.075 | 0.043  | 0.825 |
| **══ Query = text+video ══** |      |            |          |       |       |        |       |
| Qwen3-VL-2B                  | 2048 | text+video | 78.0     | 91.0  | 0.838 | 0.378  | 0.901 |
| Qwen3-VL-8B                  | 4096 | text+video | 89.5     | 99.0  | 0.940 | 0.621  | 0.888 |
| gemini-emb-2                 | 3072 | text+video | **97.0** | 99.0  | 0.980 | 2.586  | 0.970 |
| bge-m3 †                     | 1024 | text       | 46.0     | 65.0  | 0.542 | 0.024  | 0.939 |
| Qwen3-Emb-0.6B †             | 1024 | text       | 20.0     | 36.0  | 0.267 | 0.020  | 0.865 |
| Qwen3-Emb-4B †               | 2560 | text       | 32.5     | 54.5  | 0.423 | 0.039  | 0.873 |
| Qwen3-Emb-8B †               | 4096 | text       | 46.0     | 70.5  | 0.566 | 0.061  | 0.852 |


**Findings.** Text query: text encoders win the fair race (Emb-8B **67.0**). Video / text+video:
native VL ≫ bridge (video ≤5.0 bridge vs ≤51.0 native; text+video ≤46.0 bridge vs **97.0** native).
VL fusion is super-additive (e.g. 2B 78.0 ≫ 36.0 / 18.5).

---

## 5 · A1 rerank · Tables 2, 2t, 4

### Table 2 — VL × modality × reranker size

Cells: HIT@1 / HIT@3 / MRR@3. Dense = Table 1. Rerankers = Qwen3-VL-Reranker-2B / 8B.
Prefer phase-2 / HIT@3-invariant cells (`top_k=3`).


| Query                           | VL embedder | dense               | + Reranker-2B       | + Reranker-8B       |
| ------------------------------- | ----------- | ------------------- | ------------------- | ------------------- |
| **══ text-only ══**             |             |                     |                     |                     |
| text                            | Qwen3-VL-2B | 36.0 / 56.5 / 0.447 | 47.0 / 56.5 / 0.516 | 47.0 / 56.5 / 0.516 |
| text                            | Qwen3-VL-8B | 43.5 / 66.5 / 0.541 | 57.5 / 66.5 / 0.618 | 53.0 / 66.5 / 0.597 |
| text                            | gemini      | 57.0 / 85.0 / 0.693 | 69.5 / 85.0 / 0.768 | 68.0 / 85.0 / 0.759 |
| **══ video-only ══**            |             |                     |                     |                     |
| video                           | Qwen3-VL-2B | 18.5 / 32.5 / 0.249 | 8.5 / 32.5 / 0.185  | 8.5 / 32.5 / 0.185  |
| video                           | Qwen3-VL-8B | 45.5 / 65.0 / 0.538 | 24.5 / 65.0 / 0.414 | 21.0 / 65.0 / 0.402 |
| video                           | gemini      | 51.0 / 70.5 / 0.595 | 25.5 / 70.5 / 0.438 | 27.0 / 70.5 / 0.455 |
| **══ text+video (deployed) ══** |             |                     |                     |                     |
| text+video                      | Qwen3-VL-2B | 78.0 / 91.0 / 0.838 | 73.0 / 91.0 / 0.815 | 73.5 / 91.0 / 0.814 |
| text+video                      | Qwen3-VL-8B | 89.5 / 99.0 / 0.940 | 75.0 / 99.0 / 0.863 | 76.0 / 99.0 / 0.868 |
| text+video                      | gemini      | 97.0 / 99.0 / 0.980 | 76.5 / 99.0 / 0.868 | 77.0 / 99.0 / 0.873 |


Text-only + Reranker-2B for 8B/gemini uses offline phase-2 (`*phase2_2brr.csv`) on the Table-1
dense shortlist (HIT@3 invariant). Older live `vl-2B` runs (HIT@3 ≠ dense) are not used here.

**Rule.** Text-only: rerank **helps** (2B ≥ 8B on these cells). Video-only: **hurts**. Text+video
(deployed): **hurts**; larger reranker does not restore dense. **Drop reranker for the deployed
multimodal pipeline.**

### Table 2t — Text encoders × text reranker

Qwen3-Reranker-0.6B/4B/8B applied to **`__combined__`** (bridge two-texts) queries in this re-run.
HIT@3 invariant within each combined dense parent. Cells: HIT@1 / HIT@3 / MRR@3.


| Embedder       | dense combined      | + Reranker-0.6B     | + Reranker-4B           | + Reranker-8B       |
| -------------- | ------------------- | ------------------- | ----------------------- | ------------------- |
| bge-m3         | 46.0 / 65.0 / 0.542 | 51.5 / 65.0 / 0.577 | 51.5 / 65.0 / 0.576     | 53.0 / 65.0 / 0.582 |
| Qwen3-Emb-0.6B | 20.0 / 36.0 / 0.267 | 28.0 / 36.0 / 0.318 | 28.5 / 36.0 / 0.321     | 28.5 / 36.0 / 0.318 |
| Qwen3-Emb-4B   | 32.5 / 54.5 / 0.423 | 45.0 / 54.5 / 0.494 | 45.5 / 54.5 / 0.499     | 47.5 / 54.5 / 0.505 |
| Qwen3-Emb-8B   | 46.0 / 70.5 / 0.566 | 53.5 / 70.5 / 0.617 | **58.5 / 70.5 / 0.643** | 53.5 / 70.5 / 0.614 |


Plain-caption dense + VL-2B reranker (reference): bge 52.0→60.5, Emb-0.6B 44.0→55.0, Emb-4B
49.5→67.0, Emb-8B 67.0→70.5 (HIT@3 unchanged). Same qualitative rule: helps weak / flatter on strong.

### Table 4 — Reranker capacity


| Cell                | dense               | + 2B reranker       | + 8B reranker       |
| ------------------- | ------------------- | ------------------- | ------------------- |
| 2B-emb · text-video | 78.0 / 91.0 / 0.838 | 73.0 / 91.0 / 0.815 | 73.5 / 91.0 / 0.814 |
| 2B-emb · video-only | 18.5 / 32.5 / 0.249 | 8.5 / 32.5 / 0.185  | 8.5 / 32.5 / 0.185  |
| 8B-emb · text-video | 89.5 / 99.0 / 0.940 | 75.0 / 99.0 / 0.863 | 76.0 / 99.0 / 0.868 |
| gemini · text-video | 97.0 / 99.0 / 0.980 | 76.5 / 99.0 / 0.868 | 77.0 / 99.0 / 0.873 |


**Answer:** capacity is not the issue — 8B does not fix the text+video hurt. Drop reranking for
deployment.

---

## 6 · Mechanism baselines — BM25 / hybrid / fusion

### B1 — BM25 alone

| Retriever | HIT@1 | HIT@3 | MRR@3 | lat(s) |
| --------- | ----- | ----- | ----- | ------ |
| BM25      | 51.5  | 78.0  | 0.633 | 0.002  |

Corpus = `new_description.txt`; query = `text_query_description.txt`
(`eval/bm25_baseline.py --source mllm_caption`).

### Table 1r — dense vs + BM25 hybrid (RRF)

Offline `eval/rrf_fusion.py` (k=60) over each dense ranking + the BM25 ranking.
Cells: HIT@1 / HIT@3 / MRR@3; Δ = HIT@1(hybrid − dense).


| Embedder                                                 | dense               | + BM25 hybrid       | Δ HIT@1 |
| -------------------------------------------------------- | ------------------- | ------------------- | ------- |
| *Query = text*                                           |                     |                     |         |
| Qwen3-VL-2B                                              | 36.0 / 56.5 / 0.447 | 47.0 / 70.5 / 0.578 | +11.0   |
| Qwen3-VL-8B                                              | 43.5 / 66.5 / 0.541 | 54.5 / 77.5 / 0.646 | +11.0   |
| gemini                                                   | 57.0 / 85.0 / 0.693 | 62.0 / 86.0 / 0.729 | +5.0    |
| bge-m3                                                   | 52.0 / 71.5 / 0.605 | 57.0 / 77.5 / 0.660 | +5.0    |
| Qwen3-Emb-0.6B                                           | 44.0 / 66.0 / 0.539 | 50.0 / 78.0 / 0.627 | +6.0    |
| Qwen3-Emb-4B                                             | 49.5 / 76.0 / 0.612 | 58.0 / 80.5 / 0.682 | +8.5    |
| Qwen3-Emb-8B                                             | 67.0 / 87.0 / 0.762 | 65.0 / 87.0 / 0.751 | −2.0    |
| *Query = text+video* (VL native; † text-embedder bridge) |                     |                     |         |
| Qwen3-VL-2B                                              | 78.0 / 91.0 / 0.838 | 76.5 / 91.5 / 0.833 | −1.5    |
| Qwen3-VL-8B                                              | 89.5 / 99.0 / 0.940 | 80.5 / 98.0 / 0.882 | −9.0    |
| gemini                                                   | 97.0 / 99.0 / 0.980 | 86.0 / 99.0 / 0.923 | −11.0   |
| bge-m3 †                                                 | 46.0 / 65.0 / 0.542 | 53.5 / 74.0 / 0.624 | +7.5    |
| Qwen3-Emb-0.6B †                                         | 20.0 / 36.0 / 0.267 | 26.5 / 69.0 / 0.461 | +6.5    |
| Qwen3-Emb-4B †                                           | 32.5 / 54.5 / 0.423 | 43.0 / 76.5 / 0.583 | +10.5   |
| Qwen3-Emb-8B †                                           | 46.0 / 70.5 / 0.566 | 54.5 / 82.5 / 0.676 | +8.5    |


**Rule (reproduced).** Hybrid **helps weak dense** (+5 to +11) and **hurts strong multimodal dense**
(−1.5 to −11). BM25 (51.5) beats many text/bridge dense arms but sits far below native text+video
(97.0). **Deployed = pure dense;** lexical fusion only rescues a weak embedder.

### Table 1f — early/joint vs late fusion

Cross-modal fusion on VL embedders (text+video query setting). **early/joint** = one dense
`text-video` encoding (Table 1 deployed rows). **late** = RRF of the same embedder’s text-only
ranking × video-only ranking (`eval/rrf_fusion.py`). Cells: HIT@1 / HIT@3 / MRR@3.


| VL embedder | text-only           | video-only          | late fusion (RRF)   | **early/joint**         |
| ----------- | ------------------- | ------------------- | ------------------- | ----------------------- |
| Qwen3-VL-2B | 36.0 / 56.5 / 0.447 | 18.5 / 32.5 / 0.249 | 42.5 / 59.5 / 0.498 | **78.0 / 91.0 / 0.838** |
| Qwen3-VL-8B | 43.5 / 66.5 / 0.541 | 45.5 / 65.0 / 0.538 | 58.5 / 86.5 / 0.715 | **89.5 / 99.0 / 0.940** |
| gemini      | 57.0 / 85.0 / 0.693 | 51.0 / 70.5 / 0.595 | 75.0 / 87.5 / 0.807 | **97.0 / 99.0 / 0.980** |


**Joint ≫ late by +22 to +35.5 HIT@1** (gemini +22.0, 8B +31.0, 2B +35.5). Late fusion beats
either unimodal arm alone but cannot match joint encoding — text↔video interactions in one
vector are not recoverable by fusing unimodal rankings. Justifies a true multimodal embedder
for the deployed text+video pipeline.

> **Provenance note.** Late-fusion CSVs share the dense `text_only` slug with dense×BM25 hybrids,
> so `summary.csv` dedup keeps the newer late rows under those VL text-only hybrid keys. Table 1r
> numbers above are from the BM25 hybrid files (`*_141643_*`); Table 1f late cells from
> `*_141955_*` (and the gemini late at `*_141008_*`). Prefer the stamped run CSVs over those
> colliding summary rows.

## 6b · Caption bridge

Native VL video / text+video ≫ text-encoder + caption bridge (Table 1 †). Combined captions do not
close the gap to native multimodal (e.g. bge combined 46.0 vs gemini native text+video 97.0).

---

## 9 · Modality gap · Table 3

From `summaries/modality_gap_*.json` (N=200):


| Embedder           | matched cos(t,v) | within-text | within-video | centroid gap | text–joint / video–joint |
| ------------------ | ---------------- | ----------- | ------------ | ------------ | ------------------------ |
| Qwen3-VL-2B        | 0.51             | 0.80        | 0.81         | 0.80         | 0.90 / 0.72              |
| Qwen3-VL-8B        | 0.39             | 0.72        | 0.69         | 0.84         | 0.83 / 0.69              |
| gemini-embedding-2 | 0.42             | 0.84        | 0.85         | 0.94         | 0.60 / 0.79              |


Matched cross-modal cosine (≈0.39–0.51) stays **below** within-modal (≈0.69–0.85), explaining weak
video-only / bridge-video retrieval, text-query underperformance against joint docs, and the joint
encoding advantage.

---

## Statistical protocol

N = **200** (OLD used 245). Random floor HIT@1 ≈ 0.5%. 95% CI on HIT@1 ≈ ±6–7 pts. Saturation
guard: fair caption text+video HIT@1 ≤ 97.0 (gemini) in this re-run.

## Harness notes

1. Use `python -m eval.retrieval_eval` (dot), not `eval/retrieval_eval`.
2. Re-ingest multimodal DBs after text-only collection resets before capacity / deployed evals.
3. Prefer phase-2 offline rerank when comparing to Table-1 dense (HIT@3 invariant).
4. `rrf_fusion.py` writes unique per-dense filenames (`rrf-hybrid::<slug>`) so aggregator keeps
   all hybrid rows. Late fusion (text×video) currently reuses the same dense slug as dense×BM25 —
   see Table 1f provenance note; include the sparse stem in the slug if re-aggregating both.

## Current status (2026-07-20)

**DONE:** Table 1; Table 2/4 (primary cells); Table 2t on combined; Table 1r (14 hybrids + BM25);
Table 1f (3 VL late fusions); Table 3; caption bridge; aggregate `summary.csv` (72 rows).

**PENDING for full OLD parity:** Table 2t on plain caption (not only combined); optional N=245
corpus re-run; optional `rrf_fusion.py` slug fix so late vs hybrid do not collide in summary.

**Deployed configuration:** native VL embedder, **dense** retrieval, **text+video** query,
**top_k=3**, **no reranker**.
