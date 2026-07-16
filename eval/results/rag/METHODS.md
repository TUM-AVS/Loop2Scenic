# RAG ablation — methods, motivation, and evidence trail

Companion to `ABLATION_LOG.md` (dated findings) and `summaries/summary.csv` (numbers). This file
records the **methodology and the load-bearing design decisions** so any number can be defended or
reproduced. Each table now lives INSIDE its methodology section (§4–§9); the index below shows
per-section completion.

Machine: local RTX 5090 32 GB workstation. Data/models originate on lab server avsaw1.
Corpus: **245 driving scenarios**, each a folder with `new_description.txt` (indexed doc text),
`text_query_description.txt` (MLLM caption of video+desc), `video_only_description.txt` (caption of
video ONLY), `BEV.mp4`, `code.scenic`. Query-by-example: GT for a query = the scenario itself.

## Section & table status index


| Section   | What                                       | Table(s)                                               | Status                                           |
| --------- | ------------------------------------------ | ------------------------------------------------------ | ------------------------------------------------ |
| §4 A0     | query-modality backbone (embedder × query) | **Table 1**                                            | ✅ done                                           |
| §5 A1     | rerank on/off, all query modalities        | **Table 2** (2B vs 8B reranker, text/video/text+video) | ✅ done (full 3-modality × 3-embedder grid, bf16) |
| §5 A1     | text-encoder baselines × text reranker     | **Table 2t** (Qwen3-Reranker-0.6B/4B/8B, text-only)    | ✅ done (4 text embedders, bf16, symmetry)        |
| §5 A1     | reranker capacity (2B vs 8B reranker)      | **Table 4**                                            | ✅ done (2B/8B/gemini embedders, full bf16)       |
| §5 A3     | DSL-vs-raw query repr                      | (in-text)                                              | ✅ done                                           |
| §5 C1     | funnel width (top_k)                       | (appendix fig)                                         | ✅ done                                           |
| §6 B1/B1b | mechanism: dense/BM25/hybrid               | **Table 1r**                                           | ✅ done                                           |
| §6        | cross-modal fusion: early vs late          | **Table 1f**                                           | ✅ done                                           |
| §6        | mechanism-coverage matrix                  | (matrix)                                               | ✅ done                                           |
| §6b       | caption-bridge (text encoder + video)      | (rows in Table 1)                                      | ✅ done                                           |
| §9        | modality-gap analysis                      | **Table 3**                                            | ✅ done (2B/8B/gemini)                            |


Metrics everywhere: HIT@1, HIT@3, MRR@3 (rank@3), retrieval latency(s), avg top-1 cosine. N = 245,
zero-shot throughout. Canonical numbers: `summaries/summary.csv`.

---



## 0 · The design: one database per embedder, ablate the query

This is a **query-modality ablation**. There is exactly ONE database per embedder, in its natural
deployed form — we never rebuild the database to "match" the query:

- **Multimodal embedders** (Qwen3-VL-2B/8B, gemini): database = **text+video** (they index the
description + BEV video jointly). We ablate the **query**: text / video / text+video.
- **Text embedders** (bge-m3, Qwen3-Embedding-0.6B/4B/8B): database = **text** (they cannot embed
video — a capability limit, not a choice). Query = text; video enters only via the zero-shot
qwen3.6-plus **caption bridge** (§0b).

**Why not build modality-matched databases (the earlier "symmetric" idea, now discarded).** Matching
a database to the query is harmless for text (query caption ≠ indexed description → real match) but
catastrophic for video: a video-only database + a video query is the *same BEV file on both sides*
→ identical vectors → 98.8 self-copy **leak**. The asymmetry is fundamental: **text has an LLM
paraphrase (independent query), video does not** (one clip per scenario), so an independent video
query is impossible. We therefore use the real text+video database and accept that the text+video
*query* reuses the DB's video (a stated Limitation — shared-video boost).

## 0b · Captioners & zero-shot statement (for the setup table)

**Everything is zero-shot** — no fine-tuning anywhere (grep-confirmed: only prompt text, no
`.fit()`/LoRA/PEFT). All embedders (Qwen3-VL-2B/8B, gemini-embedding-2, bge-m3,
Qwen3-Embedding-0.6B/4B/8B), the reranker (Qwen3-VL-Reranker-2B/8B), and both captioners are
off-the-shelf checkpoints/APIs, prompted at temperature 0.

**Captioners (modality → text):**

- **Video → text:** `qwen3.6-plus` (Qwen token-plan endpoint), zero-shot, temp 0 — produces
`video_only_description.txt`. The ONLY VLM used for modality conversion; captioner for the bridge.
- **Text query: no VLM needed.** We use the pre-existing `text_query_description.txt` (independent
scenario description, generated once by `gemini-3.1-flash-lite` at temp 0). It is the SAME text
query for the native VL path and the bridge path, so native-vs-bridge differs only in video
handling — the text query cancels out and is not a confound.

**Bridge "two texts" (text-video for a text encoder):** `text_query_description` (text) +
`video_only_description` (qwen3.6-plus video caption), concatenated, embedded by the text encoder.

## 1 · Environment & data parity (Phase 0)

**Motivation.** No number is meaningful unless index contents, checkpoints, and query protocol are
pinned; AAAI grades a reproducibility checklist.

**How.**

- Conda env `loop2scenic` (py3.11), `torch==2.8.0+cu128` (RTX 5090 = Blackwell sm_120 needs cu128),
`requirements.txt` + `rank_bm25`, `python-dotenv`.
- Data via `ssh-copy-id` + `rsync` from avsaw1: 245 scenarios (243 MB), model checkpoints
(Qwen3-VL-Embedding-2B/8B, Qwen3-VL-Reranker-2B), later the Qwen3-Embedding-0.6B/4B/8B text
encoders (HF, ~24 GB) and bge-m3 (sentence-transformers).
- Milvus v2.3.3 standalone via three `docker run` containers (etcd + minio + milvus, network-aliased).
- Ingestion is **deterministic and mostly API-free**: `scripts/ingest_local_scenarios.py` reads the
pre-existing `new_description.txt/.json` (no VLM interpret call), embeds with the configured
embedder, id = folder name. `--doc-modality {text_video,text_only,video_only}` selects what the
document embeds. One collection per (embedder, doc-modality).
- All long runs execute in **named tmux sessions** with tee'd logs under `logs/` (survives laptop
shutdown — they run on the workstation).



## 2 · Query & document provenance (audit)

**Motivation.** A retrieval eval is invalid if the query is not independent of the indexed document;
if query == indexed text it measures self-retrieval and saturates.

**Audit results.**

- **image-16 =** `kept_from_image16/rag_cross_model/` **(June-21/22).** Every cell reproduces exactly.
**All queries ==** `text_query_description.txt` **(MLLM caption)**, differing from the indexed text for
**245/245** scenarios → image-16 is NOT self-retrieval.
- **image-16 used a modality-matched (text-only) index for its text cells.** Confirmed by
reproduction: a local text-only 2B index + text-only query yields **50.2 HIT@1 ≈ image-16's 51.4**;
the same query on a text+video index yields 36.3. The ~15-pt gap is index modality, not a bug.
(We adopt the single text+video DB — §0 — so we report 36.3, not the modality-matched 50.2.)
- **A separate June-1 experiment WAS leaky** (`legacy_leaky_20260601/`): query ==
`new_description.txt`, 100% in all 9 files. Never published in image-16; audit trail only.
(Self-retrieval ceiling: indexed-text query → 85.7 HIT@1 vs caption 36.3 on the same index.)



## 3 · Retrieval mechanism (how the backbone works — dense semantic search)

The backbone (Table 1, §4) is **classic dense retrieval — embeddings + cosine similarity, no
reranker, no LLM**:

1. **Index (offline):** each of 245 scenarios → one embedding vector (VL embedders embed text+video;
  text embedders embed text). Stored in **Milvus**, index `IVF_FLAT`, metric `cosine` (nlist=1024,
   nprobe=10).
2. **Embed query:** the query (text / video / text+video) → one vector via the **same** embedder.
3. **Similarity search:** Milvus returns nearest neighbours by **cosine similarity**, ranked.
4. **Score:** top-1 = HIT@1; GT in top-3 = HIT@3; rank of GT → MRR@3; cosine of top-1 → `avg-sim`.
  BM25 replaces steps 1–3 with lexical (bag-of-words) matching instead of embeddings.



### Workflow: retrieval (Table 1) vs retrieval + rerank (Table 2)

```
OFFLINE (build the database, once)
  245 scenarios ──[ embedding model ]──▶ 245 vectors ──▶  Milvus vector DB
                                                          (index IVF_FLAT, metric cosine)

QUERY TIME:   query  =  text / video / text+video
                 │
                 ▼
 ┌───────── STAGE 1 · RETRIEVAL   (embedding model + cosine — a "bi-encoder") ──────────┐
 │  query ──[ same embedding model ]──▶ query vector                                    │
 │      cosine similarity  vs  ALL 245 document vectors  ──▶  rank  ──▶  TOP-3           │
 │      • searches the WHOLE database   • fast (pre-indexed)   • coarse                  │
 └───────────────────────────────────┬──────────────────────────────────────────────────┘
                                      │  pass only the 3 shortlisted candidates
                                      ▼
 ┌──────── STAGE 2 · RERANK   (reranker model — a "cross-encoder") · OPTIONAL ──────────┐
 │  each (query, candidate) pair ──[ reranker: query+doc read TOGETHER ]──▶ score       │
 │             reorder the 3   ──▶   new TOP-1                                            │
 │      • re-scores ONLY the shortlist (never the DB)   • slow   • accurate              │
 └───────────────────────────────────────────────────────────────────────────────────────┘

   Table 1 (§4) = Stage 1 only    ·    Table 2 (§5) = Stage 1 + Stage 2

  BI-ENCODER  (retrieval / embedder)        CROSS-ENCODER  (rerank / reranker)
    query ─[embed]─▶ ●q                        query ┐
    doc   ─[embed]─▶ ●d                              ├─▶[ ONE model ]─▶ relevance score
              cosine(●q, ●d)                   doc  ┘
    embedded APART, compare vectors            read TOGETHER, judged directly
    → pre-index millions, fast, coarse         → per-pair, few candidates, accurate
```

The reranker is a **relevance re-checker on a 3-item shortlist**, not a retriever — it never searches
the database (that's the embedder's job).

### 3b · What the retrieved scenario is FOR — feeding Scenic-code generation

Retrieval seeds **Scenic-code generation**. Two retrievals exist:

- **Level 1 — Scenario retrieval (what this document ablates).** Embedding = description(text) + BEV
video; **scenic code is NOT embedded.** Milvus stores `embedding + metadata + scenario_id`.
Retrieval → top-1 `base_scenario_id`; the scenic code is attached by a **folder join**, not by
similarity — `find_scenic_code_with_scenario_id(id)` reads `data/scenarios/<id>/code.scenic`.
  ```
  data/scenarios/<id>/
     ├── new_description.txt   ← embedded (text)
     ├── BEV.mp4               ← embedded (video)
     └── code.scenic           ← NOT embedded; fetched by id after retrieval → the base template
  ```
- **Level 2 — Component-snippet retrieval (coder agent).** `ScenicCoderAgent.get_snippets(text, comp_type)` uses **all-MiniLM-L6-v2** to search the SEPARATE `scenario_components` collection by
component type (Ego / Adversarial / Spatial Relation / Requirement) → top-3 code snippets.

**Generation loop:**

```
query → Level-1 RAG: embed(description+video) → cosine → TOP-1 base scenario
      → load that folder's code.scenic (by id) → BASE scenic code
      ├─▶ run_simulation : base .scenic in CARLA → BEV video
      ├─▶ evaluate_with_vlm (critic) : score vs the query
      ├─▶ interpret : query/feedback → target DSL
      └─▶ adapt_code : coder edits base code toward target DSL (+ Level-2 snippets; debug if error)
          └── loop back to run_simulation until satisfied → output_best_scenario → human_review
```

**Link to A2.** The base code is chosen by text+video similarity, never code similarity → a better
retrieval means less adaptation → fewer CARLA loops → better final scenario. This is the causal chain
from retrieval quality (Table 1) to the A2 closed-loop result. **Limitation:** similar
description/video ≠ similar Scenic code — a high-similarity retrieval isn't guaranteed easy to adapt.

## 4 · A0 — Query-modality ablation (the backbone) · Table 1

**Motivation.** Which embedder, and which query modality, retrieves best — the paper's core
multimodal claim. One database per embedder (§0); vary the query. Dense, no reranker.

**Table 1 — Backbone (dense).** VL embedders use a text+video DB; text embedders use a text DB
(† = video via the zero-shot qwen3.6-plus caption bridge). One table, three query groups; best HIT@1
per group in **bold**.


| Embedding model              | dim  | DB         | HIT@1    | HIT@3 | MRR@3 | lat(s) | sim   |
| ---------------------------- | ---- | ---------- | -------- | ----- | ----- | ------ | ----- |
| **══ Query = text ══**       |      |            |          |       |       |        |       |
| Qwen3-VL-2B                  | 2048 | text+video | 36.3     | 57.6  | 0.449 | 0.030  | 0.833 |
| Qwen3-VL-8B                  | 4096 | text+video | 38.4     | 63.7  | 0.492 | 0.052  | 0.769 |
| gemini-emb-2                 | 3072 | text+video | 33.1     | 53.5  | 0.417 | 0.331  | 0.592 |
| bge-m3                       | 1024 | text       | 49.8     | 71.0  | 0.591 | 0.028  | 0.946 |
| Qwen3-Emb-0.6B               | 1024 | text       | 41.2     | 63.3  | 0.512 | 0.028  | 0.851 |
| Qwen3-Emb-4B                 | 2560 | text       | 47.8     | 72.7  | 0.588 | 0.046  | 0.846 |
| Qwen3-Emb-8B                 | 4096 | text       | **68.2** | 86.5  | 0.763 | 0.067  | 0.877 |
| **══ Query = video ══**      |      |            |          |       |       |        |       |
| Qwen3-VL-2B                  | 2048 | text+video | 11.4     | 19.2  | 0.148 | 0.339  | 0.753 |
| Qwen3-VL-8B                  | 4096 | text+video | **30.2** | 49.4  | 0.389 | 0.564  | 0.685 |
| gemini-emb-2                 | 3072 | text+video | 17.1     | 27.8  | 0.219 | 2.118  | 0.779 |
| bge-m3 †                     | 1024 | text       | 2.9      | 6.1   | 0.044 | 0.027  | 0.910 |
| Qwen3-Emb-0.6B †             | 1024 | text       | 1.6      | 6.5   | 0.037 | 0.027  | 0.851 |
| Qwen3-Emb-4B †               | 2560 | text       | 4.5      | 7.8   | 0.059 | 0.043  | 0.821 |
| Qwen3-Emb-8B †               | 4096 | text       | 2.9      | 8.6   | 0.054 | 0.059  | 0.811 |
| **══ Query = text+video ══** |      |            |          |       |       |        |       |
| Qwen3-VL-2B                  | 2048 | text+video | 75.9     | 89.0  | 0.820 | 0.348  | 0.898 |
| Qwen3-VL-8B                  | 4096 | text+video | 88.6     | 96.3  | 0.921 | 0.581  | 0.874 |
| gemini-emb-2                 | 3072 | text+video | **96.7** | 99.2  | 0.979 | 2.272  | 0.966 |
| bge-m3 †                     | 1024 | text       | 48.2     | 67.8  | 0.564 | 0.035  | 0.939 |
| Qwen3-Emb-0.6B †             | 1024 | text       | 18.0     | 34.7  | 0.249 | 0.029  | 0.862 |
| Qwen3-Emb-4B †               | 2560 | text       | 26.5     | 54.3  | 0.389 | 0.055  | 0.871 |
| Qwen3-Emb-8B †               | 4096 | text       | 50.6     | 75.5  | 0.620 | 0.086  | 0.850 |


Findings: **text query** — text encoders win (Qwen3-Emb-8B 68.2 > best VL 38.4); **video / text+video
query** — native VL wins decisively (native video 30.2 vs bridge ≤4.5; text+video 96.7 vs bridge
≤50.6) because captioning discards most of the video signal (§9). For the VL embedders fusion is
super-additive — text+video ≫ either channel alone (96.7 vs 33.1/17.1) — the core multimodal claim.
Dropped as artifacts (not real DBs): pure-video-DB "video-only" 98.8 (self-copy leak) and the
modality-matched text-only VL numbers 50.2/51.4/59.6.

## 5 · A1 rerank on/off · A3 DSL-vs-raw · C1 top_k



### A1 (the funnel) · Tables 2 and 4

Paired runs differing only in `enable_reranking`; dense retrieve top-3, then a reranker (Qwen3-VL-
Reranker) re-scores them. **HIT@3 is invariant by construction** (rerank only reorders the top-3) —
a harness sanity check — so only HIT@1/MRR move.

**Table 2 — Reranking across all query modalities (dense → +Reranker), mirroring Table 1's three
query groups.** VL embedders, joint text+video DB, top-3 → rerank top-3. Cells: HIT@1 / HIT@3 / MRR@3.
HIT@3 is invariant under reranking (only the top-3 order changes) — it equals each row's Table-1 dense
HIT@3 to the decimal, a built-in consistency check. All full bf16 via the two-pass method (see ¹).


| Query                           | VL embedder | dense               | + Reranker-2B       | + Reranker-8B            |
| ------------------------------- | ----------- | ------------------- | ------------------- | ------------------------ |
| **══ text-only ══**             |             |                     |                     |                          |
| text                            | Qwen3-VL-2B | 36.3 / 57.6 / 0.449 | 49.8 / 57.6 / 0.535 | 49.4 / 57.6 / 0.533      |
| text                            | Qwen3-VL-8B | 38.4 / 63.7 / 0.492 | 56.7 / 63.7 / 0.601 | 56.7 / 63.7 / 0.600      |
| text                            | gemini      | 33.1 / 53.5 / 0.417 | 46.9 / 53.5 / 0.501 | 45.7 / 53.5 / 0.492      |
| **══ video-only ══**            |             |                     |                     |                          |
| video                           | Qwen3-VL-2B | 11.4 / 19.2 / 0.148 | 4.9 / 19.2 / 0.109  | 6.1 / 19.2 / 0.118       |
| video                           | Qwen3-VL-8B | 30.2 / 49.4 / 0.389 | 17.6 / 49.4 / 0.306 | 16.7 / 49.4 / 0.305      |
| video                           | gemini      | 17.1 / 27.8 / 0.219 | 8.6 / 27.8 / 0.170  | 8.6 / 27.8 / 0.167       |
| **══ text+video (deployed) ══** |             |                     |                     |                          |
| text+video                      | Qwen3-VL-2B | 75.9 / 89.0 / 0.820 | 73.5 / 89.0 / 0.808 | **72.2 / 89.0 / 0.801**  |
| text+video                      | Qwen3-VL-8B | 88.6 / 96.3 / 0.921 | 78.0 / 96.3 / 0.865 | **75.5 / 96.3 / 0.852**¹ |
| text+video                      | gemini      | 96.7 / 99.2 / 0.979 | 84.1 / 99.2 / 0.909 | **80.8 / 99.2 / 0.893**¹ |


¹ **Full bf16, no quantization.** The 8B-embedder + 8B-reranker cell can't co-reside on the 32 GB card
(measured: 8B embedder 15.2 GiB + 8B reranker 16.3 GiB = **31.5 GiB of weights** > 31.36 GiB usable,
before any activations). But reranking only reorders the dense **top-3** shortlist and never needs the
embedder in memory, so *all* reranker cells are run **two-pass** (`eval/rerank_from_run.py`): (a) the
embedder's dense top-3 (already saved), (b) load the reranker *alone* and re-score. This is
*mathematically identical* to the co-resident pipeline (HIT@3 invariant; only top-3 order changes) —
**validated against live runs to the decimal** in three cells: 2B-emb+8B-rr 72.24, gemini+2B-rr 84.08,
2B-emb video-only+2B-rr 4.90 (all = their live values). A GGUF/llama.cpp or bitsandbytes path was
rejected (GGUF has no `qwen3vl` arch in transformers and would swap the whole backend; a quantized
reranker would carry an asterisk this method avoids).

**One rule spans the whole grid — reranking helps where dense is weak with headroom, and hurts the
cross-modal query.** By query modality:

- **text-only (HELPS, +13 to +18):** 2B 36.3→49.8, 8B 38.4→56.7, gemini 33.1→46.9. Dense text-only is
weak but GT usually sits *inside* the top-3 (high HIT@3, low HIT@1); the cross-encoder promotes it to
rank-1. Reranker size barely matters here (2B ≈ 8B).
- **video-only (HURTS):** 2B 11.4→4.9/6.1, 8B 30.2→17.6/16.7, gemini 17.1→8.6/8.6. The reranker cannot
discriminate near-duplicate BEV clips (self-vs-other pair test 0.785 vs 0.730), so it *demotes* the
correct video.
- **text+video — the deployed mode (HURTS, and a BIGGER reranker hurts MORE):** 2B 75.9→73.5→72.2 (−3.7),
8B 88.6→78.0→75.5 (−13.1), gemini 96.7→84.1→80.8 (−15.9). Dense is already strong; reranking only
disturbs it, and scaling the cross-encoder 2B→8B amplifies the damage — the stronger the embedder,
the larger the hurt.

Same "helps weak, hurts strong" rule as the hybrid mechanism (§6). **Design conclusion: drop the
reranker for the deployed multimodal (text+video) pipeline** (Stage-1 dense alone) — confirmed across
all three embedders and both reranker sizes. (The text-only rerank *gain* is real but irrelevant to
deployment, which uses the text+video query; it is reported for completeness / the mechanism story.)

**Table 2t — Text-encoder baselines × text reranker (symmetry).** The Table-2 grid reranks the
*multimodal* embedders with the VL cross-encoder; for completeness we also rerank the **text-encoder
baselines** with the *matched* **text** cross-encoder (Qwen3-Reranker-0.6B/4B/8B) on the text-only query
(their native modality — the fair, leak-free headline, §7). Same two-pass method, full bf16; HIT@3 is
invariant (= each embedder's Table-1 dense HIT@3). Cells: HIT@1 / HIT@3 / MRR@3.

| Embedder (text-only) | dense               | + Reranker-0.6B     | + Reranker-4B       | + Reranker-8B           |
| -------------------- | ------------------- | ------------------- | ------------------- | ----------------------- |
| bge-m3               | 49.8 / 71.0 / 0.591 | 58.4 / 71.0 / 0.644 | 53.1 / 71.0 / 0.612 | 55.1 / 71.0 / 0.627     |
| Qwen3-Emb-0.6B       | 41.2 / 63.3 / 0.512 | 50.2 / 63.3 / 0.565 | 50.6 / 63.3 / 0.563 | 49.0 / 63.3 / 0.559     |
| Qwen3-Emb-4B         | 47.8 / 72.7 / 0.588 | 56.7 / 72.7 / 0.640 | 53.9 / 72.7 / 0.622 | 56.3 / 72.7 / 0.640     |
| Qwen3-Emb-8B         | 68.2 / 86.5 / 0.763 | 66.1 / 86.5 / 0.757 | 65.3 / 86.5 / 0.747 | **70.2 / 86.5 / 0.779** |

**Same rule on the text side — reranking helps the weak, is flat on the strong.** The three weaker text
encoders (dense 41–50; GT often sits in the top-3 but not rank-1) gain **+3 to +9** from any text
reranker; the strong **Qwen3-Emb-8B** (68.2, little headroom) is essentially flat and only the
*matched-capacity* 8B reranker nudges it up (+2.0 → **70.2**, the best fair/leak-free retrieval number
in the paper — though within the ±5–6 pt CI of dense). The reranker-size effect is **non-monotone** for
the weak encoders (the 0.6B reranker is often as good as the 8B): reranking a weak dense list mostly
re-surfaces a GT already present, which does not need a large cross-encoder. **Baseline completeness
only** — the deployed system is the VL pipeline (text+video query), so no reranker ships regardless.

**Table 4 — Reranker capacity: 2B vs 8B reranker** (does a bigger reranker fix the video hurt?).
All embedders, all at full bf16 (the 8B-embedder + gemini 8B-reranker cells via the two-pass method —
see ¹). Cells: HIT@1 / HIT@3 / MRR@3.


| Cell                    | dense                   | + 2B reranker           | + 8B reranker           |
| ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| 2B-emb · text-video     | 75.9 / 89.0 / 0.820     | 73.5 / 89.0 / 0.808     | 72.2 / 89.0 / 0.801     |
| 2B-emb · video-only     | 11.4 / 19.2 / 0.148     | 4.9 / 19.2 / 0.109      | 6.1 / 19.2 / 0.118      |
| **8B-emb · text-video** | **88.6 / 96.3 / 0.921** | **78.0 / 96.3 / 0.865** | **75.5 / 96.3 / 0.852** |
| **gemini · text-video** | **96.7 / 99.2 / 0.979** | **84.1 / 99.2 / 0.909** | **80.8 / 99.2 / 0.893** |


**Answer: capacity is decisively not the issue — a bigger reranker hurts more.** On every embedder the
8B reranker drops HIT@1 *further* than the 2B (2B-emb 72.2<73.5; 8B-emb 75.5<78.0; gemini 80.8<84.1),
and never approaches dense. The degradation is fundamental to applying a text-centric cross-encoder to
a joint text+video ranking, and scaling the cross-encoder amplifies it. (8B-reranker download hung 4×
then finished on the retry loop; co-resident-OOM cells filled at full precision by
`eval/rerank_from_run.py`, validated twice against live runs.)

### A3 (query representation)

`--query-repr {raw,dsl}`; `dsl` runs the production `interpreter.generate_dsl_from_user_query` path
exactly as `workflow.embed_query`. Result: **−7.8 text-only, +4.3 text-video** — the symmetric
VLM→DSL→flatten representation pays off precisely in the multimodal deployment condition (the
interpreter grounds the query in the video). (Two queries error in the VLM step → text-video-DSL is
243/245.)

### C1 (funnel width)

`top_k ∈ {5,10} × dense/rerank`; appendix figure justifying top_k=3; also produces the depth-10
ranking used by RRF (§6).

## 6 · Retrieval MECHANISM baselines — dense vs lexical (B1) vs hybrid (B1b) vs fusion

**Motivation.** Pre-empt "why not lexical / hybrid?". Measured, not asserted — `flatten_dsl_to_text`
is templated, keyword-heavy text where BM25 is a priori competitive. Mechanism ≠ embedder: Table 1
compares embedders (all dense); here we hold the text fixed and vary the *mechanism*.

**How.** `eval/bm25_baseline.py` (rank_bm25 over the same corpus, caption queries) and
`eval/rrf_fusion.py` (offline Σ 1/(60+rank) RRF fusion). No new retrieval infra.

**BM25 alone** — standalone lexical retriever (no embedder; word-overlap). Full metrics: **HIT@1
55.5, HIT@3 77.1, MRR@3 0.654, latency 0.002 s.** Text-only and embedder-independent → the constant
`BM25` reference and the fusion input for every hybrid.

**Table 1r — dense vs + BM25 hybrid (RRF), per model.** Cells: HIT@1 / HIT@3 / MRR@3; Δ = HIT@1(hyb−dense).


| Embedder                                                 | dense                   | + BM25 hybrid           | Δ HIT@1 |
| -------------------------------------------------------- | ----------------------- | ----------------------- | ------- |
| *Query = text*                                           |                         |                         |         |
| Qwen3-VL-2B                                              | 36.3 / 57.6 / 0.449     | 46.9 / 69.8 / 0.609     | +10.6   |
| Qwen3-VL-8B                                              | 38.4 / 63.7 / 0.492     | 49.8 / 73.5 / 0.633     | +11.4   |
| gemini                                                   | 33.1 / 53.5 / 0.417     | 44.5 / 71.4 / 0.597     | +11.4   |
| bge-m3                                                   | 49.8 / 71.0 / 0.591     | 57.1 / 78.8 / 0.688     | +7.3    |
| Qwen3-Emb-0.6B                                           | 41.2 / 63.3 / 0.512     | 49.0 / 78.8 / 0.645     | +7.8    |
| Qwen3-Emb-4B                                             | 47.8 / 72.7 / 0.588     | 60.0 / 79.6 / 0.700     | +12.2   |
| Qwen3-Emb-8B                                             | 68.2 / 86.5 / 0.763     | **69.4 / 87.3 / 0.786** | +1.2    |
| *Query = text+video* (VL native; † text-embedder bridge) |                         |                         |         |
| Qwen3-VL-2B                                              | 75.9 / 89.0 / 0.820     | 71.4 / 89.0 / 0.804     | −4.5    |
| Qwen3-VL-8B                                              | 88.6 / 96.3 / 0.921     | 78.4 / 96.3 / 0.867     | −10.2   |
| gemini                                                   | **96.7 / 99.2 / 0.979** | 87.3 / 98.8 / 0.929     | −9.4    |
| bge-m3 †                                                 | 48.2 / 67.8 / 0.564     | 55.5 / 76.7 / 0.673     | +7.3    |
| Qwen3-Emb-0.6B †                                         | 18.0 / 34.7 / 0.249     | 29.0 / 69.0 / 0.501     | +11.0   |
| Qwen3-Emb-4B †                                           | 26.5 / 54.3 / 0.389     | 44.5 / 74.7 / 0.608     | +18.0   |
| Qwen3-Emb-8B †                                           | 50.6 / 75.5 / 0.620     | 61.2 / 86.1 / 0.738     | +10.6   |


**One coherent rule (both query types, both embedder types):** hybrid **helps a weak dense retriever**
(+1 to +18) and **hurts a strong one** (−4.5 to −10.2) — RRF pulls a strong dense down toward the
weaker BM25. Split is purely dense strength: weak = text queries + text-embedder bridges (18–68
dense); strong = native VL text+video (76–97). BM25 (55.5) beats most text/bridge dense retrievers
but is far below multimodal dense (96.7). **Deployed = pure dense; lexical fusion only rescues a
weak embedder.**

**Table 1f — Cross-modal FUSION mechanism (VL, text+video): early/joint vs late.** early/joint = the
VL embedder encodes text+video into ONE vector (what we use); late = embed text & video separately,
RRF-fuse the rankings. Cells: HIT@1 / HIT@3 / MRR@3.


| VL embedder | text-only           | video-only          | late fusion (RRF)   | **early/joint**         |
| ----------- | ------------------- | ------------------- | ------------------- | ----------------------- |
| Qwen3-VL-2B | 36.3 / 57.6 / 0.449 | 11.4 / 19.2 / 0.148 | 39.2 / 51.4 / 0.472 | **75.9 / 89.0 / 0.820** |
| Qwen3-VL-8B | 38.4 / 63.7 / 0.492 | 30.2 / 49.4 / 0.389 | 50.6 / 73.1 / 0.636 | **88.6 / 96.3 / 0.921** |
| gemini      | 33.1 / 53.5 / 0.417 | 17.1 / 27.8 / 0.219 | 39.2 / 58.0 / 0.497 | **96.7 / 99.2 / 0.979** |


**Joint ≫ late fusion by +36 to +57 pts** — you cannot fake multimodal retrieval by fusing unimodal
rankings; the joint encoder captures text↔video interactions RRF cannot. Justifies using a true
multimodal embedder, not fuse-then-search.

**Mechanism coverage per query modality** (why some cells are structurally empty):


| Query          | dense | lexical (BM25) | hybrid      | fusion             | late-interaction (multi-vector) |
| -------------- | ----- | -------------- | ----------- | ------------------ | ------------------------------- |
| text           | ✅     | ✅              | ✅           | ✗ n/a (1 modality) | future work                     |
| text+video     | ✅     | ✅ (text arm)   | ✅           | ✅ (early vs late)  | future work                     |
| **video-only** | ✅     | ✗ undefined    | ✗ undefined | ✗ n/a (1 modality) | future work                     |


**Video-only has ONLY the dense mechanism** with single-vector embedders — BM25/hybrid need words a
video lacks, fusion needs ≥2 modalities. The sole alternative for pure video is **multi-vector late
interaction (Video-ColBERT / ColQwen-video)** — a *different encoder*, not a config change; left to
future work (arXiv:2503.19009, 2407.01449). So "video-only = dense" is a structural fact, not a gap.

## 6b · Caption-bridge — can a text encoder handle video by captioning?

**Motivation.** Do you need a multimodal embedder, or does "caption everything → text encoder"
suffice? The bridge converts video → qwen3.6-plus caption (§0b), so a text encoder fills the video &
text+video columns (rows marked † in Table 1, §4).

**Result (complete):** **native multimodal ≫ caption bridge** for anything involving video. Video-
caption alone is near-chance (text encoders 1.6–4.5 HIT@1 vs native VL 11–30); the "two texts"
combined caption does not beat text-caption alone (bge-m3 48.2 ≈ its 49.8 text-only), while native VL
text+video reaches 88.6 / 96.7. Captioning throws away most of the video signal (explained by the
modality gap, §9). Strong evidence a true multimodal embedder is necessary.

## 7 · Corrections (kept honest, not buried)

1. **"image-16 is leaky" — WRONG, retracted.** I spot-checked a June-1 leaky file and wrongly
  assumed it was image-16's source. image-16 used independent captions (§2). The re-measurement's
   value is a single documented protocol (§0); it surfaced the real issue (per-cell doc-index
   modality was under-specified).
2. **"gemini-embedding-2 ignores video" — WRONG, retracted.** Our provider passed `[text, video]` as
  a LIST → `embed_content` treated it as a batch and we kept `embeddings[0]` (text). Fixed to a
   single `types.Content(parts=[...])` → joint embedding (cos 0.63/0.75 to text/video). Re-ingested;
   **gemini text-video jumped 66.9 → 96.7**.
3. **"video-only symmetric = 98.8" is a leak, not a result.** Rebuilding a video-only DB makes query
  video ≡ doc video (same file) → self-copy. Dropped; §0.
4. **"hybrid can't do text+video" — WRONG, retracted.** Hybrid fuses *rankings*, so multimodal-dense +
  text-BM25 is valid; Table 1r includes it.
5. **Ingestion completeness.** First gemini ingest 243/245 (two transient 503s → guaranteed
  self-miss). Added retry-with-backoff + a `GEMINI_ROWS 245` assertion; all collections = 245.



## 8 · Harness integrity fixes (F0–F9)


| Fix | Defect                                                          | Change                                                                       |
| --- | --------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| F0  | `FOLDER_PATH` hardcoded to the server                           | repo-relative + `--folder`                                                   |
| F1  | eval bypassed the production query path (`dsl=query.text`)      | `--query-repr {raw,dsl}`; `dsl` = A3                                         |
| F2  | rank position invisible                                         | MRR@3 from the ordered id list (nDCG omitted: redundant under single GT)     |
| F3  | runs not self-identifying                                       | stamp mode/source/repr/rerank/embedder/top_k; structured `runs/<embedder>/…` |
| F9  | reranker silently scores "NULL" on vision-preprocessing failure | media paths auto-absolutized; loud error                                     |
| —   | missing caption crashed as error row                            | clean skip                                                                   |




## 9 · Analysis — the modality gap (arXiv:2203.02053) · Table 3

**Motivation.** Explain *why* the retrieval numbers come out as they do (analysis AAAI rewards). For
each VL embedder, embed each scenario as text-only, video-only, and joint; measure how far the
modality subspaces sit apart (L2-normalized embeddings).

**Table 3 — Modality gap.**


| Embedder           | matched cos(t_i,v_i) | within-text | within-video | centroid gap | text–joint / video–joint |
| ------------------ | -------------------- | ----------- | ------------ | ------------ | ------------------------ |
| Qwen3-VL-2B        | 0.51                 | 0.80        | 0.82         | 0.80         | 0.91 / 0.69              |
| Qwen3-VL-8B        | 0.40                 | 0.72        | 0.69         | 0.82         | 0.86 / 0.65              |
| gemini-embedding-2 | 0.43                 | 0.83        | 0.84         | 0.93         | 0.63 / 0.75              |


**Key result:** matched cross-modal cosine (0.40–0.51) is *lower* than within-modal (0.69–0.84) — a
scenario's own text and video embeddings are less alike than two random scenarios' texts. This one
number explains the tables: **within-video ≈ 0.7–0.84** means all BEV clips look nearly identical →
video-only and bridge-video retrieval are near-chance; the **0.8–0.93 centroid gap** means a text
query lands off the joint document vectors → text-query underperforms; joint wins because each
modality adds a component the other lacks (text–joint 0.63–0.91, video–joint 0.65–0.75 — for the
Qwen VL embedders video is the more distinct add, for gemini it is text). Turns empirical scores into
an explained mechanism.

## Statistical protocol

|corpus| = |queries| = 245; random floor HIT@1 = 1/245 ≈ 0.41 %. 95% CI on HIT@1 at N=245 ≈ ±5–6 pts;
paired arms (rerank on/off, DSL/raw) warrant McNemar before claiming small deltas. Saturation guard:
fair-caption HIT@1 ≤ 96.7 (gemini text+video) — the 85.7 self-retrieval figure is reported only as an
upper bound, never as a system result.

## Current status (2026-07-06)

**DONE:** env/data; provenance & protocol; **Table 1** backbone (7 embedders × 3 query modalities);
**Table 1r** mechanism (dense/BM25/hybrid, per model, both query types); **Table 1f** early-vs-late
fusion; mechanism-coverage matrix; **Table 2** rerank with the **2B and 8B** rerankers (complete
+Reranker-8B column, all 3 embedders); **Table 4** reranker-capacity (2B/8B/gemini embedders × 2B/8B
reranker → capacity is not the issue, a bigger reranker hurts more); A3 DSL-vs-raw; C1 top_k;
caption-bridge (§6b); **Table 3** modality-gap (2B/8B/gemini); captions 245/245; all collections = 245.

**8B reranker — COMPLETE (2026-07-06):** the HF download that hung 4× finished on the retry loop (4/4
shards, 16.5 GB). All 8B-reranker cells filled at **full bf16, no gaps**: the co-resident 8B-emb×8B-rr
OOM (31.5 GiB weights > 31.36 usable) was resolved by a **two-pass offline rerank**
(`eval/rerank_from_run.py`) — reuse saved dense top-3, load reranker alone, re-score; identical to the
co-resident pipeline, **validated twice against live runs to the decimal**. Result across all
embedders: reranking hurts, the 8B reranker hurts *more* than the 2B (2B-emb 75.9→73.5→72.2; 8B-emb
88.6→78.0→75.5; gemini 96.7→84.1→80.8). **The whole retrieval study is closed — no remaining table
gaps.**

**A2 closed-loop — DESCOPED (2026-07-06).** Each dataset scenario is a mutually-aligned, curated
triple *(description, BEV video,* `code.scenic`*)* with a known-valid Scenic seed. So a correct retrieval
seeds generation with the ground-truth-aligned Scenic code by construction, and retrieval **HIT@1
already = the fraction of queries seeded with the correct, validated Scenic seed** — no CARLA run is
needed to re-verify a seed the dataset guarantees. Tables 1–4 are therefore the complete, self-contained
evaluation. Claims are scoped retrieval-centric ("multimodal RAG fetches the correct, validated base
scenario + Scenic seed"); the agentic generate→simulate→adapt loop is presented as deployment context,
optionally with 1–2 qualitative end-to-end examples (no quantitative CARLA sweep).

**PENDING (optional, non-blocking):** aggregator dedup + check_results gate (cosmetic — canonical
values are hand-verified above); rotate the two API keys shared in chat.