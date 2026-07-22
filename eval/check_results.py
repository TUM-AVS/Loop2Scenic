"""Sanity gate for the aggregated retrieval results (summary.csv).

Fails loudly on: missing/NaN metrics, out-of-range values, HIT@1 > HIT@3, MRR@3 outside
[HIT@1, HIT@3] (fractions), leftover duplicate variants, or an empty table. Warns (does not fail)
on suspicious all-perfect rows. Exit code 0 = PASS, 1 = FAIL.

Usage: python eval/check_results.py [--summary eval/results/rag/summaries/summary.csv]
"""
import argparse
import csv
import math
import sys
from pathlib import Path

TOL = 0.5  # percentage-point tolerance for rounding when comparing MRR bounds


def _num(v):
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="eval/results/rag/summaries/summary.csv")
    args = ap.parse_args()

    p = Path(args.summary)
    if not p.exists():
        print(f"[FAIL] summary not found: {p}")
        return 1
    rows = list(csv.DictReader(p.open()))

    fails, warns = [], []
    if not rows:
        print("[FAIL] summary.csv is empty")
        return 1

    seen = {}
    for i, r in enumerate(rows):
        tag = f"row {i+1} [{r.get('embedder')}·{r.get('mode')}·{r.get('reranker','?')}·{r.get('query_text_source')}]"
        h1, h3 = _num(r.get("hit@1")), _num(r.get("hit@3"))
        mrr, ne = _num(r.get("mrr@3")), _num(r.get("no_error_rate"))

        # 1. present & numeric
        if None in (h1, h3, mrr, ne):
            fails.append(f"{tag}: missing/NaN metric (hit@1={r.get('hit@1')}, hit@3={r.get('hit@3')}, "
                         f"mrr@3={r.get('mrr@3')}, no_error={r.get('no_error_rate')})")
            continue
        # 2. ranges
        for nm, val, hi in (("hit@1", h1, 100), ("hit@3", h3, 100), ("no_error_rate", ne, 100), ("mrr@3", mrr, 1)):
            if not (0 <= val <= hi):
                fails.append(f"{tag}: {nm}={val} out of [0,{hi}]")
        # 3. HIT@1 ⊆ HIT@3
        if h1 > h3 + TOL:
            fails.append(f"{tag}: HIT@1 {h1} > HIT@3 {h3} (rank-1 hits cannot exceed top-3 hits)")
        # 4. MRR@3 ∈ [HIT@1, HIT@3] as fractions
        if not (h1 / 100 - TOL / 100 <= mrr <= h3 / 100 + TOL / 100):
            fails.append(f"{tag}: MRR@3 {mrr:.3f} outside [HIT@1/100={h1/100:.3f}, HIT@3/100={h3/100:.3f}]")
        # 5. all-perfect → warn
        if h1 >= 100:
            warns.append(f"{tag}: HIT@1=100 (verify this is not a leak/self-retrieval)")
        # 6. duplicate variants (should be none post-dedup)
        key = (r.get("embedder"), r.get("mode"), r.get("query_text_source"),
               r.get("query_repr"), r.get("reranker"), r.get("top_k"))
        if key in seen:
            fails.append(f"{tag}: duplicate variant of row {seen[key]+1} (dedup did not collapse it)")
        seen[key] = i

    print(f"Checked {len(rows)} rows.")
    for w in warns:
        print(f"  [WARN] {w}")
    if fails:
        print(f"[FAIL] {len(fails)} problem(s):")
        for f in fails:
            print(f"  [FAIL] {f}")
        return 1
    print(f"[PASS] all {len(rows)} rows sane ({len(warns)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
