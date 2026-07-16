"""Full codeICL 145->245 decomposition (into a SEPARATE file/collection, non-destructive).

Decomposes the 100 missing scenarios' code.scenic into the same 5-type snippet format as the existing
922, concurrently (qwen3.6-plus, temp 0), verbatim-validates each component, then writes a 245-covering
library to a NEW file: data/raw_snippets/recovered_scenario_components_245.json
(existing 922 kept untouched; the 245 collection is built separately so it is cleanly deletable).
"""
import json, os, sys, collections, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.services.llm.providers.QwenAPIModel import QwenAPIModel

LIB = "data/raw_snippets/recovered_scenario_components_with_subject.json"
OUT = "data/raw_snippets/recovered_scenario_components_245.json"
TYPES = ["Ego", "Adversarial", "Spatial Relation", "Requirement and restrictions", "Scenario"]
MAX_WORKERS = 8


def load_existing():
    d = json.load(open(LIB))
    by_sid = collections.defaultdict(dict)
    for x in d:
        by_sid[str(x["scenario_id"])][x["component_type"]] = x
    return d, by_sid


def build_msgs(ex_code, ex_comps, target_code):
    ex_out = [{"component_type": t, "code": ex_comps[t]["code"], "description": ex_comps[t]["description"]}
              for t in TYPES if t in ex_comps]
    sys_msg = (
        "You decompose a Scenic 3 autonomous-driving scenario program into reusable, typed component "
        "snippets. Use EXACTLY these five component_type labels: "
        "Ego, Adversarial, Spatial Relation, Requirement and restrictions, Scenario.\n"
        "- Ego: the ego vehicle's params, behavior, and instantiation.\n"
        "- Adversarial: adversary/pedestrian actors' params, behaviors, instantiation "
        "(OMIT this component entirely if the scenario has no adversary).\n"
        "- Spatial Relation: road/lane/intersection selection, maneuvers, trajectories, spawn points.\n"
        "- Requirement and restrictions: require/terminate clauses and their distance params.\n"
        "- Scenario: the header + overall assembly (description, param map/model/MODEL, top-level glue).\n"
        "`code` must be VERBATIM lines copied from the program (never invent code); `description` is a "
        "one-sentence natural-language summary in the example's style. Return ONLY a JSON array of "
        "{component_type, code, description}."
    )
    user_msg = (f"EXAMPLE INPUT (code.scenic):\n```\n{ex_code}\n```\n\n"
                f"EXAMPLE OUTPUT:\n{json.dumps(ex_out, ensure_ascii=False, indent=1)}\n\n"
                f"NOW DECOMPOSE THIS scenario:\n```\n{target_code}\n```\n\nReturn the JSON array only.")
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def parse_json(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(json)?", "", raw).rsplit("```", 1)[0].strip()
    return json.loads(raw)


def verbatim_ratio(code, src):
    lines = [l.strip() for l in code.splitlines() if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return 0.0
    return sum(1 for l in lines if l in src) / len(lines)


def decompose_one(llm, ex_code, ex_comps, sid):
    src = Path(f"data/scenarios/{sid}/code.scenic").read_text()
    for attempt in range(2):
        try:
            raw = llm.chat(build_msgs(ex_code, ex_comps, src), temperature=0, max_tokens=4096)
            comps = parse_json(raw)
            out = []
            for c in comps:
                ct, code, desc = c.get("component_type"), (c.get("code") or "").strip(), (c.get("description") or "").strip()
                if ct not in TYPES or not code or not desc:
                    continue
                vr = verbatim_ratio(code, src)
                out.append({"scenario_id": sid, "component_type": ct, "code": code,
                            "description": desc, "_verbatim": round(vr, 2)})
            if out:
                return sid, out, None
        except Exception as e:
            err = str(e)
    return sid, [], err if 'err' in dir() else "no components"


def main():
    existing, by_sid = load_existing()
    covered = set(by_sid)
    folders = {p for p in os.listdir("data/scenarios")
               if os.path.isdir(f"data/scenarios/{p}") and os.path.exists(f"data/scenarios/{p}/code.scenic")}
    missing = sorted(folders - covered)
    print(f"existing snippets: {len(existing)} from {len(covered)} scenarios | missing to decompose: {len(missing)}", flush=True)

    # exemplar with all 5 types + local code
    ex_sid = next(s for s, c in by_sid.items() if set(c) >= set(TYPES) and Path(f"data/scenarios/{s}/code.scenic").exists())
    ex_code = Path(f"data/scenarios/{ex_sid}/code.scenic").read_text()
    ex_comps = by_sid[ex_sid]
    print(f"exemplar scenario: {ex_sid}", flush=True)

    llm = QwenAPIModel(model="qwen3.6-plus", temperature=0, max_tokens=4096)
    new_snippets, failures, low_verbatim = [], [], 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(decompose_one, llm, ex_code, ex_comps, s): s for s in missing}
        done = 0
        for fut in as_completed(futs):
            sid, comps, err = fut.result()
            done += 1
            if err or not comps:
                failures.append((sid, err))
                print(f"  [{done}/{len(missing)}] FAIL {sid}: {err}", flush=True)
                continue
            new_snippets.extend(comps)
            low = [c for c in comps if c["_verbatim"] < 0.5]
            low_verbatim += len(low)
            print(f"  [{done}/{len(missing)}] {sid}: {len(comps)} comps ({[c['component_type'][:3] for c in comps]})"
                  + (f"  ⚠{len(low)} low-verbatim" if low else ""), flush=True)

    # assign fresh int ids after the max existing id; strip helper field
    max_id = max(int(x["id"]) for x in existing)
    for i, c in enumerate(new_snippets):
        c["id"] = max_id + 1 + i
        c.pop("_verbatim", None)
        c = {k: c[k] for k in ("code", "id", "scenario_id", "component_type", "description")}
        new_snippets[i] = c

    full = existing + new_snippets
    Path(OUT).write_text(json.dumps(full, ensure_ascii=False, indent=1))
    tdist = collections.Counter(x["component_type"] for x in new_snippets)
    print(f"\n=== DONE ===", flush=True)
    print(f"new snippets: {len(new_snippets)} from {len(missing)-len(failures)} scenarios | failures: {len(failures)} | low-verbatim comps: {low_verbatim}", flush=True)
    print(f"new type dist: {dict(tdist)}", flush=True)
    print(f"TOTAL 245 library: {len(full)} snippets from {len(covered | (folders - covered - {f[0] for f in failures}))} scenarios", flush=True)
    print(f"wrote {OUT}", flush=True)
    if failures:
        print("failed scenarios:", [f[0] for f in failures], flush=True)
    print("DECOMPOSE_FULL_DONE", flush=True)


if __name__ == "__main__":
    main()
