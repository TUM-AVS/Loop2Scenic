"""SAMPLE codeICL decomposition: decompose a few missing scenarios' code.scenic into the same
5-type component-snippet format as the existing 922, for quality review before the full 100.

Uses qwen3.6-plus (temp 0) with a few-shot exemplar drawn from the existing library so the output
matches the established typing/granularity/description style.
"""
import json, sys, collections, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.services.llm.providers.QwenAPIModel import QwenAPIModel

LIB = "data/raw_snippets/recovered_scenario_components_with_subject.json"
TYPES = ["Ego", "Adversarial", "Spatial Relation", "Requirement and restrictions", "Scenario"]
TARGETS = sys.argv[1:] or ["CARLA_Leaderboard_1", "NHTSA_Crash_1", "CARLA_Leaderboard_5"]


def load_existing():
    d = json.load(open(LIB))
    by_sid = collections.defaultdict(dict)
    for x in d:
        by_sid[str(x["scenario_id"])][x["component_type"]] = x
    return d, by_sid


def pick_exemplar(by_sid):
    # a scenario that has all 5 types AND whose code.scenic exists locally
    for sid, comps in by_sid.items():
        if set(comps) >= set(TYPES) and Path(f"data/scenarios/{sid}/code.scenic").exists():
            return sid, comps
    raise SystemExit("no exemplar with all 5 types found")


def build_prompt(ex_code, ex_comps, target_code):
    ex_out = [{"component_type": t, "code": ex_comps[t]["code"], "description": ex_comps[t]["description"]}
              for t in TYPES]
    sys_msg = (
        "You decompose a Scenic 3 autonomous-driving scenario program into reusable, typed component "
        "snippets. Use EXACTLY these five component_type labels: "
        "Ego, Adversarial, Spatial Relation, Requirement and restrictions, Scenario.\n"
        "- Ego: the ego vehicle's params, behavior, and instantiation.\n"
        "- Adversarial: adversary/pedestrian actors' params, behaviors, and instantiation "
        "(omit this component if the scenario has no adversary).\n"
        "- Spatial Relation: road/lane/intersection selection, maneuvers, trajectories, spawn points.\n"
        "- Requirement and restrictions: require/terminate clauses and their distance params.\n"
        "- Scenario: the header + overall assembly (description, param map/model/MODEL, and the top-level glue).\n"
        "For each component, `code` is the verbatim lines from the program that belong to it, and "
        "`description` is a one-sentence natural-language summary in the same style as the example. "
        "Return ONLY a JSON array of {component_type, code, description}."
    )
    user_msg = (
        f"EXAMPLE INPUT (code.scenic):\n```\n{ex_code}\n```\n\n"
        f"EXAMPLE OUTPUT:\n{json.dumps(ex_out, ensure_ascii=False, indent=1)}\n\n"
        f"NOW DECOMPOSE THIS scenario:\n```\n{target_code}\n```\n\nReturn the JSON array only."
    )
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def main():
    _, by_sid = load_existing()
    ex_sid, ex_comps = pick_exemplar(by_sid)
    ex_code = Path(f"data/scenarios/{ex_sid}/code.scenic").read_text()
    print(f"[exemplar] scenario {ex_sid} (all 5 types)\n", flush=True)

    llm = QwenAPIModel(model="qwen3.6-plus", temperature=0, max_tokens=4096)
    out = {}
    for tgt in TARGETS:
        code = Path(f"data/scenarios/{tgt}/code.scenic").read_text()
        msgs = build_prompt(ex_code, ex_comps, code)
        raw = llm.chat(msgs).strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1].lstrip("json").strip() if "```" in raw else raw
        try:
            comps = json.loads(raw)
        except Exception as e:
            print(f"[{tgt}] JSON parse failed: {e}\nRAW:\n{raw[:800]}\n"); continue
        out[tgt] = comps
        print(f"===== {tgt}: {len(comps)} components =====", flush=True)
        for c in comps:
            print(f"  [{c.get('component_type')}] {c.get('description','')[:150]}")
        print()
    Path("data/raw_snippets/_sample_new.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote data/raw_snippets/_sample_new.json")


if __name__ == "__main__":
    main()
