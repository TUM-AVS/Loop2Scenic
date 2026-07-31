#!/usr/bin/env python3
"""Build codegen ablation prompt groups under ``src/prompt/gen_eval/``.

Ablation factors (from current component_generator_*.txt):
  CP  = contextual prompting (hard constraints + Scenic API/syntax docs + ready_components)
  CoT = chain-of-thought / reasoning plan
  ICL = illustrative few-shot examples
  Snippets = retrieved code snippets ({reference_components})

Groups:
  g1 zeroshot          — none
  g2 CP
  g3 CP + CoT
  g4 CP + ICL
  g5 CP + ICL + CoT
  g6 CP + snippets
  g7 CP + CoT + snippets
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_DIR = ROOT / "src" / "prompt"
OUT_ROOT = PROMPT_DIR / "gen_eval"

COMPONENTS = [
    "component_generator_ego",
    "component_generator_adv",
    "component_generator_spatial",
    "component_generator_requirement",
    "component_generator_road_side_structure",
    "component_generator_temporary_modification",
]

GROUPS = {
    "g1_zeroshot": {"cp": False, "cot": False, "icl": False, "snippets": False},
    "g2_cp": {"cp": True, "cot": False, "icl": False, "snippets": False},
    "g3_cp_cot": {"cp": True, "cot": True, "icl": False, "snippets": False},
    "g4_cp_icl": {"cp": True, "cot": False, "icl": True, "snippets": False},
    "g5_cp_icl_cot": {"cp": True, "cot": True, "icl": True, "snippets": False},
    "g6_cp_snippets": {"cp": True, "cot": False, "icl": False, "snippets": True},
    "g7_cp_cot_snippets": {"cp": True, "cot": True, "icl": False, "snippets": True},
}


def _find_line(lines: list[str], pattern: str, start: int = 0) -> int:
    cre = re.compile(pattern, re.I)
    for i in range(start, len(lines)):
        if cre.search(lines[i]):
            return i
    return -1


def _slice(lines: list[str], start: int, end: int) -> str:
    if start < 0 or start >= len(lines):
        return ""
    if end < 0:
        end = len(lines)
    return "\n".join(lines[start:end]).strip()


def split_prompt(text: str) -> dict[str, str]:
    """Split a component_generator prompt into reusable blocks."""
    lines = text.splitlines()

    abs_i = _find_line(lines, r"^\*\*.*Absolute Output Constraint")
    hard_i = _find_line(lines, r"^\*\*.*Hard [Cc]onstraints?")
    cot_i = _find_line(
        lines,
        r"^\*\*.*(Reasoning\s*&\s*Implementation|Internal Reasoning Directive)",
    )
    # Domain/API context starts at Hard Constraints if present, else after CoT
    # (spatial has hierarchy/specifiers after CoT with no Hard Constraints header).
    syntax_i = _find_line(
        lines,
        r"^\*\*.*(Syntax of|Require Statements Syntax|Blueprint mapping|Spatial Relation Hierarchy)",
    )
    # For spatial, hierarchy comes after CoT — treat from hierarchy as CP docs.
    if hard_i < 0:
        hard_i = _find_line(lines, r"^\*\*Spatial Relation Hierarchy")
        if hard_i < 0:
            hard_i = syntax_i

    icl_i = _find_line(lines, r"^\*\*.*Illustrative Examples")
    inputs_i = _find_line(lines, r"^\*\*Inputs\*\*")
    output_i = _find_line(lines, r"^\*\*Output Format\*\*")

    if abs_i < 0 or inputs_i < 0 or output_i < 0:
        raise ValueError("Missing required markers (Absolute Output / Inputs / Output Format)")

    # Header: role/task through Absolute Output Constraint (stop before Hard/CoT/CP)
    header_end = abs_i + 1
    # include Absolute Output block until next major section
    next_after_abs = min(
        i for i in (hard_i, cot_i, icl_i, inputs_i) if i > abs_i
    )
    header = _slice(lines, 0, next_after_abs)

    # CoT block
    cot = ""
    if cot_i >= 0:
        cot_end_candidates = [i for i in (hard_i, syntax_i, icl_i, inputs_i) if i > cot_i]
        # If hard_i is before cot (ego/adv), cot ends at syntax/icl/inputs
        cot_end_candidates = [i for i in (syntax_i, icl_i, inputs_i, hard_i) if i > cot_i]
        # Prefer next section after cot among syntax, icl, inputs, or hard if hard after cot
        after = [i for i in (syntax_i, hard_i, icl_i, inputs_i) if i > cot_i]
        cot_end = min(after) if after else inputs_i
        cot = _slice(lines, cot_i, cot_end)

    # CP docs: hard constraints + syntax/API (everything that is not header/cot/icl/inputs)
    # Collect from hard_i (or hierarchy) up to icl or inputs, excluding cot if nested.
    cp = ""
    if hard_i >= 0:
        cp_end = min(i for i in (icl_i, inputs_i) if i > hard_i) if any(
            i > hard_i for i in (icl_i, inputs_i)
        ) else inputs_i
        cp_lines = lines[hard_i:cp_end]
        # If CoT sits inside this range (spatial: cot before hard), already excluded.
        # If CoT is before hard (ego), fine. If somehow overlapping, drop cot lines.
        if cot_i >= hard_i and cot_i < cp_end:
            # rare: strip cot from cp
            cot_end = min(i for i in (syntax_i, icl_i, inputs_i) if i > cot_i)
            cp = _slice(lines, hard_i, cot_i)
            if cot_end < cp_end:
                cp = (cp + "\n\n" + _slice(lines, cot_end, cp_end)).strip()
        else:
            cp = "\n".join(cp_lines).strip()

    # If hard constraints are before CoT (ego/adv), CP also includes syntax after CoT
    if hard_i >= 0 and cot_i > hard_i and syntax_i > cot_i:
        cp_part1 = _slice(lines, hard_i, cot_i)
        cp_part2_end = icl_i if icl_i > syntax_i else inputs_i
        cp_part2 = _slice(lines, syntax_i, cp_part2_end)
        cp = (cp_part1 + "\n\n" + cp_part2).strip()

    icl = ""
    if icl_i >= 0:
        icl = _slice(lines, icl_i, inputs_i)

    output = _slice(lines, output_i, len(lines))

    return {
        "header": header,
        "cot": cot,
        "cp": cp,
        "icl": icl,
        "output": output,
    }


def build_inputs(flags: dict) -> str:
    parts = [
        "**Inputs**:",
        "User Requirements:",
        "{user_criteria}",
        "",
    ]
    if flags["cp"]:
        parts += [
            "Already Determined Components (context only, do not redefine):",
            "{ready_components}",
            "",
        ]
    if flags["snippets"]:
        parts += [
            "Retrieved reference code snippets:",
            "{reference_components}",
            "",
        ]
    # Always append format placeholders so ``str.format`` never KeyErrors.
    # Runtime fills them with "" when the corresponding factor is disabled.
    parts += [
        "{ready_components}" if not flags["cp"] else "",
        "{reference_components}" if not flags["snippets"] else "",
    ]
    return "\n".join(p for p in parts if p is not None).rstrip() + "\n"


def compose(blocks: dict[str, str], flags: dict) -> str:
    chunks = [blocks["header"]]
    if flags["cp"] and blocks["cp"]:
        chunks.append(blocks["cp"])
    if flags["cot"] and blocks["cot"]:
        chunks.append(blocks["cot"])
    if flags["icl"] and blocks["icl"]:
        chunks.append(blocks["icl"])
    chunks.append(build_inputs(flags))
    chunks.append(blocks["output"])
    text = "\n\n".join(c.strip() for c in chunks if c and c.strip()).strip() + "\n"
    # Ensure required format keys exist
    for key in ("user_criteria", "ready_components", "reference_components"):
        if "{" + key + "}" not in text:
            raise RuntimeError(f"Missing {{{key}}} placeholder in composed prompt")
    return text


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)

    manifest = [
        "Codegen ablation prompts (component generators)",
        "",
        "Factors:",
        "  CP       = contextual prompting (hard constraints + Scenic API/syntax + prior components)",
        "  CoT      = chain-of-thought / reasoning plan",
        "  ICL      = illustrative few-shot examples",
        "  snippets = retrieved code snippets",
        "",
        "Groups:",
        "  g1_zeroshot         — vanilla (no CP / CoT / ICL / snippets)",
        "  g2_cp               — + CP",
        "  g3_cp_cot           — CP + CoT",
        "  g4_cp_icl           — CP + ICL",
        "  g5_cp_icl_cot       — CP + ICL + CoT",
        "  g6_cp_snippets      — CP + snippets",
        "  g7_cp_cot_snippets  — CP + CoT + snippets",
        "",
    ]

    split_cache: dict[str, dict[str, str]] = {}
    for comp in COMPONENTS:
        src = PROMPT_DIR / f"{comp}.txt"
        split_cache[comp] = split_prompt(src.read_text(encoding="utf-8"))
        print(f"split {comp}: header={len(split_cache[comp]['header'])} "
              f"cp={len(split_cache[comp]['cp'])} cot={len(split_cache[comp]['cot'])} "
              f"icl={len(split_cache[comp]['icl'])}")

    for group, flags in GROUPS.items():
        gdir = OUT_ROOT / group
        gdir.mkdir(parents=True)
        manifest.append(f"{group}: cp={flags['cp']} cot={flags['cot']} "
                        f"icl={flags['icl']} snippets={flags['snippets']}")
        for comp in COMPONENTS:
            body = compose(split_cache[comp], flags)
            (gdir / f"{comp}.txt").write_text(body, encoding="utf-8")
            manifest.append(f"  - {comp}.txt ({len(body)} chars)")
        # flags sidecar for runtime
        flag_lines = [
            f"cp={int(flags['cp'])}",
            f"cot={int(flags['cot'])}",
            f"icl={int(flags['icl'])}",
            f"snippets={int(flags['snippets'])}",
            "",
        ]
        (gdir / "FLAGS.txt").write_text("\n".join(flag_lines), encoding="utf-8")
        manifest.append("")

    (OUT_ROOT / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"Wrote ablation prompts under {OUT_ROOT}")


if __name__ == "__main__":
    main()
