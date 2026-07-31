# Codegen / gen_eval ablation (G1–G7)

Ablation study over Scenic **component-generator** prompts (`src/prompt/gen_eval/`).

| Group | Config | Prompt group (`src/prompt/`) | CP | CoT | ICL | Snippets |
|-------|--------|------------------------------|----|-----|-----|----------|
| G1 | `config_g1_zeroshot.yaml` | `gen_eval/g1_zeroshot` | no | no | no | no |
| G2 | `config_g2_cp.yaml` | `gen_eval/g2_cp` | yes | no | no | no |
| G3 | `config_g3_cp_cot.yaml` | `gen_eval/g3_cp_cot` | yes | yes | no | no |
| G4 | `config_g4_cp_icl.yaml` | `gen_eval/g4_cp_icl` | yes | no | yes | no |
| G5 | `config_g5_cp_icl_cot.yaml` | `gen_eval/g5_cp_icl_cot` | yes | yes | yes | no |
| G6 | `config_g6_cp_snippets.yaml` | `gen_eval/g6_cp_snippets` | yes | no | no | yes |
| G7 | `config_g7_cp_cot_snippets.yaml` | `gen_eval/g7_cp_cot_snippets` | yes | yes | no | yes |

Factors:

- **CP** — contextual prompting (hard constraints + Scenic API/syntax + prior components)
- **CoT** — chain-of-thought / reasoning plan
- **ICL** — illustrative few-shot examples
- **Snippets** — retrieved code snippets (`{reference_components}`)

Rebuild prompts (if you change the root `component_generator_*.txt`):

```bash
python scripts/build_gen_eval_ablation_prompts.py
```

Each YAML sets:

```yaml
codegen:
  prompt_group: gen_eval/<group>   # relative to src/prompt/
  use_contextual: true|false       # CP
  use_cot: true|false
  use_icl: true|false
  use_snippet_retrieval: true|false
```

Configs live in `config/config_generator_ablation/` (cloned from `config/config.yaml` plus the `codegen` block).

## Run all seven groups

```bash
bash scripts/run_gen_eval_ablation.sh
# optional smoke:
bash scripts/run_gen_eval_ablation.sh --limit 2 --categories text-only
# optional custom benchmark:
bash scripts/run_gen_eval_ablation.sh --benchmark-root data/benchmark
```

Outputs:

```
eval/results/gen_eval_ablation/
  run_<timestamp>/
    config_g1_zeroshot/batch_results.csv
    config_g2_cp/...
    ...
  logs/
    run_<timestamp>.log
```

Ordinary per-group failures continue to the next group. A **CUDA out-of-memory**
error aborts the whole sweep immediately (e2e exit code `99`).
