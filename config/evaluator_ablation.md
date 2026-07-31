# VLM evaluator ablation (G1–G5)

Ablation study over `evaluate_with_vlm` prompt structure and generated-scenario modalities.

| Group | Config | Prompt (`src/prompt/`) | BEV video | Scenic code |
|-------|--------|------------------------|-----------|-------------|
| G1 | `config_g1_vanilla.yaml` | `vlm_eval/evaluate_with_vlm_g1_vanilla` | yes | no |
| G2 | `config_g2_contextual.yaml` | `vlm_eval/evaluate_with_vlm_g2_contextual` | yes | no |
| G3 | `config_g3_full.yaml` | `vlm_eval/evaluate_with_vlm_g3_full` | yes | no |
| G4 | `config_g4_code_only.yaml` | `vlm_eval/evaluate_with_vlm_g4_code_only` | no | yes |
| G5 | `config_g5_code_video.yaml` | `vlm_eval/evaluate_with_vlm_g5_code_video` | yes | yes |

- **G1**: vanilla scoring prompt (no contextual catalogs / Scenic taxonomy, no CoT stages).
- **G2**: G1 + contextual prompting (vocabularies, BEV/Scenic perception hints), still no CoT.
- **G3**: full evaluator (contextual + 3-stage CoT) — current production-style critic.
- **G4**: score from generated Scenic code only (no BEV video).
- **G5**: score from Scenic code + generated BEV video.

Each YAML sets:

```yaml
critic:
  prompt_name: vlm_eval/<prompt>   # relative to src/prompt/, without .txt
  include_bev_video: true|false
  include_scenic_code: true|false
```

## Run all five groups

```bash
bash scripts/run_evaluator_ablation.sh
# optional smoke:
bash scripts/run_evaluator_ablation.sh --limit 2 --categories text-only
```

Outputs:

```
eval/results/vlm_ablation/
  run_<timestamp>/
    config_g1_vanilla/batch_results.csv
    config_g2_contextual/...
    ...
  logs/
    run_<timestamp>.log
```

Ordinary per-group failures continue to the next group. A **CUDA out-of-memory**
error aborts the whole sweep immediately (e2e exit code `99`).
