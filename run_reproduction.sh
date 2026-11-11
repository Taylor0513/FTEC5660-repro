#!/usr/bin/env bash
# Reproduction pipeline for FTEC5660 reproducibility assignment.
# Project: A-Mem (Agentic Memory for LLM Agents), https://github.com/WujiangXu/AgenticMemory
# Target: LoCoMo QA, first 2 conversations (--ratio 0.2), F1 / BLEU-1 per category.
# LLM backend: DeepSeek (deepseek-chat) via the OpenAI-compatible endpoint.
#
# Prereqs:
#   pip install -r requirements.txt
#   source .env   # OPENAI_API_KEY / OPENAI_BASE_URL (see .env.example)
set -euo pipefail

MODEL=deepseek-chat
DATA=data/locomo10.json
COMMON="--backend openai --model ${MODEL} --dataset ${DATA} --ratio 0.2 --legacy_c5_reference"

# 0. Smoke test (tiny subset, ~5 min, validates the full pipeline)
python make_smoke_dataset.py
python test_advanced_robust.py --backend openai --model ${MODEL} \
    --dataset data/locomo_smoke.json --output results_smoke.json --ratio 1.0
# NOTE: delete the smoke cache before the real runs — it shares the cache dir of the
# full configuration:
#   rm -rf cached_memories_robust_openai_${MODEL}

# 1. Reproduction run: full A-Mem, seed 42 (builds + caches memories, then QA)
python test_advanced_robust.py ${COMMON} --seed 42 --output results_full_s42.json

# 2. Modification (ablation w/o memory evolution), seed 42
python test_advanced_robust.py ${COMMON} --seed 42 --disable_evolution --output results_noevo_s42.json

# 3. Variance trials: QA step re-uses the cached memories, only the seeds change
python test_advanced_robust.py ${COMMON} --seed 43 --output results_full_s43.json
python test_advanced_robust.py ${COMMON} --seed 44 --output results_full_s44.json
python test_advanced_robust.py ${COMMON} --seed 43 --disable_evolution --output results_noevo_s43.json
python test_advanced_robust.py ${COMMON} --seed 44 --disable_evolution --output results_noevo_s44.json

# 4. Evaluation tightening: full A-Mem scored with the corrected category-5 reference
#    ("Not mentioned in the conversation") instead of the paper's adversarial reference
python test_advanced_robust.py --backend openai --model ${MODEL} --dataset ${DATA} \
    --ratio 0.2 --seed 42 --output results_full_s42_corrected_c5.json

# 5. Aggregate everything into tables/plots
python analyze_results.py
