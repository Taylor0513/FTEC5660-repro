# Reproducing A-Mem on LoCoMo with DeepSeek (FTEC5660 Reproducibility Work)

> This repo is a fork of [WujiangXu/AgenticMemory](https://github.com/WujiangXu/AgenticMemory)
> used for a reproducibility study. The upstream README is kept below; **this section**
> documents everything about our reproduction.

## What we reproduced

- **Paper**: *A-Mem: Agentic Memory for LLM Agents* (arXiv:2502.12110, NeurIPS 2025)
- **Target claim**: A-Mem's QA performance on the LoCoMo benchmark — per-category
  F1 / BLEU-1, paper Table 1 (GPT-4o-mini rows) — and the ablation "w/o memory
  evolution", paper Table 3.
- **Scope**: first 2 conversations of `data/locomo10.json` (`--ratio 0.2`,
  788 dialogue turns, 304 QA pairs).
- **Controlled change of substrate (assignment Edge Case A)**: the backbone LLM is
  **DeepSeek (`deepseek-chat`)** through its OpenAI-compatible endpoint instead of
  GPT-4o-mini. Numbers are therefore **not directly comparable** to the paper; we
  compare as fairly as possible (same tasks, same metrics, same evaluation code).
- **Our modification (isolated, measurable)**: a new `--disable_evolution` flag that
  removes the *memory evolution* step (the "update neighbors" LLM call) while keeping
  note construction and link generation intact — i.e. the paper's "w/o ME" ablation.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your DeepSeek key; never commit .env
source .env
```

CPU-only is fine (the `all-MiniLM-L6-v2` embedder runs locally). If `huggingface.co`
is unreachable, set `HF_ENDPOINT=https://hf-mirror.com` first.

## Run

```bash
bash run_reproduction.sh
```

Order: smoke test → full reproduction (seed 42) → ablation (seed 42) →
`analyze_results.py`, which writes `analysis/comparison_table.md` and
`analysis/fig_f1_comparison.png`.

## Code changes vs upstream (one commit each)

1. `memory_layer_robust.py`, `test_advanced_robust.py`, `utils.py`: add the
   `disable_evolution` switch + `--disable_evolution` CLI flag (ablation runs use a
   separate memory-cache directory `*_noevo`); make the NLTK `wordnet` download
   non-blocking (wordnet only feeds the auxiliary METEOR metric, which is already
   guarded by try/except — no effect on F1/BLEU).
2. Added `make_smoke_dataset.py`, `run_reproduction.sh`, `analyze_results.py`,
   `.env.example`, this README section.

## Results

See `analysis/comparison_table.md` (generated) and the written report for discussion.

## Secrets

No API keys are committed. `.env` is git-ignored; only `.env.example` is tracked.

---

# Agentic Memory 🧠

A novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way.

> **Note:** This repository is specifically designed to reproduce the results presented in our paper. If you want to use the A-Mem system in building your agents, please refer to our official implementation at: [A-mem-sys](https://github.com/WujiangXu/A-mem-sys)

For more details, please refer to our paper: [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/pdf/2502.12110)

> 🔥 **News (Oct 2026):** We provide a stronger version of A-Mem, **FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents** ([paper](https://arxiv.org/html/2610.33274v1), [code](https://github.com/WujiangXu/Flex-Amem)). FlexA-Mem keeps A-Mem's linked notes but uses them as the compressed history itself: at every step, the agent sees the whole history as a timeline of notes, and the notes most relevant to the current step are expanded in place into their raw content or excerpts, so the prompt fits any token budget. It is evaluated on LongMemEval-S, LoCoMo, SWE-bench Verified and AppWorld.

## Introduction 🌟

Large Language Model (LLM) agents have demonstrated remarkable capabilities in handling complex real-world tasks through external tool usage. However, to effectively leverage historical experiences, they require sophisticated memory systems. Traditional memory systems, while providing basic storage and retrieval functionality, often lack advanced memory organization capabilities.

Our project introduces an innovative **Agentic Memory** system that revolutionizes how LLM agents manage and utilize their memories:

<div align="center">
  <img src="Figure/intro-a.jpg" alt="Traditional Memory System" width="600"/>
  <img src="Figure/intro-b.jpg" alt="Our Proposed Agentic Memory" width="600"/>
  <br>
  <em>Comparison between traditional memory system (top) and our proposed agentic memory (bottom). Our system enables dynamic memory operations and flexible agent-memory interactions.</em>
</div>

## Key Features ✨

- 🔄 Dynamic memory organization based on Zettelkasten principles
- 🔍 Intelligent indexing and linking of memories
- 📝 Comprehensive note generation with structured attributes
- 🌐 Interconnected knowledge networks
- 🔄 Continuous memory evolution and refinement
- 🤖 Agent-driven decision making for adaptive memory management

## Framework 🏗️

<div align="center">
  <img src="Figure/framework.jpg" alt="Agentic Memory Framework" width="800"/>
  <br>
  <em>The framework of our Agentic Memory system showing the dynamic interaction between LLM agents and memory components.</em>
</div>

## How It Works 🛠️

When a new memory is added to the system:
1. Generates comprehensive notes with structured attributes
2. Creates contextual descriptions and tags
3. Analyzes historical memories for relevant connections
4. Establishes meaningful links based on similarities
5. Enables dynamic memory evolution and updates

## Results 📊

Empirical experiments conducted on six foundation models demonstrate superior performance compared to existing SOTA baselines.

## Getting Started 🚀

1. Clone the repository:
```bash
git clone https://github.com/WujiangXu/AgenticMemory.git
cd AgenticMemory
```

2. Install dependencies:
Option 1: Using venv (Python virtual environment)
```bash
# Create and activate virtual environment
python -m venv a-mem
source a-mem/bin/activate  # Linux/Mac
a-mem\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

Option 2: Using Conda
```bash
# Create and activate conda environment
conda create -n myenv python=3.9
conda activate myenv

# Install dependencies
pip install -r requirements.txt
```

3. Run the experiments in LoCoMo dataset:

**Option A — Original evaluation (requires OpenAI JSON schema support):**
```bash
python test_advanced.py
```

**Option B — Robust evaluation (recommended, works with any LLM backend):**

The robust evaluation (`test_advanced_robust.py`) removes the JSON schema dependency and supports OpenAI API, vLLM, and Ollama backends.

```bash
# OpenAI models
python test_advanced_robust.py --backend openai --model gpt-4o-mini \
    --dataset data/locomo10.json --output results_robust_gpt-4o-mini.json

# vLLM-served open-source models (start vLLM server first)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct --port 30000 \
    --dtype float16 --enforce-eager --max-model-len 8192
python test_advanced_robust.py --backend vllm --model Qwen/Qwen2.5-3B-Instruct \
    --dataset data/locomo10.json --output results_robust_qwen3b.json \
    --sglang_port 30000

# Ollama models
python test_advanced_robust.py --backend ollama --model qwen2.5:3b \
    --dataset data/locomo10.json --output results_robust_ollama_qwen3b.json
```

Key arguments:
- `--retrieve_k`: Number of memories to retrieve per query (default: 10). Tune this per model for best results.
- `--ratio`: Fraction of dataset to evaluate (e.g., `--ratio 0.1` for 10% quick test).
- `--backend`: One of `openai`, `vllm`, `ollama`.
- `--sglang_port`: Port for vLLM/SGLang server (default: 30000).

4. Run the full k-sweep to find optimal retrieval k per model:
```bash
bash run_k_sweep.sh
```

**Note:** To achieve the optimal performance reported in our paper, please adjust the hyperparameter `k` value accordingly. Memories are cached after the first run, so subsequent k-sweep evaluations only re-run the QA answering step.

**Categories Information:** The LoCoMo dataset contains the following categories:
* Category 1: Multi-hop
* Category 2: Temporal
* Category 3: Open-domain
* Category 4: Single-hop
* Category 5: Adversarial

For more details about the categories, please refer to [this GitHub issue](https://github.com/snap-research/locomo/issues/6). 

## Citation 📚

If you use this code in your research, please cite our series of works on agentic memory:

```bibtex
@inproceedings{xu2025amem,
  title={A-Mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Liang, Zujie and Mei, Kai and Gao, Hang and Tan, Juntao and Zhang, Yongfeng},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}

@article{xu2026flexamem,
  title={FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents},
  author={Xu, Wujiang and Han, Jiaojiao and Zhu, Xi and Zhang, Qizheng and Liang, Kaiqu and Yin, Yutong and Mei, Kai and Metaxas, Dimitris N.},
  journal={arXiv preprint arXiv:2610.33274},
  year={2026}
}
```

## License 📄

This project is licensed under the MIT License. See LICENSE for details.


