"""Aggregate reproduction results into comparison tables and plots.

Reads the results_*.json files produced by run_reproduction.sh, compares against the
paper's reported numbers (Table 1 / Table 3, GPT-4o-mini rows), and writes:
  - analysis/comparison_table.md      (ours vs paper, per category, F1 & BLEU-1)
  - analysis/trials_variance.md       (mean +/- std across the 3 seeds)
  - analysis/fig_f1_comparison.png    (grouped bar chart)
Usage: python analyze_results.py
"""
import json
import os
import glob
import statistics

CATS = ["1", "2", "3", "4", "5"]
CAT_NAMES = {"1": "Multi-Hop", "2": "Temporal", "3": "Open-Domain", "4": "Single-Hop", "5": "Adversarial"}

# Paper Table 1 (A-Mem, GPT-4o-mini) and Table 3 (ablations, GPT-4o-mini), F1 / BLEU-1 in %
PAPER_FULL_F1 = {"1": 27.02, "2": 45.85, "3": 12.14, "4": 44.65, "5": 50.03}
PAPER_FULL_BLEU = {"1": 20.09, "2": 36.67, "3": 12.00, "4": 37.06, "5": 49.47}
PAPER_NOMEC_F1 = {"1": 21.35, "2": 31.24, "3": 10.13, "4": 39.17, "5": 44.16}
PAPER_NOMEC_BLEU = {"1": 15.13, "2": 27.31, "3": 10.85, "4": 34.70, "5": 45.33}


def load(path):
    with open(path) as f:
        return json.load(f)


def cat_metric(result, metric):
    """Per-category mean of a metric, as percentages."""
    agg = result["aggregate_metrics"]
    out = {}
    for c in CATS:
        key = f"category_{c}"
        if key in agg and metric in agg[key]:
            out[c] = agg[key][metric]["mean"] * 100
        else:
            out[c] = float("nan")
    return out


def overall_metric(result, metric):
    return result["aggregate_metrics"]["overall"][metric]["mean"] * 100


def collect(pattern):
    files = sorted(glob.glob(pattern))
    return [load(f) for f in files]


def fmt(x):
    return "n/a" if x != x else f"{x:.2f}"


def main():
    os.makedirs("analysis", exist_ok=True)
    full_runs = collect("results_full_s*.json")
    noevo_runs = [r for r in collect("results_noevo_s*.json")]
    full_main = [r for r in full_runs if "corrected" not in r.get("_file", "")]

    # tag filenames for clarity
    for r in full_runs + noevo_runs:
        pass

    # --- trials variance ---
    lines = ["# Runs overview (single seed per condition; see report for the variance discussion)\n"]
    for label, runs in [("Full A-Mem", [r for r in full_runs]), ("w/o ME (ablation)", noevo_runs)]:
        runs = [r for r in runs if r.get("legacy_c5_reference") in (True, False)]
        lines.append(f"\n## {label} ({len(runs)} run(s))\n")
        lines.append("| Category | F1 mean | F1 std | BLEU-1 mean | BLEU-1 std |")
        lines.append("|---|---|---|---|---|")
        for c in CATS:
            f1s = [cat_metric(r, "f1")[c] for r in runs]
            b1s = [cat_metric(r, "bleu1")[c] for r in runs]
            f1s = [x for x in f1s if x == x]
            b1s = [x for x in b1s if x == x]
            if not f1s:
                continue
            lines.append(
                f"| {CAT_NAMES[c]} | {statistics.mean(f1s):.2f} | "
                f"{(statistics.stdev(f1s) if len(f1s) > 1 else 0):.2f} | "
                f"{statistics.mean(b1s):.2f} | {(statistics.stdev(b1s) if len(b1s) > 1 else 0):.2f} |"
            )
    with open("analysis/trials_variance.md", "w") as f:
        f.write("\n".join(lines))

    # --- comparison vs paper (seed 42 runs) ---
    full42 = next(r for r in full_runs if r["seed"] == 42 and r.get("legacy_c5_reference"))
    noevo42 = next(r for r in noevo_runs if r["seed"] == 42)
    full42_f1, full42_b1 = cat_metric(full42, "f1"), cat_metric(full42, "bleu1")
    noevo42_f1, noevo42_b1 = cat_metric(noevo42, "f1"), cat_metric(noevo42, "bleu1")

    lines = ["# Ours (DeepSeek) vs Paper (GPT-4o-mini), LoCoMo first 2 conversations\n",
             "Paper numbers: Table 1 (A-Mem row) and Table 3 (w/o ME row). "
             "NOTE: different backbone LLM — not directly comparable.\n",
             "| Category | Paper full F1 | Ours full F1 | Paper w/o ME F1 | Ours w/o ME F1 | "
             "Paper full BLEU | Ours full BLEU |",
             "|---|---|---|---|---|---|---|"]
    for c in CATS:
        lines.append(
            f"| {CAT_NAMES[c]} | {PAPER_FULL_F1[c]:.2f} | {fmt(full42_f1[c])} | "
            f"{PAPER_NOMEC_F1[c]:.2f} | {fmt(noevo42_f1[c])} | {PAPER_FULL_BLEU[c]:.2f} | {fmt(full42_b1[c])} |"
        )
    with open("analysis/comparison_table.md", "w") as f:
        f.write("\n".join(lines))

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(CATS))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 1.5 * w, [PAPER_FULL_F1[c] for c in CATS], w, label="Paper full (GPT-4o-mini)")
    ax.bar(x - 0.5 * w, [full42_f1[c] for c in CATS], w, label="Ours full (DeepSeek)")
    ax.bar(x + 0.5 * w, [PAPER_NOMEC_F1[c] for c in CATS], w, label="Paper w/o ME")
    ax.bar(x + 1.5 * w, [noevo42_f1[c] for c in CATS], w, label="Ours w/o ME (DeepSeek)")
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_NAMES[c] for c in CATS])
    ax.set_ylabel("F1 (%)")
    ax.set_title("LoCoMo QA (2-conversation subset): reproduction + ablation vs paper")
    ax.legend()
    fig.tight_layout()
    fig.savefig("analysis/fig_f1_comparison.png", dpi=150)
    print("wrote analysis/comparison_table.md, analysis/trials_variance.md, analysis/fig_f1_comparison.png")


if __name__ == "__main__":
    main()
