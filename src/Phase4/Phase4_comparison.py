"""
PHASE 4 - Cross-phase evaluation & comparative analysis
Part A: time Phase 2 and Phase 3 end-to-end.
Part B: combined two-stage confusion matrix, from phase2_metrics.json + phase3_recheck.json.
Part C: McNemar's exact test on the two-stage improvement.
Part D: comparison plots (combined confusion matrix + Phase 2 vs Phase 3 vs combined bars).
Run: python Phase4_comparison.py [--time]
"""
import json, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.resolve().parents[1] / "results"
PHASE2_SCRIPT = HERE.parent / "phase2_unsupervised" / "phase2_unsupervised.py"
PHASE3_RUNNER = HERE.parent / "phase3_supervised" / "run3.py"
PY = sys.executable

# flagged_packet_ids.csv comes from AE_threshold's predictions - that's the method Phase 3 rechecks
PHASE2_METHOD = "AE_threshold"


# SECTION 1 - TIMING (Part A)

def time_script(label, script_path, cwd):
    print(f"\nTiming: {label} ...")
    t0 = time.perf_counter()
    rc = subprocess.run([PY, str(script_path)], cwd=str(cwd)).returncode
    elapsed = time.perf_counter() - t0
    if rc != 0:
        sys.exit(f"[STOPPED] {label} failed.")
    print(f"  {label}: {elapsed:.1f}s")
    return elapsed


def run_timing():
    # Phase 2 and Phase 3 are each one-click runners, so we just clock the whole process.
    t2 = time_script("Phase 2", PHASE2_SCRIPT, PHASE2_SCRIPT.parent)
    t3 = time_script("Phase 3", PHASE3_RUNNER, PHASE3_RUNNER.parent)
    return {
        "phase2_seconds": round(t2, 2),
        "phase3_seconds": round(t3, 2),
        "phase3_over_phase2_ratio": round(t3 / t2, 2) if t2 else None,
    }


# SECTION 2 - COMBINED TWO-STAGE CONFUSION MATRIX (Part B)

def combined_confusion(phase2, recheck):
    """
    Phase 2's confusion matrix covers every packet. Phase 3 only ever touches
    the subset of Phase 2's alerts it could match to a flow, so we walk each
    alert to its final outcome instead of just adding numbers together:
      - FP that Phase 3 correctly cleared          -> becomes TN
      - TP that Phase 3 incorrectly dropped         -> becomes FN
      - anything Phase 3 couldn't match to a flow   -> keeps Phase 2's label
    """
    tn2, fp2, fn2, tp2 = phase2["tn"], phase2["fp"], phase2["fn"], phase2["tp"]
    fp_before, tp_before = recheck["false_positives_before"], recheck["attacks_reaching_phase3"]
    fp_after, tp_kept = recheck["false_positives_after"], recheck["attacks_kept"]

    if not (0 <= fp_before <= fp2 and 0 <= tp_before <= tp2):
        sys.exit("phase3_recheck.json doesn't fit inside phase2_metrics.json - check both are from the same run.")

    # alerts Phase 3 never got a chance to look at (no flow match) - unchanged
    fp_unmatched = fp2 - fp_before
    tp_unmatched = tp2 - tp_before

    fp_removed = fp_before - fp_after   # false positives Phase 3 corrected
    tp_removed = tp_before - tp_kept    # true positives Phase 3 wrongly dropped

    tn_combined = tn2 + fp_removed
    fp_combined = fp_unmatched + fp_after
    fn_combined = fn2 + tp_removed
    tp_combined = tp_unmatched + tp_kept

    total = tn_combined + fp_combined + fn_combined + tp_combined
    precision = tp_combined / (tp_combined + fp_combined) if (tp_combined + fp_combined) else 0.0
    recall = tp_combined / (tp_combined + fn_combined) if (tp_combined + fn_combined) else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return {
        "tn": int(tn_combined), "fp": int(fp_combined),
        "fn": int(fn_combined), "tp": int(tp_combined),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round((tp_combined + tn_combined) / total, 4),
        "false_positive_reduction_vs_phase2": round(1 - fp_combined / fp2, 4) if fp2 else None,
        # discordant pairs for McNemar's test, kept private with the leading underscore
        "_b": int(fp_removed),
        "_c": int(tp_removed),
    }


# SECTION 3 - MCNEMAR'S EXACT TEST (Part C)

def mcnemar_significance(b, c):
    # b = alerts Phase 3 got WRONG that Phase 2 had right (TP -> FN)
    # c = alerts Phase 3 got RIGHT that Phase 2 had wrong (FP -> TN)
    # Only the disagreements between the two systems carry information here.
    from scipy.stats import binomtest

    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "note": "no discordant pairs"}

    p_value = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    sig = p_value < 0.05
    if sig and b > c:
        interp = "significant improvement over Phase 2 alone"
    elif sig:
        interp = "significantly worse than Phase 2 alone"
    else:
        interp = "not statistically significant at alpha=0.05"

    return {"b": b, "c": c, "p_value": round(p_value, 6), "significant": bool(sig), "interpretation": interp}


# SECTION 4 - COMPARISON PLOTS (Part D) -> results/phase4_comparison.png

# bar colors: keep Phase 2 / Phase 3 neutral so the combined pipeline (the
# actual headline result) is the one that pops
COLOR_PHASE2 = "#B0B3B8"
COLOR_PHASE3 = "#4C72B0"
COLOR_COMBINED = "#D9534F"


def save_plots(combined, phase2_alone, phase3_alone):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    plt.rcParams.update({"axes.edgecolor": "#555555", "axes.grid": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))
    fig.patch.set_facecolor("white")

    # left panel - the combined pipeline's confusion matrix, same look as
    # the Phase 2/Phase 3 heatmaps so the report stays visually consistent
    cm = np.array([[combined["tn"], combined["fp"]],
                   [combined["fn"], combined["tp"]]])
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="rocket_r", ax=ax1, cbar=False,
        annot_kws={"size": 12, "weight": "bold"}, linewidths=1, linecolor="white",
        xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"],
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Combined Two-Stage\nConfusion Matrix", fontsize=11, weight="bold")

    # right panel - Phase 2 alone vs Phase 3 alone vs the combined pipeline,
    # so Phase 3's contribution is actually visible instead of implied
    metrics = ["precision", "recall", "f1", "accuracy"]
    phase2_vals = [phase2_alone[m] for m in metrics]
    phase3_vals = [phase3_alone[m] for m in metrics]
    combined_vals = [combined[m] for m in metrics]

    x = np.arange(len(metrics))
    bar_width = 0.26
    bars_phase2 = ax2.bar(x - bar_width, phase2_vals, bar_width,
                           label="Phase 2 at cascade\nhandoff point", color=COLOR_PHASE2, edgecolor="white")
    bars_phase3 = ax2.bar(x, phase3_vals, bar_width,
                           label="Phase 3 on its own\nflow test set", color=COLOR_PHASE3, edgecolor="white")
    bars_combined = ax2.bar(x + bar_width, combined_vals, bar_width,
                             label="Phase 2 + Phase 3\n(combined pipeline)", color=COLOR_COMBINED, edgecolor="white")

    for bars in (bars_phase2, bars_phase3, bars_combined):
        ax2.bar_label(bars, fmt="%.2f", fontsize=7.5, padding=2)

    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in metrics])
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Phase 2 vs. Phase 3 vs. Combined Pipeline", fontsize=11, weight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.savefig(RESULTS / "phase4_comparison.png", dpi=150, facecolor="white")
    plt.close()


# helper - {tn, fp, fn, tp, precision, recall, f1} block -> add accuracy, keep only the 4 headline metrics
def summarize(metrics_block):
    total = metrics_block["tn"] + metrics_block["fp"] + metrics_block["fn"] + metrics_block["tp"]
    accuracy = (metrics_block["tn"] + metrics_block["tp"]) / total
    return {
        "precision": metrics_block["precision"],
        "recall": metrics_block["recall"],
        "f1": metrics_block["f1"],
        "accuracy": round(accuracy, 4),
    }


# SECTION 5 - RUN

if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)

    p2_path = RESULTS / "phase2_metrics.json"
    p3_path = RESULTS / "phase3_metrics.json"
    recheck_path = RESULTS / "phase3_recheck.json"
    missing = [p for p in (p2_path, p3_path, recheck_path) if not p.exists()]
    if missing:
        sys.exit("Missing inputs:\n  " + "\n  ".join(str(p) for p in missing))

    phase2 = json.loads(p2_path.read_text())[PHASE2_METHOD]
    phase3 = json.loads(p3_path.read_text())
    recheck = json.loads(recheck_path.read_text())

    timing = run_timing() if "--time" in sys.argv else {"note": "run with --time to measure wall-clock time"}
    print("\ntiming:", json.dumps(timing, indent=2))

    combined = combined_confusion(phase2, recheck)
    combined_public = {k: v for k, v in combined.items() if not k.startswith("_")}
    print("\ncombined two-stage system:", json.dumps(combined_public, indent=2))

    sig = mcnemar_significance(combined["_b"], combined["_c"])
    print("\nsignificance:", json.dumps(sig, indent=2))

    phase2_alone = summarize(phase2)
    phase3_alone = summarize(phase3)

    # These two are NOT the same-denominator "alone" baselines. Phase 2 here is at its
    # cascade handoff point (high recall, low precision), and Phase 3 here is scored on
    # its own flow test set. For the like-for-like three-way comparison on identical
    # packets, see phase4_cascade_analysis.json.
    out = {
        "timing": timing,
        "combined_two_stage_system": combined_public,
        "significance": sig,
        "phase2_at_cascade_handoff": phase2_alone,
        "phase3_on_own_flow_testset": phase3_alone,
        "note": "like-for-like baselines are in phase4_cascade_analysis.json",
    }
    (RESULTS / "phase4_combined_metrics.json").write_text(json.dumps(out, indent=2))
    print("\nsaved:", RESULTS / "phase4_combined_metrics.json")

    save_plots(combined, phase2_alone, phase3_alone)
    print("saved:", RESULTS / "phase4_comparison.png")
