"""
PHASE 4 - Cross-phase evaluation & comparative analysis
Part A: time Phase 2 and Phase 3 end-to-end.
Part B: combined two-stage confusion matrix, from phase2_metrics.json + phase3_recheck.json.
Part C: McNemar's exact test on the two-stage improvement.
Run: python phase4_analysis.py [--time]
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
    t2 = time_script("Phase 2", PHASE2_SCRIPT, PHASE2_SCRIPT.parent)
    t3 = time_script("Phase 3", PHASE3_RUNNER, PHASE3_RUNNER.parent)
    return {"phase2_seconds": round(t2, 2), "phase3_seconds": round(t3, 2),
            "phase3_over_phase2_ratio": round(t3 / t2, 2) if t2 else None}

# SECTION 2 - COMBINED TWO-STAGE CONFUSION MATRIX (Part B)

def combined_confusion(phase2, recheck):
    tn2, fp2, fn2, tp2 = phase2["tn"], phase2["fp"], phase2["fn"], phase2["tp"]
    fp_before, tp_before = recheck["false_positives_before"], recheck["attacks_reaching_phase3"]
    fp_after, tp_kept = recheck["false_positives_after"], recheck["attacks_kept"]

    if not (0 <= fp_before <= fp2 and 0 <= tp_before <= tp2):
        sys.exit("phase3_recheck.json doesn't fit inside phase2_metrics.json - check both are from the same run.")

    fp_unmatched, tp_unmatched = fp2 - fp_before, tp2 - tp_before
    fp_removed = fp_before - fp_after
    tp_removed = tp_before - tp_kept

    TN, FP = tn2 + fp_removed, fp_unmatched + fp_after
    FN, TP = fn2 + tp_removed, tp_unmatched + tp_kept

    total = TN + FP + FN + TP
    p = TP / (TP + FP) if (TP + FP) else 0.0
    r = TP / (TP + FN) if (TP + FN) else 0.0
    f1 = 2 * p * r / (p + r + 1e-12)

    return {"tn": int(TN), "fp": int(FP), "fn": int(FN), "tp": int(TP),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "accuracy": round((TP + TN) / total, 4),
            "false_positive_reduction_vs_phase2": round(1 - FP / fp2, 4) if fp2 else None,
            "_b": int(fp_removed), "_c": int(tp_removed)}

# SECTION 3 - MCNEMAR'S EXACT TEST (Part C)

def mcnemar_significance(b, c):
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "note": "no discordant pairs"}
    p_value = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    sig = p_value < 0.05
    interp = ("significant improvement over Phase 2 alone" if sig and b > c else
              "significantly worse than Phase 2 alone" if sig else
              "not statistically significant at alpha=0.05")
    return {"b": b, "c": c, "p_value": round(p_value, 6), "significant": bool(sig), "interpretation": interp}

# SECTION 4 - RUN

if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    p2_path, recheck_path = RESULTS / "phase2_metrics.json", RESULTS / "phase3_recheck.json"
    if not p2_path.exists() or not recheck_path.exists():
        sys.exit(f"Missing inputs:\n  {p2_path}\n  {recheck_path}")

    phase2 = json.loads(p2_path.read_text())[PHASE2_METHOD]
    recheck = json.loads(recheck_path.read_text())

    timing = run_timing() if "--time" in sys.argv else {"note": "run with --time to measure wall-clock time"}
    print("\ntiming:", json.dumps(timing, indent=2))

    combined = combined_confusion(phase2, recheck)
    print("\ncombined two-stage system:", json.dumps({k: v for k, v in combined.items() if not k.startswith("_")}, indent=2))

    sig = mcnemar_significance(combined["_b"], combined["_c"])
    print("\nsignificance:", json.dumps(sig, indent=2))

    phase2_total = phase2["tn"] + phase2["fp"] + phase2["fn"] + phase2["tp"]
    out = {"timing": timing,
          "combined_two_stage_system": {k: v for k, v in combined.items() if not k.startswith("_")},
          "significance": sig,
          "phase2_alone": {"precision": phase2["precision"], "recall": phase2["recall"], "f1": phase2["f1"],
                           "accuracy": round((phase2["tn"] + phase2["tp"]) / phase2_total, 4)}}
    (RESULTS / "phase4_combined_metrics.json").write_text(json.dumps(out, indent=2))
    print("\nsaved:", RESULTS / "phase4_combined_metrics.json")
