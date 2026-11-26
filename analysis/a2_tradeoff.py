"""Draw the A2 accuracy against latency trade-off figure (analysis/output/figures/a2_tradeoff.png).

One point per speed-tested configuration (gemma4:e4b alone, Cascade S1, Cascade S2):
  x = pooled median latency (p50) of the successful POST /tickets samples in the 3 official .jtl files (n = 90)
  y = test-set accuracy from analysis/output/a2_accuracy.csv (88 tickets)
The dashed line is the overall accuracy limit. Runs listed in analysis/excluded_runs.csv (the trial run 99) are
skipped. Every value is checked against docs/A2_SPEED_RUNS_SUMMARY.md and the script stops on a mismatch.

    py analysis/a2_tradeoff.py

Needs matplotlib (py -m pip install matplotlib).
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "analysis" / "output"
FIG = OUT / "figures"

# speed config folder name -> (label, accuracy config in a2_accuracy.csv)
CONFIGS = {
    "gemmaAlone": ("gemma4:e4b alone", "gemma4:e4b"),
    "cascadeS1": ("Cascade S1", "phi3:3.8b -> gemma4:e4b (s1)"),
    "cascadeS2": ("Cascade S2", "phi3:3.8b -> gemma4:e4b (s2)"),
}
# Figures as printed in docs/A2_SPEED_RUNS_SUMMARY.md: (pooled p50 in s, correct out of 88)
EXPECTED = {"gemmaAlone": (43.6, 69), "cascadeS1": (26.7, 70), "cascadeS2": (3.8, 67)}
OVERALL_LIMIT_PCT = 68.4  # 61/88, from analysis/output/a2_limits.md


def excluded() -> set[str]:
    with open(ROOT / "analysis" / "excluded_runs.csv", newline="", encoding="utf-8") as fh:
        return {r["run_id"] for r in csv.DictReader(fh)}


def pooled_p50(cfg: str, skip: set[str]) -> tuple[float, int]:
    """Median of every successful POST /tickets sample across runs 1 to 3, checked against client_summary.json."""
    pool: list[float] = []
    for n in (1, 2, 3):
        [run_dir] = [p for p in RUNS.glob(f"*_speed_{cfg}_run{n}") if p.name not in skip]
        [jtl] = list(run_dir.glob("*.jtl"))
        with open(jtl, newline="", encoding="utf-8") as fh:
            ok = [int(r["elapsed"]) / 1000 for r in csv.DictReader(fh)
                  if r["label"] == "POST /tickets" and r["success"] == "true"]
        cs = json.loads((run_dir / "client_summary.json").read_text(encoding="utf-8"))
        assert len(ok) == cs["completed_ok"], f"{run_dir.name}: OK count differs from client_summary.json"
        assert abs(st.median(ok) * 1000 - cs["p50_ms"]) < 1, f"{run_dir.name}: p50 differs from client_summary.json"
        pool += ok
    return st.median(pool), len(pool)


def accuracy() -> dict[str, dict]:
    with open(OUT / "a2_accuracy.csv", newline="", encoding="utf-8") as fh:
        return {r["config"]: r for r in csv.DictReader(fh)}


def main() -> None:
    skip, acc = excluded(), accuracy()
    points = {}
    for cfg, (label, acc_cfg) in CONFIGS.items():
        p50, n = pooled_p50(cfg, skip)
        row = acc[acc_cfg]
        want_p50, want_correct = EXPECTED[cfg]
        assert n == 90, f"{cfg}: pooled n = {n}, expected 90"
        assert round(p50, 1) == want_p50, f"{cfg}: pooled p50 {p50:.1f} s != summary {want_p50} s"
        assert int(row["correct"]) == want_correct, f"{cfg}: {row['correct']}/88 != summary {want_correct}/88"
        points[cfg] = (label, p50, float(row["accuracy_pct"]))

    gemma_pct = points["gemmaAlone"][2]
    limit = gemma_pct - 10  # the brief: overall limit = top model's test-set accuracy minus 10 pp
    assert round(limit, 1) == OVERALL_LIMIT_PCT, f"overall limit {limit:.1f}% != {OVERALL_LIMIT_PCT}%"

    plt.rcParams.update({"font.family": "Calibri", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#A6A6A6"})
    red, grey = "#C00020", "#6E6E6E"
    style = {"gemmaAlone": ("o", grey, grey, "passes both limits", (8, -58), "center"),
             "cascadeS1": ("D", red, red, "passes both limits, recommended", (10, 8), "left"),
             "cascadeS2": ("^", "white", grey, "fails the Consumer loan limit", (10, 8), "left")}

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    ax.axhspan(limit, 85, color="#F4F4F4", zorder=0)
    ax.axhline(limit, color=red, ls="--", lw=1.4)
    ax.text(49.5, limit - 1.0, f"Overall limit {limit:.1f}%", color=red, ha="right", va="top")
    ax.text(49.5, 84.3, "Meets the overall limit", color=grey, ha="right", va="top")
    for cfg, (label, x, y) in points.items():
        marker, face, edge, note, offset, ha = style[cfg]
        ax.scatter([x], [y], marker=marker, s=130, facecolor=face, edgecolor=edge, linewidth=2, zorder=3)
        ax.annotate(f"{label}\n{x:.1f} s, {y:.1f}%\n({note})", (x, y), textcoords="offset points",
                    xytext=offset, ha=ha, fontsize=9.5)
    ax.set(xlim=(0, 50), ylim=(60, 85), xlabel="Pooled median latency, p50 (s), 3 runs, n = 90",
           ylabel="Test-set accuracy (%), 88 tickets")
    ax.grid(color="#EEEEEE")
    fig.tight_layout()

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "a2_tradeoff.png"
    fig.savefig(out)
    plt.close(fig)
    for label, x, y in points.values():
        print(f"{label:18s} p50 {x:5.1f} s   accuracy {y:.1f}%")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
