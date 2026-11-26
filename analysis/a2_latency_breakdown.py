"""Draw the A2 mean time per ticket figure (analysis/output/figures/a2_latency_breakdown.png).

Three stacked bars, pooled over the 3 official runs of each configuration:
  gemma4:e4b alone     mean client latency from the .jtl files
  S1 escalated ticket  mean phi3:3.8b call + mean gemma4:e4b call (small_ms and large_ms in server_access.log)
  S1 accepted ticket   mean phi3:3.8b call
It also checks the brief's cascade formula, small + f x large, against the measured S1 mean. Runs listed in
analysis/excluded_runs.csv (the trial run 99) are skipped. Counts are checked against client_summary.json,
sut_summary.json and docs/A2_SPEED_RUNS_SUMMARY.md, and the script stops on a mismatch.

    py analysis/a2_latency_breakdown.py

Needs matplotlib (py -m pip install matplotlib).
"""
from __future__ import annotations

import csv
import json
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FIG = ROOT / "analysis" / "output" / "figures"

EXPECTED_ESCALATED = 48  # cascadeS1, 16 of 30 in each run (docs/A2_SPEED_RUNS_SUMMARY.md)
EXPECTED_RUN_MEANS = {"gemmaAlone": (74.5, 42.8, 43.4), "cascadeS1": (25.7, 21.8, 22.5)}


def excluded() -> set[str]:
    with open(ROOT / "analysis" / "excluded_runs.csv", newline="", encoding="utf-8") as fh:
        return {r["run_id"] for r in csv.DictReader(fh)}


def run_dirs(cfg: str, skip: set[str]) -> list[Path]:
    dirs = []
    for n in (1, 2, 3):
        [d] = [p for p in RUNS.glob(f"*_speed_{cfg}_run{n}") if p.name not in skip]
        dirs.append(d)
    return dirs


def jtl_ok(run_dir: Path) -> list[float]:
    """Elapsed seconds of the successful POST /tickets samples, checked against client_summary.json."""
    [jtl] = list(run_dir.glob("*.jtl"))
    with open(jtl, newline="", encoding="utf-8") as fh:
        ok = [int(r["elapsed"]) / 1000 for r in csv.DictReader(fh)
              if r["label"] == "POST /tickets" and r["success"] == "true"]
    cs = json.loads((run_dir / "client_summary.json").read_text(encoding="utf-8"))
    assert len(ok) == cs["completed_ok"], f"{run_dir.name}: OK count differs from client_summary.json"
    return ok


def log_lines(run_dir: Path) -> list[dict]:
    """Answered POST /tickets lines of the service log. Warm-up traffic (run=-) is skipped."""
    rows = []
    for line in (run_dir / "server_access.log").read_text(encoding="utf-8").splitlines():
        if "POST /tickets" not in line or " 200 " not in line:
            continue
        d = {k: v.strip('"') for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)', line)}
        if d.get("run", "-") != "-":
            rows.append(d)
    return rows


def ms_field(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) / 1000 for r in rows if r.get(key, "-") not in ("-", "")]


def main() -> None:
    skip = excluded()

    gemma_pool: list[float] = []
    for d, want in zip(run_dirs("gemmaAlone", skip), EXPECTED_RUN_MEANS["gemmaAlone"]):
        ok = jtl_ok(d)
        assert round(st.mean(ok), 1) == want, f"{d.name}: mean {st.mean(ok):.1f} s != summary {want} s"
        gemma_pool += ok

    s1_pool: list[float] = []
    small: list[float] = []
    large: list[float] = []
    escalated = answered = 0
    for d, want in zip(run_dirs("cascadeS1", skip), EXPECTED_RUN_MEANS["cascadeS1"]):
        ok = jtl_ok(d)
        assert round(st.mean(ok), 1) == want, f"{d.name}: mean {st.mean(ok):.1f} s != summary {want} s"
        s1_pool += ok
        rows = log_lines(d)
        esc = sum(r.get("escalated") == "1" for r in rows)
        sut = json.loads((d / "sut_summary.json").read_text(encoding="utf-8"))
        assert esc == sut["log_escalated"], f"{d.name}: escalation differs from sut_summary.json"
        escalated += esc
        answered += len(rows)
        small += ms_field(rows, "small_ms")
        large += ms_field(rows, "large_ms")
    assert escalated == EXPECTED_ESCALATED, f"cascadeS1: escalated {escalated} != summary {EXPECTED_ESCALATED}"

    gemma_mean, s1_mean = st.mean(gemma_pool), st.mean(s1_pool)
    small_mean, large_mean = st.mean(small), st.mean(large)
    f = escalated / answered
    formula = small_mean + f * large_mean

    plt.rcParams.update({"font.family": "Calibri", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#A6A6A6"})
    red, grey = "#C00020", "#6E6E6E"
    bars = [("gemma4:e4b alone", 0, gemma_mean), ("S1 escalated ticket", small_mean, large_mean),
            ("S1 accepted ticket", small_mean, 0)]

    fig, ax = plt.subplots(figsize=(7.5, 3.0), dpi=200)
    for i, (label, a, b) in enumerate(bars):
        ax.barh(i, a, color=grey, label="phi3:3.8b call" if i == 1 else None)
        ax.barh(i, b, left=a, color=red, label="gemma4:e4b call" if i == 1 else None)
        ax.text(a + b + 0.8, i, f"{a + b:.1f} s", va="center")
    ax.set_yticks(range(3), [b[0] for b in bars])
    ax.set_xlabel("Mean time per ticket (s), from the service log and .jtl files")
    ax.set_xlim(0, 62)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout()

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "a2_latency_breakdown.png"
    fig.savefig(out)
    plt.close(fig)

    print(f"phi3:3.8b call         median {st.median(small):5.1f} s   mean {small_mean:5.1f} s   (n = {len(small)})")
    print(f"gemma4:e4b in cascade  median {st.median(large):5.1f} s   mean {large_mean:5.1f} s   (n = {len(large)})")
    print(f"escalated              {escalated}/{answered} = {f:.1%}")
    print(f"formula small + f x large = {small_mean:.1f} + {f:.3f} x {large_mean:.1f} = {formula:.1f} s")
    print(f"measured mean          Cascade S1 {s1_mean:.1f} s   gemma4:e4b alone {gemma_mean:.1f} s")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
