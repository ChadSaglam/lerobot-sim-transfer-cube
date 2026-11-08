"""Collect lerobot-eval results into one Markdown table.

Reads every outputs/eval/<job>/eval_info.json, extracts the success rate and its
95% Wilson interval, and writes results/results.md plus results/results.csv.

Usage:
    python scripts/collect_results.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

EVAL_DIR = Path("outputs/eval")
OUT_DIR = Path("results")

JOB_RE = re.compile(r"^act_seed(?P<seed>\d+)_(?P<step>\d{6})_n(?P<n>\d+)$")


def describe(job: str) -> tuple[str, str, int]:
    """Return (model, step, sort_key) for a job name."""
    if job.startswith("baseline_official_act"):
        return "official lerobot/act_aloha_sim_transfer_cube_human", "80k (reported)", 0
    m = JOB_RE.match(job)
    if m:
        step = int(m.group("step"))
        return f"own ACT, seed {m.group('seed')}", f"{step // 1000}k", step
    return job, "-", 10**9


def main() -> None:
    rows = []
    for info_path in sorted(EVAL_DIR.glob("*/eval_info.json")):
        job = info_path.parent.name
        overall = json.loads(info_path.read_text())["overall"]
        model, step, key = describe(job)
        lo, hi = overall["pc_success_ci95"]
        rows.append(
            {
                "job": job,
                "model": model,
                "step": step,
                "n_episodes": overall["n_episodes"],
                "n_success": overall["n_success"],
                "success_pct": round(overall["pc_success"], 1),
                "ci95_low": round(lo, 1),
                "ci95_high": round(hi, 1),
                "avg_max_reward": round(overall["avg_max_reward"], 2),
                "_key": (model, key, overall["n_episodes"]),
            }
        )
    rows.sort(key=lambda r: r["_key"])

    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "results.csv").open("w", newline="") as f:
        fields = [k for k in rows[0] if not k.startswith("_")]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    lines = [
        "| Model | Steps | Episodes | Success | 95% CI (Wilson) | Avg. max reward |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['step']} | {r['n_episodes']} | "
            f"{r['success_pct']:.1f}% ({r['n_success']}/{r['n_episodes']}) | "
            f"{r['ci95_low']:.1f}-{r['ci95_high']:.1f} | {r['avg_max_reward']:.2f} |"
        )
    (OUT_DIR / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {OUT_DIR / 'results.md'} and {OUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
