"""Plot training loss and checkpoint success rate from the lerobot logs.

Inputs:
    outputs/train/act_transfer_cube_seed<seed>.log (and *_resume80k.log) for the loss curve.
    results/results.csv (from scripts/collect_results.py) for success per checkpoint.
Outputs:
    results/training_loss.png, results/success_vs_steps.png

Usage:
    python scripts/plot_training_curves.py
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRAIN_DIR = Path("outputs/train")
OUT_DIR = Path("results")

# tqdm writes "\r" progress; the INFO line shares the physical line with the exact step counter.
LINE_RE = re.compile(r"(?P<step>\d+)/\d+ \[.*?INFO .*? loss:(?P<loss>[0-9.]+) ")


def read_loss(log_path: Path) -> list[tuple[int, float]]:
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    points = []
    for line in text.splitlines():
        m = LINE_RE.search(line)
        if m:
            points.append((int(m.group("step")), float(m.group("loss"))))
    return points


def plot_loss() -> None:
    plt.figure(figsize=(7, 4))
    for log_path in sorted(TRAIN_DIR.glob("act_transfer_cube_seed*.log")):
        seed = re.search(r"seed(\d+)", log_path.name).group(1)
        label = f"seed {seed}" + (" (resumed to 80k)" if "resume" in log_path.name else "")
        pts = read_loss(log_path)
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        plt.plot(xs, ys, label=label, linewidth=1)
    plt.yscale("log")
    plt.xlabel("training step")
    plt.ylabel("loss (L1 + KL), log scale")
    plt.title("ACT on AlohaTransferCube: training loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_loss.png", dpi=150)


def plot_success() -> None:
    rows = list(csv.DictReader((OUT_DIR / "results.csv").open()))
    by_model: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    baseline = None
    for r in rows:
        if r["model"].startswith("official"):
            if baseline is None or int(r["n_episodes"]) > baseline[3]:
                baseline = (
                    float(r["success_pct"]),
                    float(r["ci95_low"]),
                    float(r["ci95_high"]),
                    int(r["n_episodes"]),
                )
            continue
        if int(r["n_episodes"]) != 50:
            continue  # the per-checkpoint curve uses the 50-episode evals only
        step = int(r["step"].rstrip("k")) * 1000
        by_model[r["model"]].append(
            (step, float(r["success_pct"]), float(r["ci95_low"]), float(r["ci95_high"]))
        )

    plt.figure(figsize=(7, 4))
    for model, pts in sorted(by_model.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        err = [[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]]
        plt.errorbar(xs, ys, yerr=err, marker="o", capsize=3, label=f"{model} (50 episodes, 95% CI)")
    if baseline:
        plt.axhline(baseline[0], color="black", linestyle="--", linewidth=1,
                    label=f"official checkpoint, {baseline[3]} episodes: {baseline[0]:.1f}%")
        plt.axhspan(baseline[1], baseline[2], color="black", alpha=0.08)
    plt.ylim(0, 100)
    plt.xlabel("training step")
    plt.ylabel("success rate (%)")
    plt.title("ACT on AlohaTransferCube: success per checkpoint")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "success_vs_steps.png", dpi=150)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    plot_loss()
    plot_success()
    print(f"wrote {OUT_DIR / 'training_loss.png'} and {OUT_DIR / 'success_vs_steps.png'}")


if __name__ == "__main__":
    main()
