"""Reproduce CECL mechanism figures; requires optional matplotlib."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit, ndtr, ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pereval.tasks.cecl.generator import default_paths, path_losses


def main():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )
    colors = ["#2479A8", "#D36B28", "#A02C9D"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle(
        "CECL: from economic conditions to lifetime loss", fontsize=19, weight="bold"
    )
    u = np.linspace(2, 12, 200)
    ax = axes[0, 0]
    for h, color in zip([-0.05, 0.03, 0.11], colors):
        mean = expit(-4.25 + 0.45 * (u - 5) - 0.325 * ((h - 0.03) / 0.04))
        ax.plot(u, 100 * mean, color=color, label=f"House-price growth {h:.0%}")
    ax.set(
        title="1  Economic conditions move the mean",
        xlabel="Unemployment (%)",
        ylabel="Mean quarterly default probability (%)",
    )
    ax.legend(fontsize=9)
    ax = axes[0, 1]
    z = np.linspace(-3, 3, 300)
    for rho, color in zip([0, 0.02, 0.10], colors):
        ax.plot(
            z,
            100 * ndtr((ndtri(0.03) + np.sqrt(rho) * z) / np.sqrt(1 - rho)),
            color=color,
            label=f"rho = {rho:.2f}",
        )
    ax.set(
        title="2  Stronger shocks spread default rates",
        xlabel="Systemic shock (z)",
        ylabel="Quarterly default fraction (%)",
    )
    ax.legend(fontsize=9)
    ax = axes[1, 0]
    for phi, color in zip([0, 0.6], colors):
        pd = default_paths(
            np.full(40, 0.03), 1, np.random.default_rng(41), persistence=phi
        )[0]
        ax.plot(
            np.arange(1, 41), 100 * pd, color=color, label=f"Persistence = {phi:.1f}"
        )
    ax.axhline(
        3, color="#555555", linestyle="--", linewidth=1, label="Marginal mean = 3%"
    )
    ax.set(
        title="3  Persistence clusters good and bad quarters",
        xlabel="Quarter",
        ylabel="Quarterly default fraction (%)",
    )
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    rates = np.tile([0.03, 0.45, 0.02], (40, 1))
    for phi, color in zip([0, 0.6], colors):
        pd = default_paths(
            rates[:, 0], 100000, np.random.default_rng(20260925), persistence=phi
        )
        losses = path_losses(100, rates, pd)
        ax.hist(
            losses,
            bins=np.linspace(6, 26, 100),
            density=True,
            histtype="step",
            linewidth=1.8,
            color=color,
            label=f"Persistence {phi:.1f}; mean {losses.mean():.2f}%",
        )
        ax.axvline(losses.mean(), color=color, linestyle="--", linewidth=1)
        print(
            f"phi={phi}: mean={losses.mean():.4f}%, 95% interval={np.quantile(losses, [0.025, 0.975])}"
        )
    ax.set(
        title="4  Full paths determine lifetime loss",
        xlabel="Lifetime loss (% of initial principal)",
        ylabel="Probability density",
    )
    ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=0.15)
    out = Path(__file__).resolve().parents[1] / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(out / f"cecl-functional-form.{extension}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
