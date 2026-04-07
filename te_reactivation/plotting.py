"""
Visualization for TE reactivation analysis.

1. Footprint plots — sense/antisense coverage profile across the TE body
2. Volcano plot — P(reactivated) vs fold-change over background
3. Sense vs antisense scatter — dsRNA signature
4. ELBO convergence
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .data import TEFamilyData


def plot_footprints(
    data: TEFamilyData,
    results: dict,
    top_n: int = 6,
    n_bins: int = 50,
    save_path: str = None,
):
    """
    Plot aggregate sense/antisense coverage profiles across the TE body
    for the top reactivated families, analogous to ATAC-seq footprint plots.

    Each TE locus is normalized to relative position [0, 1], then
    sense and antisense counts are binned to show the coverage shape.
    """
    z = results["z_posterior"]
    names = results["family_names"]
    order = np.argsort(-z)

    # Select top families
    top_idx = [i for i in order if z[i] > 0.01][:top_n]
    if len(top_idx) == 0:
        print("No families with P(reactivated) > 0.01 to plot.")
        return

    n_plots = len(top_idx)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots), squeeze=False)

    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    for row, f_idx in enumerate(top_idx):
        ax = axes[row, 0]
        sense = data.sense_counts[f_idx]
        antisense = data.antisense_counts[f_idx]
        lengths = data.locus_lengths[f_idx]

        # We don't have per-base coverage, but we can show per-locus
        # counts as a bar chart sorted by locus position within TE body.
        # For a true footprint, we'd need BAM read positions.
        # Here we show the distribution of counts across loci as a proxy.

        # Sort loci by sense count to show expression landscape
        locus_order = np.argsort(-sense)
        n_loci = len(sense)
        x = np.arange(n_loci)

        ax.bar(x, sense[locus_order], width=1.0, alpha=0.7,
               color="#2166ac", label="Sense")
        ax.bar(x, -antisense[locus_order], width=1.0, alpha=0.7,
               color="#b2182b", label="Antisense")

        ax.set_title(f"{names[f_idx]}  —  P(reactivated) = {z[f_idx]:.3f}",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Read count")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.legend(loc="upper right", fontsize=8)

        if row == n_plots - 1:
            ax.set_xlabel("Loci (sorted by sense count)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Footprint plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_volcano(results: dict, threshold: float = 0.5, save_path: str = None):
    """
    Volcano-style plot: P(reactivated) vs estimated foreground magnitude
    for all TE families.
    """
    z = results["z_posterior"]
    names = results["family_names"]
    params = results["params"]

    # Foreground magnitude from learned log_fg_loc
    log_fg_loc = params.get("log_fg_loc", np.zeros(len(z)))
    fg_magnitude = np.exp(log_fg_loc)

    fig, ax = plt.subplots(figsize=(8, 6))

    active = z >= threshold
    silent = ~active

    ax.scatter(fg_magnitude[silent], z[silent],
               c="#999999", alpha=0.6, s=40, edgecolors="none", label="Silent")
    ax.scatter(fg_magnitude[active], z[active],
               c="#d62728", alpha=0.8, s=60, edgecolors="black", linewidths=0.5,
               label="Active")

    # Label active families
    for i in np.where(active)[0]:
        ax.annotate(names[i], (fg_magnitude[i], z[i]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    ax.axhline(threshold, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Foreground magnitude (exp(log_fg))", fontsize=11)
    ax.set_ylabel("P(reactivated)", fontsize=11)
    ax.set_title("TE Family Reactivation Volcano", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Volcano plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_sense_antisense(
    data: TEFamilyData,
    results: dict,
    threshold: float = 0.5,
    save_path: str = None,
):
    """
    Sense vs antisense mean counts per family, colored by reactivation status.
    Families on the diagonal have equal sense/antisense (strong dsRNA).
    """
    z = results["z_posterior"]
    names = results["family_names"]

    sense_means = np.array([s.mean() for s in data.sense_counts])
    antisense_means = np.array([a.mean() for a in data.antisense_counts])

    fig, ax = plt.subplots(figsize=(7, 7))

    active = z >= threshold
    silent = ~active

    ax.scatter(sense_means[silent], antisense_means[silent],
               c="#999999", alpha=0.5, s=40, edgecolors="none", label="Silent")
    ax.scatter(sense_means[active], antisense_means[active],
               c="#d62728", alpha=0.8, s=60, edgecolors="black", linewidths=0.5,
               label="Active")

    for i in np.where(active)[0]:
        ax.annotate(names[i], (sense_means[i], antisense_means[i]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    # Diagonal = equal sense/antisense
    lim = max(sense_means.max(), antisense_means.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean sense count per locus", fontsize=11)
    ax.set_ylabel("Mean antisense count per locus", fontsize=11)
    ax.set_title("Sense vs Antisense — dsRNA Signature", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sense/antisense plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_elbo(results: dict, save_path: str = None):
    """Plot ELBO loss convergence."""
    fig, ax = plt.subplots(figsize=(8, 4))

    losses = results["losses"]
    ax.plot(losses, color="#2166ac", alpha=0.3, linewidth=0.5)
    # Smoothed
    window = min(100, len(losses) // 10)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(losses)), smoothed,
                color="#2166ac", linewidth=1.5, label=f"Smoothed (w={window})")
        ax.legend(fontsize=9)

    ax.set_xlabel("SVI Step", fontsize=11)
    ax.set_ylabel("ELBO Loss", fontsize=11)
    ax.set_title("SVI Convergence", fontsize=13)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ELBO plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_summary(
    data: TEFamilyData,
    results: dict,
    threshold: float = 0.5,
    save_dir: str = None,
):
    """Generate all plots. If save_dir is given, saves PNGs there."""
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_footprints(data, results, save_path=os.path.join(save_dir, "footprints.png"))
        plot_volcano(results, threshold, save_path=os.path.join(save_dir, "volcano.png"))
        plot_sense_antisense(data, results, threshold, save_path=os.path.join(save_dir, "sense_antisense.png"))
        plot_elbo(results, save_path=os.path.join(save_dir, "elbo.png"))
        print(f"\nAll plots saved to {save_dir}/")
    else:
        plot_footprints(data, results)
        plot_volcano(results, threshold)
        plot_sense_antisense(data, results, threshold)
        plot_elbo(results)
