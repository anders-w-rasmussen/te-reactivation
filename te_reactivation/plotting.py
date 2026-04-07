"""
Visualization for TE reactivation analysis.

1. Per-family footprint panels with bg/fg overlay for ALL families
2. Volcano plot — P(reactivated) vs fold-change over background
3. Sense vs antisense scatter — dsRNA signature
4. ELBO convergence
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .data import TEFamilyData


def plot_family_footprints(
    data: TEFamilyData,
    results: dict,
    profiles: dict = None,
    flank_profiles: dict = None,
    save_dir: str = None,
    n_bins: int = 50,
):
    """
    Plot footprint panels for EACH TE family.

    Each family gets a PNG with up to 3 panels:
      1. Positional footprint with bg/fg overlay (if BAM profiles available)
      2. TE body vs flank per locus (bar chart)
      3. Locus-level w_fi distribution (if hierarchical model)
    """
    z = results["z_posterior"]
    names = results["family_names"]
    params = results["params"]

    if save_dir:
        fp_dir = os.path.join(save_dir, "family_footprints")
        os.makedirs(fp_dir, exist_ok=True)

    order = np.argsort(-z)

    for f_idx in order:
        fam_name = names[f_idx]
        sense = data.sense_counts[f_idx]
        antisense = data.antisense_counts[f_idx]
        lengths_kb = data.locus_lengths[f_idx] / 1000.0
        n_loci = len(sense)

        has_flanks = data.flank_sense_rates is not None
        has_profiles = profiles is not None and fam_name in profiles
        has_w = f"log_w_loc_{f_idx}" in params

        n_rows = 1 + int(has_profiles) + int(has_w)
        fig, axes = plt.subplots(n_rows, 1, figsize=(12, 4 * n_rows))
        if n_rows == 1:
            axes = [axes]

        row = 0

        # ---- Panel 1: Positional footprint with bg/fg overlay ----
        if has_profiles:
            ax = axes[row]
            prof = profiles[fam_name]
            sense_prof = prof["sense"]
            antisense_prof = prof["antisense"]
            n_prof_loci = prof["n_loci"]
            bin_centers = np.linspace(0, 1, len(sense_prof) + 1)
            bin_centers = (bin_centers[:-1] + bin_centers[1:]) / 2

            # Total signal (what we observe)
            ax.fill_between(bin_centers, sense_prof, alpha=0.4,
                            color="#2166ac", label="Observed sense")
            ax.fill_between(bin_centers, -antisense_prof, alpha=0.4,
                            color="#b2182b", label="Observed antisense")
            ax.plot(bin_centers, sense_prof, color="#2166ac", linewidth=1.0)
            ax.plot(bin_centers, -antisense_prof, color="#b2182b", linewidth=1.0)

            # Estimated background level from flanks (horizontal band)
            if has_flanks:
                flank_s = data.flank_sense_rates[f_idx]
                flank_a = data.flank_antisense_rates[f_idx]
                # Profile units: total reads in bin / n_loci (from extract_positional_profiles)
                # If background is uniform across TE body, expected per-bin per-locus:
                #   mean_flank_rate(reads/kb) * mean_length(kb) / n_bins
                mean_len_kb = lengths_kb.mean()
                n_profile_bins = len(sense_prof)
                bg_sense_level = flank_s.mean() * mean_len_kb / n_profile_bins
                bg_antisense_level = flank_a.mean() * mean_len_kb / n_profile_bins

                ax.axhline(bg_sense_level, color="#2166ac", linestyle=":",
                           linewidth=2, alpha=0.7, label="Flank bg (sense)")
                ax.axhline(-bg_antisense_level, color="#b2182b", linestyle=":",
                           linewidth=2, alpha=0.7, label="Flank bg (antisense)")

                # Shade the foreground region (above background)
                ax.fill_between(bin_centers,
                                bg_sense_level,
                                np.maximum(sense_prof, bg_sense_level),
                                alpha=0.25, color="#ff7f0e",
                                label="Foreground (excess)")

            ax.axhline(0, color="black", linewidth=0.5)
            ax.text(0.02, 0.95, "5'", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="top", color="#333")
            ax.text(0.95, 0.95, "3'", transform=ax.transAxes,
                    fontsize=13, fontweight="bold", va="top", color="#333")
            ax.set_ylabel("Mean reads per locus", fontsize=10)
            ax.set_xlabel("Relative position within TE body", fontsize=10)
            ax.set_title("Positional Footprint (observed = bg + fg)", fontsize=11)
            ax.legend(fontsize=7, loc="upper right", ncol=2)
            row += 1

        # ---- Panel 2: TE body vs flank per locus ----
        ax = axes[row]

        te_sense_rate = sense / (lengths_kb + 1e-6)
        te_antisense_rate = antisense / (lengths_kb + 1e-6)
        total_te_rate = te_sense_rate + te_antisense_rate
        locus_order = np.argsort(-total_te_rate)
        x = np.arange(n_loci)

        ax.bar(x, te_sense_rate[locus_order], width=1.0, alpha=0.7,
               color="#2166ac", label="TE sense")
        ax.bar(x, -te_antisense_rate[locus_order], width=1.0, alpha=0.7,
               color="#b2182b", label="TE antisense")

        if has_flanks:
            flank_s = data.flank_sense_rates[f_idx]
            flank_a = data.flank_antisense_rates[f_idx]

            ax.step(x, flank_s[locus_order], where="mid",
                    color="#2166ac", linewidth=1.5, linestyle="--",
                    alpha=0.8, label="Flank sense (bg)")
            ax.step(x, -flank_a[locus_order], where="mid",
                    color="#b2182b", linewidth=1.5, linestyle="--",
                    alpha=0.8, label="Flank antisense (bg)")

            # Show foreground (excess over flank) as orange overlay
            excess_s = np.maximum(te_sense_rate[locus_order] - flank_s[locus_order], 0)
            ax.bar(x, excess_s, width=1.0, alpha=0.4,
                   color="#ff7f0e", label="Foreground (excess)")

            mean_te_s = te_sense_rate.mean()
            mean_flank_s = flank_s.mean()
            fc = mean_te_s / (mean_flank_s + 1e-6)
            ax.text(0.98, 0.02, f"FC(sense) = {fc:.1f}x",
                    transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Reads per kb", fontsize=10)
        ax.set_xlabel("Loci (sorted by expression)", fontsize=10)
        ax.set_title("TE Body (solid) vs Flank Background (dashed), Foreground (orange)",
                      fontsize=11)
        ax.legend(fontsize=7, loc="upper right", ncol=2)
        row += 1

        # ---- Panel 3: Locus-level w_fi distribution ----
        if has_w:
            ax = axes[row]
            log_w_loc = params[f"log_w_loc_{f_idx}"]
            w_vals = np.exp(log_w_loc)
            sigma_w_val = params.get("sigma_w_loc", np.zeros(len(z)))[f_idx]

            ax.hist(w_vals, bins=30, color="#7570b3", alpha=0.7, edgecolor="black",
                    linewidth=0.5)
            ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="w = 1 (family mean)")
            ax.axvline(np.median(w_vals), color="#d95f02", linestyle="-", linewidth=1.5,
                       label=f"Median = {np.median(w_vals):.2f}")

            ax.set_xlabel("Locus scale w (exp(log_w))", fontsize=10)
            ax.set_ylabel("Count", fontsize=10)
            ax.set_title(f"Locus Heterogeneity (sigma_w = {sigma_w_val:.3f})", fontsize=11)
            ax.legend(fontsize=8)

        # Overall title
        sigma_w_str = ""
        if "sigma_w_loc" in params:
            sigma_w_str = f"   |   sigma_w = {params['sigma_w_loc'][f_idx]:.3f}"

        fig.suptitle(
            f"{fam_name}   |   P(reactivated) = {z[f_idx]:.3f}   |   "
            f"{n_loci} loci{sigma_w_str}",
            fontsize=13, fontweight="bold", y=1.02,
        )

        plt.tight_layout()

        if save_dir:
            safe_name = fam_name.replace("/", "_").replace(" ", "_")
            path = os.path.join(fp_dir, f"footprint_{safe_name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()

    if save_dir:
        print(f"  Per-family footprints saved to {fp_dir}/ ({len(names)} files)")


def plot_volcano(results: dict, threshold: float = 0.5, save_path: str = None):
    """Volcano: P(reactivated) vs foreground magnitude."""
    z = results["z_posterior"]
    names = results["family_names"]
    params = results["params"]

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
        print(f"  Volcano plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_sense_antisense(
    data: TEFamilyData,
    results: dict,
    threshold: float = 0.5,
    save_path: str = None,
):
    """Sense vs antisense mean counts per family."""
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
        print(f"  Sense/antisense plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_elbo(results: dict, save_path: str = None):
    """ELBO convergence."""
    fig, ax = plt.subplots(figsize=(8, 4))

    losses = results["losses"]
    ax.plot(losses, color="#2166ac", alpha=0.3, linewidth=0.5)
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
        print(f"  ELBO plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_summary(
    data: TEFamilyData,
    results: dict,
    threshold: float = 0.5,
    save_dir: str = None,
    profiles: dict = None,
):
    """Generate all plots including per-family footprints."""

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print("Generating plots...")
        plot_family_footprints(data, results, profiles=profiles, save_dir=save_dir)
        plot_volcano(results, threshold, save_path=os.path.join(save_dir, "volcano.png"))
        plot_sense_antisense(data, results, threshold,
                             save_path=os.path.join(save_dir, "sense_antisense.png"))
        plot_elbo(results, save_path=os.path.join(save_dir, "elbo.png"))
        print(f"\nAll plots saved to {save_dir}/")
    else:
        plot_family_footprints(data, results, profiles=profiles)
        plot_volcano(results, threshold)
        plot_sense_antisense(data, results, threshold)
        plot_elbo(results)
