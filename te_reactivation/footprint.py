"""
TE transcription footprinting.

Computes aggregate sense/antisense coverage profiles across TE bodies
and flanking regions, with background estimation via local regression
and antisense-as-null.

Input: strand-specific bigWig files + BED annotation.
Output: per-family footprint plots.

bigWig generation (paired-end, fr-firststrand library):
    # Forward-strand coverage (reads on + strand)
    bamCoverage -b aligned.bam -o forward.bw --filterRNAstrand forward --binSize 1 --normalizeUsing CPM
    # Reverse-strand coverage (reads on - strand)
    bamCoverage -b aligned.bam -o reverse.bw --filterRNAstrand reverse --binSize 1 --normalizeUsing CPM

For unstranded libraries, skip sense/antisense decomposition.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class FamilyFootprint:
    """Aggregate footprint for one TE family."""
    family_name: str
    n_loci: int
    bin_centers: np.ndarray        # relative positions, 0=5' end, 1=3' end
    sense_mean: np.ndarray         # mean sense coverage per bin
    sense_median: np.ndarray
    antisense_mean: np.ndarray
    antisense_median: np.ndarray
    bg_sense: np.ndarray           # estimated background (local regression)
    bg_antisense: np.ndarray
    # Per-locus matrix for variability (optional, can be large)
    sense_matrix: np.ndarray | None = None   # (n_loci, n_bins)
    antisense_matrix: np.ndarray | None = None


def load_bed(bed_path: str) -> pd.DataFrame:
    """Load 5-column BED: chrom, start, stop, TE_family, strand."""
    return pd.read_csv(
        bed_path, sep="\t", header=None,
        names=["chrom", "start", "stop", "te_family", "strand"],
        dtype={"chrom": str, "start": int, "stop": int,
               "te_family": str, "strand": str},
    )


def extract_coverage(
    bw_forward_path: str,
    bw_reverse_path: str,
    bed_df: pd.DataFrame,
    n_bins: int = 100,
    flank_frac: float = 0.5,
    min_te_length: int = 50,
) -> dict[str, dict]:
    """
    Extract per-base coverage for each TE locus + flanks from bigWig files,
    resize to fixed bins, orient to TE strand, split into sense/antisense.

    Returns dict: family -> {
        "sense": (n_loci, total_bins) array,
        "antisense": (n_loci, total_bins) array,
        "bin_centers": array,
    }
    """
    import pyBigWig

    bw_fwd = pyBigWig.open(bw_forward_path)
    bw_rev = pyBigWig.open(bw_reverse_path)

    # Total bins: flank + TE body + flank
    flank_bins = int(n_bins * flank_frac)
    total_bins = flank_bins + n_bins + flank_bins
    range_start = -flank_frac
    range_end = 1.0 + flank_frac
    bin_centers = np.linspace(range_start, range_end, total_bins)

    # Pre-group by family
    families = bed_df["te_family"].unique()
    family_data = {fam: {"sense": [], "antisense": []} for fam in families}

    chrom_sizes = dict(zip(bw_fwd.chroms().keys(), bw_fwd.chroms().values()))

    for _, row in bed_df.iterrows():
        te_len = row["stop"] - row["start"]
        if te_len < min_te_length:
            continue

        chrom = row["chrom"]
        if chrom not in chrom_sizes:
            continue

        chrom_len = chrom_sizes[chrom]
        flank_bp = int(te_len * flank_frac)

        # Extended region
        ext_start = max(0, row["start"] - flank_bp)
        ext_stop = min(chrom_len, row["stop"] + flank_bp)

        # Get per-base coverage from both strands
        try:
            fwd_vals = np.array(bw_fwd.values(chrom, ext_start, ext_stop), dtype=np.float32)
            rev_vals = np.array(bw_rev.values(chrom, ext_start, ext_stop), dtype=np.float32)
        except Exception:
            continue

        # Replace NaN with 0
        fwd_vals = np.nan_to_num(fwd_vals, 0.0)
        rev_vals = np.nan_to_num(rev_vals, 0.0)

        # Orient to TE strand: sense = same as TE, antisense = opposite
        if row["strand"] == "+":
            sense_vals = fwd_vals
            antisense_vals = rev_vals
        else:
            # Flip both strand assignment AND spatial orientation
            sense_vals = rev_vals[::-1]
            antisense_vals = fwd_vals[::-1]

        # Resize to fixed bins using mean pooling
        sense_binned = _resize_to_bins(sense_vals, total_bins)
        antisense_binned = _resize_to_bins(antisense_vals, total_bins)

        family_data[row["te_family"]]["sense"].append(sense_binned)
        family_data[row["te_family"]]["antisense"].append(antisense_binned)

    bw_fwd.close()
    bw_rev.close()

    # Stack into matrices
    result = {}
    for fam in families:
        s_list = family_data[fam]["sense"]
        a_list = family_data[fam]["antisense"]
        if len(s_list) == 0:
            continue
        result[fam] = {
            "sense": np.stack(s_list),       # (n_loci, total_bins)
            "antisense": np.stack(a_list),
            "bin_centers": bin_centers,
        }

    return result


def _resize_to_bins(vals: np.ndarray, n_bins: int) -> np.ndarray:
    """Resize a per-base array to n_bins by mean-pooling."""
    n = len(vals)
    if n == 0:
        return np.zeros(n_bins, dtype=np.float32)
    if n == n_bins:
        return vals
    # Use reshape if evenly divisible, otherwise interpolate
    if n >= n_bins:
        # Trim to make evenly divisible, then reshape + mean
        trim = n - (n % n_bins)
        return vals[:trim].reshape(n_bins, -1).mean(axis=1).astype(np.float32)
    else:
        # Upsample via linear interpolation
        x_old = np.linspace(0, 1, n)
        x_new = np.linspace(0, 1, n_bins)
        return np.interp(x_new, x_old, vals).astype(np.float32)


def compute_footprints(
    family_coverages: dict[str, dict],
    flank_frac: float = 0.5,
    bg_smooth_window: int = 15,
) -> dict[str, FamilyFootprint]:
    """
    Compute aggregate footprints with background estimation for each family.

    Background estimation:
    1. Local regression: smooth the signal in flank regions, interpolate
       across the TE body to get expected level assuming read-through.
    2. Antisense as null: the antisense signal serves as a second
       background estimate — autonomous transcription is sense-biased.
    """
    footprints = {}

    for fam, cov in family_coverages.items():
        sense_mat = cov["sense"]       # (n_loci, n_bins)
        antisense_mat = cov["antisense"]
        bin_centers = cov["bin_centers"]
        n_loci = sense_mat.shape[0]

        # Aggregate
        sense_mean = sense_mat.mean(axis=0)
        sense_median = np.median(sense_mat, axis=0)
        antisense_mean = antisense_mat.mean(axis=0)
        antisense_median = np.median(antisense_mat, axis=0)

        # Background via local regression through flanks
        bg_sense = _estimate_background(sense_mean, bin_centers, flank_frac, bg_smooth_window)
        bg_antisense = _estimate_background(antisense_mean, bin_centers, flank_frac, bg_smooth_window)

        footprints[fam] = FamilyFootprint(
            family_name=fam,
            n_loci=n_loci,
            bin_centers=bin_centers,
            sense_mean=sense_mean,
            sense_median=sense_median,
            antisense_mean=antisense_mean,
            antisense_median=antisense_median,
            bg_sense=bg_sense,
            bg_antisense=bg_antisense,
        )

    return footprints


def _estimate_background(
    signal: np.ndarray,
    bin_centers: np.ndarray,
    flank_frac: float,
    smooth_window: int,
) -> np.ndarray:
    """
    Estimate background by fitting through flank regions and interpolating
    across the TE body.

    1. Smooth the entire signal with a rolling mean
    2. Mask the TE body [0, 1]
    3. Interpolate the TE body region from the smoothed flank values
    """
    # Smooth everything first
    kernel = np.ones(smooth_window) / smooth_window
    smoothed = np.convolve(signal, kernel, mode="same")

    # Identify flank bins vs TE body bins
    flank_mask = (bin_centers < 0) | (bin_centers > 1)
    te_mask = (bin_centers >= 0) & (bin_centers <= 1)

    # Use flank values as anchors, interpolate across TE body
    flank_x = bin_centers[flank_mask]
    flank_y = smoothed[flank_mask]

    # Interpolate across full range using flank anchors
    bg = np.interp(bin_centers, flank_x, flank_y)

    return bg.astype(np.float32)


def plot_footprints(
    footprints: dict[str, FamilyFootprint],
    save_dir: str = None,
    top_n: int = None,
    sort_by: str = "enrichment",
):
    """
    Plot footprint for each family.

    Each plot has 3 panels:
    1. Sense footprint: observed vs background, with excess shaded
    2. Antisense footprint: observed vs background
    3. Sense - antisense difference (autonomous transcription signal)
    """
    import matplotlib.pyplot as plt
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Sort families by enrichment score
    scored = []
    for fam, fp in footprints.items():
        te_mask = (fp.bin_centers >= 0) & (fp.bin_centers <= 1)
        excess = np.maximum(fp.sense_mean[te_mask] - fp.bg_sense[te_mask], 0).mean()
        scored.append((fam, excess))
    scored.sort(key=lambda x: -x[1])

    if top_n:
        scored = scored[:top_n]

    for fam, enrichment in scored:
        fp = footprints[fam]
        bc = fp.bin_centers
        te_mask = (bc >= 0) & (bc <= 1)

        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # ---- Panel 1: Sense ----
        ax = axes[0]
        ax.axvspan(0, 1, alpha=0.06, color="#333")
        ax.axvline(0, color="#333", linewidth=0.8, alpha=0.4)
        ax.axvline(1, color="#333", linewidth=0.8, alpha=0.4)

        ax.fill_between(bc, fp.sense_mean, alpha=0.4, color="#2166ac", label="Observed sense")
        ax.plot(bc, fp.sense_mean, color="#2166ac", linewidth=1.0)
        ax.plot(bc, fp.bg_sense, color="#2166ac", linewidth=2, linestyle="--",
                label="Expected bg (flank interpolation)")

        # Shade excess
        ax.fill_between(bc[te_mask],
                        fp.bg_sense[te_mask],
                        np.maximum(fp.sense_mean[te_mask], fp.bg_sense[te_mask]),
                        alpha=0.35, color="#ff7f0e", label="Excess over background")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Mean coverage (CPM)", fontsize=10)
        ax.set_title("Sense strand", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.text(0.0, 1.03, "5'", transform=ax.get_xaxis_transform(),
                fontsize=11, fontweight="bold", ha="center")
        ax.text(1.0, 1.03, "3'", transform=ax.get_xaxis_transform(),
                fontsize=11, fontweight="bold", ha="center")

        # ---- Panel 2: Antisense ----
        ax = axes[1]
        ax.axvspan(0, 1, alpha=0.06, color="#333")
        ax.axvline(0, color="#333", linewidth=0.8, alpha=0.4)
        ax.axvline(1, color="#333", linewidth=0.8, alpha=0.4)

        ax.fill_between(bc, fp.antisense_mean, alpha=0.4, color="#b2182b", label="Observed antisense")
        ax.plot(bc, fp.antisense_mean, color="#b2182b", linewidth=1.0)
        ax.plot(bc, fp.bg_antisense, color="#b2182b", linewidth=2, linestyle="--",
                label="Expected bg (flank interpolation)")

        ax.fill_between(bc[te_mask],
                        fp.bg_antisense[te_mask],
                        np.maximum(fp.antisense_mean[te_mask], fp.bg_antisense[te_mask]),
                        alpha=0.35, color="#ff7f0e", label="Excess over background")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Mean coverage (CPM)", fontsize=10)
        ax.set_title("Antisense strand", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

        # ---- Panel 3: Sense - Antisense (autonomous signal) ----
        ax = axes[2]
        ax.axvspan(0, 1, alpha=0.06, color="#333")
        ax.axvline(0, color="#333", linewidth=0.8, alpha=0.4)
        ax.axvline(1, color="#333", linewidth=0.8, alpha=0.4)

        diff = fp.sense_mean - fp.antisense_mean
        bg_diff = fp.bg_sense - fp.bg_antisense

        ax.fill_between(bc, diff, alpha=0.4, color="#7570b3", label="Sense - Antisense")
        ax.plot(bc, diff, color="#7570b3", linewidth=1.0)
        ax.plot(bc, bg_diff, color="#7570b3", linewidth=2, linestyle="--",
                label="Expected (from flanks)")

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Sense - Antisense (CPM)", fontsize=10)
        ax.set_xlabel("Relative position (0 = TE 5', 1 = TE 3')", fontsize=10)
        ax.set_title("Strand asymmetry (positive = sense-biased = autonomous)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

        # Compute summary stats
        te_sense = fp.sense_mean[te_mask].mean()
        flank_sense = fp.sense_mean[~te_mask].mean() if (~te_mask).any() else 0
        te_as = fp.antisense_mean[te_mask].mean()
        fc = te_sense / (flank_sense + 1e-10)
        as_ratio = te_as / (te_sense + 1e-10)

        fig.suptitle(
            f"{fam}   |   {fp.n_loci} loci   |   "
            f"FC(sense) = {fc:.1f}x   |   AS/S = {as_ratio:.2f}",
            fontsize=13, fontweight="bold", y=1.01,
        )

        plt.tight_layout()

        if save_dir:
            safe = fam.replace("/", "_").replace(" ", "_")
            path = os.path.join(save_dir, f"footprint_{safe}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()

    if save_dir:
        print(f"Footprints saved to {save_dir}/ ({len(scored)} families)")
