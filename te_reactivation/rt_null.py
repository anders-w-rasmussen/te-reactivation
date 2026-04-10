"""
RT-length null model for distinguishing autonomous TE transcription
from readthrough.

Strategy:
1. Learn the empirical RT (reverse transcription) length distribution
   from reads at the 3' end of long genes. These are transcripts where
   we know the polyA site, so the cDNA length = distance from read 5'
   to the polyA site.

2. For a given TE family, construct a null footprint: if all reads
   terminating at the TE 3' end originated from far upstream, what
   would the coverage and 5' end profiles look like, shaped solely
   by the RT length distribution?

3. Compare observed footprint to null. Excess signal (especially
   5' ends enriched near the TE internal promoter) indicates
   autonomous transcription.
"""

import numpy as np
import pandas as pd


def learn_rt_length_distribution(
    bam_path: str,
    genes_bed_path: str,
    min_gene_length: int = 5000,
    polyA_window: int = 200,
    max_genes: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """
    Learn empirical RT length distribution from long genes.

    For each gene, find reads whose 3' end is within polyA_window of
    the gene's 3' end (polyA site). The RT length = distance from the
    read's 5' end to its 3' end (i.e., the aligned fragment length).

    Parameters
    ----------
    bam_path : str
        Indexed BAM file.
    genes_bed_path : str
        BED file with gene annotations (chrom, start, stop, name, strand).
        Or 5-col: chrom, start, stop, name/id, strand.
    min_gene_length : int
        Only use genes longer than this (ensures RT has room to drop off).
    polyA_window : int
        How close a read's 3' end must be to the gene 3' end to count.
    max_genes : int
        Subsample genes for speed.
    seed : int
        Random seed.

    Returns
    -------
    rt_lengths : np.ndarray
        Array of observed RT/cDNA lengths (in bp).
    """
    import pysam

    # Load genes
    genes = pd.read_csv(genes_bed_path, sep="\t", header=None, usecols=[0, 1, 2, 3, 4],
                         names=["chrom", "start", "stop", "name", "strand"],
                         dtype={"chrom": str, "start": int, "stop": int,
                                "name": str, "strand": str})
    genes["length"] = genes["stop"] - genes["start"]
    genes = genes[genes["length"] >= min_gene_length].copy()

    if len(genes) > max_genes:
        genes = genes.sample(n=max_genes, random_state=seed).reset_index(drop=True)

    print(f"Learning RT length distribution from {len(genes)} genes (>{min_gene_length}bp)...")

    bam = pysam.AlignmentFile(bam_path, "rb")
    bam_chroms = set(bam.references)

    rt_lengths = []

    for i, (_, gene) in enumerate(genes.iterrows()):
        if (i + 1) % 500 == 0:
            print(f"  Processing gene {i+1}/{len(genes)}...")

        chrom = gene["chrom"]
        if chrom not in bam_chroms:
            # Try with/without chr prefix
            if f"chr{chrom}" in bam_chroms:
                chrom = f"chr{chrom}"
            elif chrom.startswith("chr") and chrom[3:] in bam_chroms:
                chrom = chrom[3:]
            else:
                continue

        strand = gene["strand"]
        # 3' end of gene (polyA site)
        if strand == "+":
            polyA_pos = gene["stop"]
            fetch_start = max(0, polyA_pos - polyA_window)
            fetch_end = polyA_pos + polyA_window
        else:
            polyA_pos = gene["start"]
            fetch_start = max(0, polyA_pos - polyA_window)
            fetch_end = polyA_pos + polyA_window

        for read in bam.fetch(chrom, fetch_start, fetch_end):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            # Only sense-strand reads (same strand as gene)
            read_is_fwd = not read.is_reverse
            if strand == "+" and not read_is_fwd:
                continue
            if strand == "-" and read_is_fwd:
                continue

            # Read 3' end
            if read.is_reverse:
                read_3p = read.reference_start
            else:
                read_3p = read.reference_end
                if read_3p is None:
                    continue

            # Check if 3' end is near polyA
            if abs(read_3p - polyA_pos) > polyA_window:
                continue

            # RT length = actual sequenced read/cDNA length
            rt_len = read.query_length
            if rt_len is not None and rt_len > 0:
                rt_lengths.append(rt_len)

    bam.close()

    rt_lengths = np.array(rt_lengths)
    print(f"  Collected {len(rt_lengths)} RT lengths")
    if len(rt_lengths) > 0:
        print(f"  Median: {np.median(rt_lengths):.0f}bp, "
              f"Mean: {np.mean(rt_lengths):.0f}bp, "
              f"90th pctl: {np.percentile(rt_lengths, 90):.0f}bp")

    return rt_lengths


def simulate_null_footprint(
    rt_lengths: np.ndarray,
    te_length: int,
    n_bins: int = 100,
    flank_frac: float = 0.5,
    n_sim: int = 50000,
    seed: int = 42,
) -> dict:
    """
    Simulate the null footprint for a TE of given length.

    Null model: every read terminates at the TE 3' end (position = te_length),
    and originated from far upstream. The read's 5' end is determined by
    drawing an RT length from the empirical distribution.

    For each simulated read:
      - 3' end is at position te_length (the TE 3' end)
      - 5' end is at position (te_length - rt_length)
      - Coverage spans from 5' to 3'
      - We record the 5' end position

    Parameters
    ----------
    rt_lengths : np.ndarray
        Empirical RT length distribution (from learn_rt_length_distribution).
    te_length : int
        Length of the TE in bp.
    n_bins : int
        Number of bins across the TE body.
    flank_frac : float
        Flank size as fraction of TE length.
    n_sim : int
        Number of reads to simulate.
    seed : int
        Random seed.

    Returns
    -------
    dict with:
        "coverage": (total_bins,) expected coverage profile
        "fivep_density": (total_bins,) expected 5' end density
        "bin_centers": (total_bins,) relative positions
    """
    rng = np.random.default_rng(seed)

    flank_bp = int(te_length * flank_frac)
    flank_bins = int(n_bins * flank_frac)
    total_bins = flank_bins + n_bins + flank_bins

    # Bin edges in bp: [-flank_bp, te_length + flank_bp]
    bin_edges = np.linspace(-flank_bp, te_length + flank_bp, total_bins + 1)
    bin_centers_bp = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Convert to relative coords: 0 = TE 5', 1 = TE 3'
    bin_centers = bin_centers_bp / te_length

    coverage = np.zeros(total_bins, dtype=np.float64)
    fivep_density = np.zeros(total_bins, dtype=np.float64)

    # Sample RT lengths
    sampled_rt = rng.choice(rt_lengths, size=n_sim, replace=True)

    for rt_len in sampled_rt:
        # Read spans from (te_length - rt_len) to te_length
        read_5p = te_length - rt_len  # in bp coords
        read_3p = te_length

        # Add coverage: increment all bins between 5' and 3'
        in_range = (bin_centers_bp >= read_5p) & (bin_centers_bp <= read_3p)
        coverage[in_range] += 1

        # Add 5' end
        fivep_bin = np.searchsorted(bin_edges, read_5p) - 1
        if 0 <= fivep_bin < total_bins:
            fivep_density[fivep_bin] += 1

    # Normalize
    coverage /= n_sim
    fivep_density /= n_sim

    return {
        "coverage": coverage.astype(np.float32),
        "fivep_density": fivep_density.astype(np.float32),
        "bin_centers": bin_centers.astype(np.float32),
        "bin_centers_bp": bin_centers_bp.astype(np.float32),
        "te_length": te_length,
    }


def test_autonomous_transcription(
    observed_5p: np.ndarray,
    observed_coverage: np.ndarray,
    null_5p: np.ndarray,
    null_coverage: np.ndarray,
    bin_centers: np.ndarray,
    te_body_range: tuple = (0.0, 1.0),
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Test whether observed 5' end density inside the TE body exceeds
    the null model prediction.

    Scaling: the null is anchored to the observed coverage at the TE
    3' end (position ~1.0), since both models agree reads terminate
    there — they only disagree about where reads originated.

    Parameters
    ----------
    observed_5p : np.ndarray
        Observed 5' end density per bin (from footprint).
    observed_coverage : np.ndarray
        Observed sense coverage per bin (from footprint).
    null_5p : np.ndarray
        Null model 5' end density per bin (from simulate_null_footprint).
    null_coverage : np.ndarray
        Null model coverage per bin (from simulate_null_footprint).
    bin_centers : np.ndarray
        Bin center positions (relative coords).
    te_body_range : tuple
        (start, end) of TE body in relative coords.
    n_permutations : int
        Number of permutations for p-value.
    seed : int
        Random seed.

    Returns
    -------
    dict with:
        "enrichment_ratio": float — observed/expected inside TE body
        "excess_5p": np.ndarray — observed - expected per bin
        "p_value": float — permutation p-value
        "test_statistic": float
    """
    rng = np.random.default_rng(seed)

    te_mask = (bin_centers >= te_body_range[0]) & (bin_centers <= te_body_range[1])

    # Scale null to match observed at the TE 3' end.
    # Use a small window near position 1.0 for robustness.
    threep_mask = (bin_centers >= 0.85) & (bin_centers <= 1.0)
    obs_3p = observed_coverage[threep_mask].mean() if threep_mask.any() else 1.0
    null_3p = null_coverage[threep_mask].mean() if threep_mask.any() else 1.0
    if null_3p > 0:
        scale = obs_3p / null_3p
    else:
        scale = 1.0
    null_scaled = null_5p * scale

    # Test statistic: excess 5' ends inside TE body
    obs_te = observed_5p[te_mask].sum()
    null_te = null_scaled[te_mask].sum()

    if null_te > 0:
        enrichment = obs_te / null_te
    else:
        enrichment = float("inf") if obs_te > 0 else 1.0

    excess = observed_5p - null_scaled
    test_stat = excess[te_mask].sum()

    # Permutation test: shuffle bin labels and recompute
    all_excess = observed_5p - null_scaled
    n_te_bins = te_mask.sum()
    perm_stats = np.zeros(n_permutations)
    for p in range(n_permutations):
        perm_idx = rng.permutation(len(all_excess))
        perm_stats[p] = all_excess[perm_idx[:n_te_bins]].sum()

    p_value = (perm_stats >= test_stat).mean()

    return {
        "enrichment_ratio": float(enrichment),
        "excess_5p": excess,
        "null_scaled": null_scaled,
        "p_value": float(p_value),
        "test_statistic": float(test_stat),
        "obs_te_sum": float(obs_te),
        "null_te_sum": float(null_te),
    }


def plot_rt_null_analysis(
    observed_footprint,
    null_result: dict,
    test_result: dict,
    family_name: str,
    save_path: str = None,
):
    """
    Plot the RT null model analysis for a TE family.

    Panels:
    1. Observed vs null coverage
    2. Observed vs null 5' end density
    3. Excess 5' ends (observed - null)
    4. RT length distribution used
    """
    import matplotlib.pyplot as plt

    bc = null_result["bin_centers"]
    te_mask = (bc >= 0) & (bc <= 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Panel 1: Observed vs null coverage
    ax = axes[0]
    ax.axvspan(0, 1, alpha=0.06, color="#333")
    ax.axvline(0, color="#333", linewidth=0.8, alpha=0.4)
    ax.axvline(1, color="#333", linewidth=0.8, alpha=0.4)

    # Scale null coverage to match observed at TE 3' end
    obs_sense = observed_footprint.sense_mean
    obs_bc = observed_footprint.bin_centers
    null_cov = null_result["coverage"]
    threep_mask_obs = (obs_bc >= 0.85) & (obs_bc <= 1.0)
    threep_mask_null = (bc >= 0.85) & (bc <= 1.0)
    obs_3p = obs_sense[threep_mask_obs].mean() if threep_mask_obs.any() else 1.0
    null_3p = null_cov[threep_mask_null].mean() if threep_mask_null.any() else 1.0
    if null_3p > 0:
        cov_scale = obs_3p / null_3p
    else:
        cov_scale = 1.0

    ax.fill_between(obs_bc, obs_sense, alpha=0.3, color="#2166ac", label="Observed sense")
    ax.plot(obs_bc, obs_sense, color="#2166ac", linewidth=1.0)
    ax.plot(bc, null_cov * cov_scale, color="#d62728", linewidth=2, linestyle="--",
            label="RT null model")
    ax.set_ylabel("Coverage", fontsize=10)
    ax.set_title("Coverage: observed vs RT null", fontsize=11)
    ax.legend(fontsize=9)
    ax.text(0, 1.03, "5'", transform=ax.get_xaxis_transform(),
            fontsize=11, fontweight="bold", ha="center")
    ax.text(1, 1.03, "3'", transform=ax.get_xaxis_transform(),
            fontsize=11, fontweight="bold", ha="center")

    # Panel 2: Observed vs null 5' ends
    ax = axes[1]
    ax.axvspan(0, 1, alpha=0.06, color="#333")
    ax.axvline(0, color="#333", linewidth=0.8, alpha=0.4)
    ax.axvline(1, color="#333", linewidth=0.8, alpha=0.4)

    obs_5p = observed_footprint.sense_5p_mean
    ax.fill_between(obs_bc, obs_5p, alpha=0.3, color="#2166ac", label="Observed 5' ends")
    ax.plot(obs_bc, obs_5p, color="#2166ac", linewidth=1.0)
    ax.plot(bc, test_result["null_scaled"], color="#d62728", linewidth=2, linestyle="--",
            label="RT null (scaled to flanks)")
    ax.set_ylabel("5' end density", fontsize=10)
    ax.set_title("Sense 5' ends: observed vs RT null", fontsize=11)
    ax.legend(fontsize=9)

    # Panel 3: Excess
    ax = axes[2]
    ax.axvspan(0, 1, alpha=0.06, color="#333")
    ax.axvline(0, color="#333", linewidth=0.8, alpha=0.4)
    ax.axvline(1, color="#333", linewidth=0.8, alpha=0.4)

    excess = test_result["excess_5p"]
    # Interpolate excess to observed bin centers if needed
    if len(excess) != len(obs_bc):
        excess_interp = np.interp(obs_bc, bc, excess)
    else:
        excess_interp = excess

    ax.fill_between(obs_bc, excess_interp, alpha=0.4, color="#ff7f0e",
                     where=excess_interp > 0, label="Excess (autonomous signal)")
    ax.fill_between(obs_bc, excess_interp, alpha=0.2, color="#999",
                     where=excess_interp <= 0)
    ax.plot(obs_bc, excess_interp, color="#ff7f0e", linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")

    ax.set_ylabel("Excess 5' ends", fontsize=10)
    ax.set_xlabel("Relative position (0 = TE 5', 1 = TE 3')", fontsize=10)
    ax.set_title("Excess 5' ends over RT null (= autonomous transcription signal)", fontsize=11)
    ax.legend(fontsize=9)

    enrich = test_result["enrichment_ratio"]
    pval = test_result["p_value"]
    fig.suptitle(
        f"{family_name}   |   "
        f"Enrichment = {enrich:.2f}x   |   "
        f"p = {pval:.4f}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"RT null analysis plot saved to {save_path}")
    else:
        plt.show()
        plt.close()
