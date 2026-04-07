"""
Data loading: .bed file parsing and BAM read counting for TE loci.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass
class TEFamilyData:
    """Aggregated read counts per TE family for model input."""
    family_names: list[str]
    # Per-locus arrays, grouped by family
    sense_counts: list[np.ndarray]      # sense_counts[f] = array of counts per locus in family f
    antisense_counts: list[np.ndarray]  # antisense_counts[f] = array per locus
    locus_lengths: list[np.ndarray]     # lengths in bp per locus
    # Flank-based background (per-locus expected rate from local genomic context)
    flank_sense_rates: list[np.ndarray] | None = None   # reads per kb in flanks, per locus
    flank_antisense_rates: list[np.ndarray] | None = None
    n_families: int = 0


def load_bed(bed_path: str) -> pd.DataFrame:
    """Load a .bed file with columns: chrom, start, stop, TE_family, strand."""
    df = pd.read_csv(
        bed_path, sep="\t", header=None,
        names=["chrom", "start", "stop", "te_family", "strand"],
        dtype={"chrom": str, "start": int, "stop": int, "te_family": str, "strand": str},
    )
    df["length"] = df["stop"] - df["start"]
    return df


def _get_read_strand(read) -> str:
    """Get strand of a read using read1 convention for paired-end."""
    if read.is_reverse:
        strand = "-"
    else:
        strand = "+"
    if read.is_paired and read.is_read2:
        strand = "-" if strand == "+" else "+"
    return strand


def _count_region(bam, chrom, start, stop, te_strand):
    """Count sense/antisense reads in a region relative to TE strand."""
    s, a = 0, 0
    if start < 0:
        start = 0
    try:
        for read in bam.fetch(chrom, start, stop):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if _get_read_strand(read) == te_strand:
                s += 1
            else:
                a += 1
    except ValueError:
        pass  # region out of bounds
    return s, a


def count_reads_from_bam(
    bam_path: str,
    bed_df: pd.DataFrame,
    flank_size: int = 2000,
) -> pd.DataFrame:
    """
    Count sense/antisense reads in each TE locus AND in flanking regions.

    Flanks: `flank_size` bp upstream and downstream of each TE.
    Flank rates are normalized to reads per kb for use as local background.
    """
    import pysam

    bam = pysam.AlignmentFile(bam_path, "rb")
    sense_counts = []
    antisense_counts = []
    flank_sense_rates = []
    flank_antisense_rates = []

    total = len(bed_df)
    for i, (_, row) in enumerate(bed_df.iterrows()):
        if (i + 1) % 5000 == 0:
            print(f"  Counting reads: {i+1}/{total} loci...")

        te_strand = row["strand"]
        chrom = row["chrom"]
        te_start = row["start"]
        te_stop = row["stop"]

        # TE body counts
        s, a = _count_region(bam, chrom, te_start, te_stop, te_strand)
        sense_counts.append(s)
        antisense_counts.append(a)

        # Left flank
        left_start = te_start - flank_size
        left_s, left_a = _count_region(bam, chrom, left_start, te_start, te_strand)

        # Right flank
        right_stop = te_stop + flank_size
        right_s, right_a = _count_region(bam, chrom, te_stop, right_stop, te_strand)

        # Combined flank rate (reads per kb)
        total_flank_bp = flank_size * 2
        flank_s_rate = (left_s + right_s) / (total_flank_bp / 1000) if total_flank_bp > 0 else 0
        flank_a_rate = (left_a + right_a) / (total_flank_bp / 1000) if total_flank_bp > 0 else 0

        flank_sense_rates.append(flank_s_rate)
        flank_antisense_rates.append(flank_a_rate)

    bam.close()
    bed_df = bed_df.copy()
    bed_df["sense_count"] = sense_counts
    bed_df["antisense_count"] = antisense_counts
    bed_df["flank_sense_rate"] = flank_sense_rates
    bed_df["flank_antisense_rate"] = flank_antisense_rates
    return bed_df


def aggregate_by_family(bed_df: pd.DataFrame) -> TEFamilyData:
    """Group locus-level data by TE family for model input."""
    families = sorted(bed_df["te_family"].unique())
    sense = []
    antisense = []
    lengths = []
    flank_s = []
    flank_a = []

    has_flanks = "flank_sense_rate" in bed_df.columns

    for fam in families:
        sub = bed_df[bed_df["te_family"] == fam]
        sense.append(sub["sense_count"].values.astype(np.float32))
        antisense.append(sub["antisense_count"].values.astype(np.float32))
        lengths.append(sub["length"].values.astype(np.float32))
        if has_flanks:
            flank_s.append(sub["flank_sense_rate"].values.astype(np.float32))
            flank_a.append(sub["flank_antisense_rate"].values.astype(np.float32))

    return TEFamilyData(
        family_names=families,
        sense_counts=sense,
        antisense_counts=antisense,
        locus_lengths=lengths,
        flank_sense_rates=flank_s if has_flanks else None,
        flank_antisense_rates=flank_a if has_flanks else None,
        n_families=len(families),
    )


def extract_positional_profiles(
    bam_path: str,
    bed_df: pd.DataFrame,
    n_bins: int = 50,
) -> dict[str, dict]:
    """
    Extract positional read profiles across the TE body for each family.

    For each read, maps its start position to a relative coordinate [0, 1]
    within the TE locus (0 = 5' end, 1 = 3' end, respecting TE strand).
    Aggregates across all loci per family.

    Returns dict: family_name -> {"sense": array[n_bins], "antisense": array[n_bins], "n_loci": int}
    """
    import pysam

    bam = pysam.AlignmentFile(bam_path, "rb")
    bins = np.linspace(0, 1, n_bins + 1)

    # Accumulate per-family
    families = bed_df["te_family"].unique()
    profiles = {fam: {"sense": np.zeros(n_bins), "antisense": np.zeros(n_bins), "n_loci": 0}
                for fam in families}

    for _, row in bed_df.iterrows():
        fam = row["te_family"]
        te_start = row["start"]
        te_stop = row["stop"]
        te_strand = row["strand"]
        te_len = te_stop - te_start
        if te_len <= 0:
            continue

        profiles[fam]["n_loci"] += 1

        for read in bam.fetch(row["chrom"], te_start, te_stop):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            # Read strand (read1 convention for paired-end)
            if read.is_reverse:
                read_strand = "-"
            else:
                read_strand = "+"
            if read.is_paired and read.is_read2:
                read_strand = "-" if read_strand == "+" else "+"

            # Read start position relative to TE
            read_pos = read.reference_start
            rel_pos = (read_pos - te_start) / te_len

            # Flip for minus-strand TEs so 0 = 5' end
            if te_strand == "-":
                rel_pos = 1.0 - rel_pos

            rel_pos = np.clip(rel_pos, 0, 1 - 1e-9)
            bin_idx = int(rel_pos * n_bins)

            if read_strand == te_strand:
                profiles[fam]["sense"][bin_idx] += 1
            else:
                profiles[fam]["antisense"][bin_idx] += 1

    bam.close()

    # Normalize by number of loci so profiles are comparable across families
    for fam in profiles:
        n = profiles[fam]["n_loci"]
        if n > 0:
            profiles[fam]["sense"] /= n
            profiles[fam]["antisense"] /= n

    return profiles


def generate_synthetic_data(
    n_families: int = 20,
    n_loci_per_family: int = 50,
    frac_reactivated: float = 0.3,
    bg_rate: float = 2.0,
    fg_boost: float = 15.0,
    antisense_ratio_active: float = 0.6,
    seed: int = 42,
) -> tuple[TEFamilyData, np.ndarray]:
    """
    Generate synthetic TE family data with known ground truth.

    Returns (data, ground_truth_z) where ground_truth_z[f] = 1 if family f is reactivated.

    Background: varies per locus (simulates different genomic contexts).
    Flanks match background (read-through affects TE body and flanks equally).
    Foreground: excess expression in TE body ONLY (not flanks).
    """
    rng = np.random.default_rng(seed)

    n_active = int(n_families * frac_reactivated)
    z_true = np.zeros(n_families)
    active_idx = rng.choice(n_families, n_active, replace=False)
    z_true[active_idx] = 1.0

    families = [f"TE_family_{i:03d}" for i in range(n_families)]
    sense_counts = []
    antisense_counts = []
    locus_lengths = []
    flank_sense_rates = []
    flank_antisense_rates = []

    for f in range(n_families):
        n_loci = rng.integers(max(10, n_loci_per_family // 2), n_loci_per_family * 2)
        lengths = rng.integers(200, 8000, size=n_loci).astype(np.float32)

        # Per-locus background rate (varies — some loci near active genes, some not)
        locus_bg_rate = rng.gamma(2.0, bg_rate / 2.0, size=n_loci)

        # Background counts in TE body
        bg_sense = rng.poisson(locus_bg_rate)
        bg_antisense = rng.poisson(locus_bg_rate * 0.8)

        # Flank rates match background (read-through = same rate in flanks and TE body)
        # Add some noise to simulate imperfect flank estimation
        flank_s = (locus_bg_rate + rng.normal(0, 0.3, size=n_loci)).clip(0.1)
        flank_a = (locus_bg_rate * 0.8 + rng.normal(0, 0.3, size=n_loci)).clip(0.1)

        if z_true[f] == 1.0:
            # Foreground: excess in TE body only, NOT in flanks
            locus_activity = rng.beta(2, 3, size=n_loci)
            fg_sense = rng.poisson(fg_boost * locus_activity)
            fg_antisense = rng.poisson(fg_boost * antisense_ratio_active * locus_activity)
            sense = (bg_sense + fg_sense).astype(np.float32)
            antisense = (bg_antisense + fg_antisense).astype(np.float32)
        else:
            sense = bg_sense.astype(np.float32)
            antisense = bg_antisense.astype(np.float32)

        sense_counts.append(sense)
        antisense_counts.append(antisense)
        locus_lengths.append(lengths)
        flank_sense_rates.append(flank_s.astype(np.float32))
        flank_antisense_rates.append(flank_a.astype(np.float32))

    data = TEFamilyData(
        family_names=families,
        sense_counts=sense_counts,
        antisense_counts=antisense_counts,
        locus_lengths=locus_lengths,
        flank_sense_rates=flank_sense_rates,
        flank_antisense_rates=flank_antisense_rates,
        n_families=n_families,
    )
    return data, z_true
