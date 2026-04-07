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
    n_families: int


def load_bed(bed_path: str) -> pd.DataFrame:
    """Load a .bed file with columns: chrom, start, stop, TE_family, strand."""
    df = pd.read_csv(
        bed_path, sep="\t", header=None,
        names=["chrom", "start", "stop", "te_family", "strand"],
        dtype={"chrom": str, "start": int, "stop": int, "te_family": str, "strand": str},
    )
    df["length"] = df["stop"] - df["start"]
    return df


def count_reads_from_bam(bam_path: str, bed_df: pd.DataFrame) -> pd.DataFrame:
    """Count sense and antisense reads overlapping each TE locus from a BAM file."""
    import pysam

    bam = pysam.AlignmentFile(bam_path, "rb")
    sense_counts = []
    antisense_counts = []

    for _, row in bed_df.iterrows():
        s_count = 0
        a_count = 0
        te_strand = row["strand"]

        for read in bam.fetch(row["chrom"], row["start"], row["stop"]):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            # Determine read strand
            if read.is_reverse:
                read_strand = "-"
            else:
                read_strand = "+"
            # For paired-end, use read1 strand convention
            if read.is_paired and read.is_read2:
                read_strand = "-" if read_strand == "+" else "+"

            if read_strand == te_strand:
                s_count += 1
            else:
                a_count += 1

        sense_counts.append(s_count)
        antisense_counts.append(a_count)

    bam.close()
    bed_df = bed_df.copy()
    bed_df["sense_count"] = sense_counts
    bed_df["antisense_count"] = antisense_counts
    return bed_df


def aggregate_by_family(bed_df: pd.DataFrame) -> TEFamilyData:
    """Group locus-level data by TE family for model input."""
    families = sorted(bed_df["te_family"].unique())
    sense = []
    antisense = []
    lengths = []

    for fam in families:
        sub = bed_df[bed_df["te_family"] == fam]
        sense.append(sub["sense_count"].values.astype(np.float32))
        antisense.append(sub["antisense_count"].values.astype(np.float32))
        lengths.append(sub["length"].values.astype(np.float32))

    return TEFamilyData(
        family_names=families,
        sense_counts=sense,
        antisense_counts=antisense,
        locus_lengths=lengths,
        n_families=len(families),
    )


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

    Background model: low-level random expression, roughly symmetric sense/antisense.
    Foreground model: elevated sense expression + correlated antisense (dsRNA).
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

    for f in range(n_families):
        n_loci = rng.integers(max(10, n_loci_per_family // 2), n_loci_per_family * 2)
        lengths = rng.integers(200, 8000, size=n_loci).astype(np.float32)

        # Background: low-level Poisson noise, roughly symmetric
        bg_sense = rng.poisson(bg_rate, size=n_loci)
        bg_antisense = rng.poisson(bg_rate * 0.8, size=n_loci)

        if z_true[f] == 1.0:
            # Foreground: elevated sense from TE promoter
            # Not all loci in an active family are necessarily active
            locus_activity = rng.beta(2, 3, size=n_loci)  # heterogeneous activation
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

    data = TEFamilyData(
        family_names=families,
        sense_counts=sense_counts,
        antisense_counts=antisense_counts,
        locus_lengths=locus_lengths,
        n_families=n_families,
    )
    return data, z_true
