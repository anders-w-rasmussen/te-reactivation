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


def _resolve_chrom(chrom: str, valid_chroms: set) -> str | None:
    """
    Resolve chromosome name mismatches between files.
    Tries the name as-is, with 'chr' prefix added, and with 'chr' prefix stripped.
    Returns the matching name or None if no match found.
    """
    if chrom in valid_chroms:
        return chrom
    # Try adding chr
    with_chr = f"chr{chrom}"
    if with_chr in valid_chroms:
        return with_chr
    # Try stripping chr
    if chrom.startswith("chr"):
        without_chr = chrom[3:]
        if without_chr in valid_chroms:
            return without_chr
    return None


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
    # 5' read-end density
    sense_5p_mean: np.ndarray | None = None
    antisense_5p_mean: np.ndarray | None = None
    bg_sense_5p: np.ndarray | None = None
    bg_antisense_5p: np.ndarray | None = None
    # Mappability
    mappability_mean: np.ndarray | None = None


def load_bed(bed_path: str) -> pd.DataFrame:
    """Load 5-column BED: chrom, start, stop, TE_family, strand."""
    return pd.read_csv(
        bed_path, sep="\t", header=None,
        names=["chrom", "start", "stop", "te_family", "strand"],
        dtype={"chrom": str, "start": int, "stop": int,
               "te_family": str, "strand": str},
    )


def load_repeatmasker_out(out_path: str) -> pd.DataFrame:
    """
    Parse RepeatMasker .out file.

    Returns DataFrame with columns:
        chrom, start, stop, strand, repeat_name, repeat_family,
        rep_start, rep_end, rep_left
    where rep_start/rep_end are positions in the consensus sequence.
    """
    rows = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("SW") or line.startswith("-"):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            try:
                chrom = parts[4]
                start = int(parts[5]) - 1  # convert to 0-based
                stop = int(parts[6])
                strand = parts[8]
                repeat_name = parts[9]
                repeat_family = parts[10]

                if strand == "+":
                    rep_start = int(parts[11])
                    rep_end = int(parts[12])
                    rep_left = parts[13].strip("()")
                else:
                    # For complement strand, columns are reordered
                    rep_left = parts[11].strip("()")
                    rep_end = int(parts[12])
                    rep_start = int(parts[13])
                    strand = "-"

                rows.append({
                    "chrom": chrom, "start": start, "stop": stop,
                    "strand": strand, "repeat_name": repeat_name,
                    "repeat_family": repeat_family,
                    "rep_start": rep_start, "rep_end": rep_end,
                })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    # Consensus length = rep_end for + strand full-length copies, approximate
    return df


def extract_coverage_consensus(
    bam_path: str,
    rm_df: pd.DataFrame,
    family_filter: list[str] = None,
    consensus_length: int = None,
    n_bins: int = 200,
    flank_bp: int = 2000,
    flank_bins: int = 50,
    min_te_length: int = 50,
    mappability_bw_path: str = None,
) -> dict[str, dict]:
    """
    Extract coverage mapped to consensus coordinates.

    Instead of normalizing each copy to [0,1], maps genomic positions
    to the consensus position using RepeatMasker's rep_start/rep_end.
    This way the promoter always lands at the same x-axis position.

    flank_bp: fixed flanking region in genomic bp (not scaled to TE length).
    consensus_length: if None, inferred from max rep_end per family.

    Returns dict: family -> {
        "sense": (n_loci, flank_bins + n_bins + flank_bins),
        "antisense": ...,
        "sense_5p": ...,
        "antisense_5p": ...,
        "mappability": ... (if provided),
        "bin_centers": array (in consensus bp, with negative for 5' flank),
    }
    """
    import pysam

    bam = pysam.AlignmentFile(bam_path, "rb")
    bam_chroms = set(bam.references)
    chrom_sizes = dict(zip(bam.references, bam.lengths))

    map_bw = None
    map_chroms = set()
    if mappability_bw_path:
        import pyBigWig
        map_bw = pyBigWig.open(mappability_bw_path)
        map_chroms = set(map_bw.chroms().keys())

    # Filter to requested families
    if family_filter:
        rm_df = rm_df[rm_df["repeat_name"].isin(family_filter)]

    families = rm_df["repeat_name"].unique()

    # Infer consensus length per family
    cons_lengths = {}
    for fam in families:
        sub = rm_df[rm_df["repeat_name"] == fam]
        cons_lengths[fam] = consensus_length or int(sub["rep_end"].max())

    total_bins = flank_bins + n_bins + flank_bins

    keys = ["sense", "antisense", "sense_5p", "antisense_5p"]
    if map_bw:
        keys.append("mappability")
    family_data = {fam: {k: [] for k in keys} for fam in families}
    family_bin_centers = {}

    total = len(rm_df)
    for i, (_, row) in enumerate(rm_df.iterrows()):
        if (i + 1) % 10000 == 0:
            print(f"  Extracting (consensus coords): {i+1}/{total} loci...")

        te_len = row["stop"] - row["start"]
        if te_len < min_te_length:
            continue

        bam_chrom = _resolve_chrom(row["chrom"], bam_chroms)
        if bam_chrom is None:
            continue

        chrom_len = chrom_sizes[bam_chrom]
        fam = row["repeat_name"]
        cons_len = cons_lengths[fam]
        rep_start = row["rep_start"]  # where this copy starts in consensus
        rep_end = row["rep_end"]      # where this copy ends in consensus

        # Genomic region with flanks
        ext_start = max(0, row["start"] - flank_bp)
        ext_stop = min(chrom_len, row["stop"] + flank_bp)
        region_len = ext_stop - ext_start

        if region_len <= 0:
            continue

        # --- Coverage ---
        try:
            fwd_counts = bam.count_coverage(
                bam_chrom, ext_start, ext_stop,
                quality_threshold=0,
                read_callback=lambda r: (not r.is_reverse and not r.is_secondary
                                         and not r.is_supplementary),
            )
            fwd_vals = np.array(fwd_counts).sum(axis=0).astype(np.float32)

            rev_counts = bam.count_coverage(
                bam_chrom, ext_start, ext_stop,
                quality_threshold=0,
                read_callback=lambda r: (r.is_reverse and not r.is_secondary
                                         and not r.is_supplementary),
            )
            rev_vals = np.array(rev_counts).sum(axis=0).astype(np.float32)
        except Exception:
            continue

        # --- 5' ends ---
        fwd_5p = np.zeros(region_len, dtype=np.float32)
        rev_5p = np.zeros(region_len, dtype=np.float32)

        for read in bam.fetch(bam_chrom, ext_start, ext_stop):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue
            if read.is_reverse:
                pos = read.reference_end
                if pos is None:
                    continue
            else:
                pos = read.reference_start
            idx = pos - ext_start
            if 0 <= idx < region_len:
                if read.is_reverse:
                    rev_5p[idx] += 1
                else:
                    fwd_5p[idx] += 1

        # --- Mappability ---
        map_vals = None
        if map_bw:
            map_chrom = _resolve_chrom(row["chrom"], map_chroms)
            if map_chrom:
                try:
                    map_vals = np.array(map_bw.values(map_chrom, ext_start, ext_stop), dtype=np.float32)
                    map_vals = np.nan_to_num(map_vals, 0.0)
                except Exception:
                    map_vals = np.ones(region_len, dtype=np.float32)
            else:
                map_vals = np.ones(region_len, dtype=np.float32)

        # --- Orient to TE strand ---
        if row["strand"] == "+":
            sense_cov, antisense_cov = fwd_vals, rev_vals
            sense_5p, antisense_5p = fwd_5p, rev_5p
        else:
            sense_cov = rev_vals[::-1]
            antisense_cov = fwd_vals[::-1]
            sense_5p = rev_5p[::-1]
            antisense_5p = fwd_5p[::-1]
            if map_vals is not None:
                map_vals = map_vals[::-1]

        # --- Map to consensus coordinates ---
        # Each base in the genomic region maps to a consensus position:
        # - 5' flank: positions from -flank_bp to 0 (relative to rep_start)
        # - TE body: positions from rep_start to rep_end in consensus
        # - 3' flank: positions from rep_end to rep_end + flank_bp
        #
        # Build a per-base consensus coordinate array, then bin.

        n_genomic = len(sense_cov)
        # How many genomic bp are in the left flank, TE body, right flank
        left_flank_bp = row["start"] - ext_start
        right_flank_bp = ext_stop - row["stop"]
        te_body_bp = row["stop"] - row["start"]

        # Consensus positions for each section
        left_flank_cons = np.linspace(-left_flank_bp, 0, left_flank_bp, endpoint=False) + rep_start
        te_body_cons = np.linspace(rep_start, rep_end, te_body_bp, endpoint=False)
        right_flank_cons = np.linspace(0, right_flank_bp, right_flank_bp, endpoint=False) + rep_end

        cons_positions = np.concatenate([left_flank_cons, te_body_cons, right_flank_cons])

        # Trim to actual array length (in case of rounding)
        cons_positions = cons_positions[:n_genomic]
        if len(cons_positions) < n_genomic:
            cons_positions = np.pad(cons_positions, (0, n_genomic - len(cons_positions)),
                                     mode='edge')

        # Bin into consensus coordinate bins
        # x-axis: (rep_start - flank_bp) to (rep_end + flank_bp) mapped to
        # (-flank_bp, 0) for 5' flank, (0, cons_len) for body, (cons_len, cons_len+flank_bp) for 3' flank
        # But we want ALL families to share the same x-axis (0 to cons_len for body)
        # Use: bin_start = -flank_bp relative to consensus pos 1
        bin_start = 1 - flank_bp
        bin_end = cons_len + flank_bp
        bin_edges = np.linspace(bin_start, bin_end, total_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if fam not in family_bin_centers:
            family_bin_centers[fam] = bin_centers

        # Histogram each signal into consensus bins
        def _bin_signal(signal, positions, edges):
            result = np.zeros(len(edges) - 1, dtype=np.float32)
            counts = np.zeros(len(edges) - 1, dtype=np.float32)
            indices = np.digitize(positions, edges) - 1
            indices = np.clip(indices, 0, len(result) - 1)
            for j in range(len(signal)):
                result[indices[j]] += signal[j]
                counts[indices[j]] += 1
            # Average (not sum) so bins with more bases don't dominate
            mask = counts > 0
            result[mask] /= counts[mask]
            return result

        family_data[fam]["sense"].append(_bin_signal(sense_cov, cons_positions, bin_edges))
        family_data[fam]["antisense"].append(_bin_signal(antisense_cov, cons_positions, bin_edges))
        family_data[fam]["sense_5p"].append(_bin_signal(sense_5p, cons_positions, bin_edges))
        family_data[fam]["antisense_5p"].append(_bin_signal(antisense_5p, cons_positions, bin_edges))
        if map_vals is not None:
            family_data[fam]["mappability"].append(_bin_signal(map_vals, cons_positions, bin_edges))

    bam.close()
    if map_bw:
        map_bw.close()

    result = {}
    for fam in families:
        s_list = family_data[fam]["sense"]
        if len(s_list) == 0:
            continue
        d = {
            "sense": np.stack(s_list),
            "antisense": np.stack(family_data[fam]["antisense"]),
            "sense_5p": np.stack(family_data[fam]["sense_5p"]),
            "antisense_5p": np.stack(family_data[fam]["antisense_5p"]),
            "bin_centers": family_bin_centers[fam],
        }
        if "mappability" in family_data[fam] and len(family_data[fam]["mappability"]) > 0:
            d["mappability"] = np.stack(family_data[fam]["mappability"])
        result[fam] = d

    return result


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

    bw_chroms = set(bw_fwd.chroms().keys())
    chrom_sizes = dict(zip(bw_fwd.chroms().keys(), bw_fwd.chroms().values()))

    for _, row in bed_df.iterrows():
        te_len = row["stop"] - row["start"]
        if te_len < min_te_length:
            continue

        bw_chrom = _resolve_chrom(row["chrom"], bw_chroms)
        if bw_chrom is None:
            continue

        chrom_len = chrom_sizes[bw_chrom]
        flank_bp = int(te_len * flank_frac)

        # Extended region
        ext_start = max(0, row["start"] - flank_bp)
        ext_stop = min(chrom_len, row["stop"] + flank_bp)

        # Get per-base coverage from both strands
        try:
            fwd_vals = np.array(bw_fwd.values(bw_chrom, ext_start, ext_stop), dtype=np.float32)
            rev_vals = np.array(bw_rev.values(bw_chrom, ext_start, ext_stop), dtype=np.float32)
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


def extract_coverage_bam(
    bam_path: str,
    bed_df: pd.DataFrame,
    n_bins: int = 100,
    flank_frac: float = 0.5,
    min_te_length: int = 50,
    mappability_bw_path: str = None,
) -> dict[str, dict]:
    """
    Extract per-base strand-specific coverage, 5' read-end positions,
    and optionally mappability from a BAM file.

    Returns dict: family -> {
        "sense": (n_loci, total_bins),
        "antisense": (n_loci, total_bins),
        "sense_5p": (n_loci, total_bins),
        "antisense_5p": (n_loci, total_bins),
        "mappability": (n_loci, total_bins) or None,
        "bin_centers": array,
    }
    """
    import pysam

    bam = pysam.AlignmentFile(bam_path, "rb")

    # Open mappability bigWig if provided
    map_bw = None
    if mappability_bw_path:
        import pyBigWig
        map_bw = pyBigWig.open(mappability_bw_path)

    flank_bins = int(n_bins * flank_frac)
    total_bins = flank_bins + n_bins + flank_bins
    range_start = -flank_frac
    range_end = 1.0 + flank_frac
    bin_centers = np.linspace(range_start, range_end, total_bins)

    families = bed_df["te_family"].unique()
    keys = ["sense", "antisense", "sense_5p", "antisense_5p"]
    if map_bw:
        keys.append("mappability")
    family_data = {fam: {k: [] for k in keys}
                   for fam in families}

    bam_chroms = set(bam.references)
    chrom_sizes = dict(zip(bam.references, bam.lengths))
    map_chroms = set(map_bw.chroms().keys()) if map_bw else set()

    total = len(bed_df)
    for i, (_, row) in enumerate(bed_df.iterrows()):
        if (i + 1) % 10000 == 0:
            print(f"  Extracting: {i+1}/{total} loci...")

        te_len = row["stop"] - row["start"]
        if te_len < min_te_length:
            continue

        # Resolve chrom name for BAM
        bed_chrom = row["chrom"]
        bam_chrom = _resolve_chrom(bed_chrom, bam_chroms)
        if bam_chrom is None:
            continue

        chrom_len = chrom_sizes[bam_chrom]
        flank_bp = int(te_len * flank_frac)
        te_start = row["start"]
        te_stop = row["stop"]
        te_strand = row["strand"]

        ext_start = max(0, te_start - flank_bp)
        ext_stop = min(chrom_len, te_stop + flank_bp)
        region_len = ext_stop - ext_start

        if region_len <= 0:
            continue

        # --- Coverage via count_coverage (fast, index-based) ---
        try:
            fwd_counts = bam.count_coverage(
                bam_chrom, ext_start, ext_stop,
                quality_threshold=0,
                read_callback=lambda r: (not r.is_reverse and not r.is_secondary
                                         and not r.is_supplementary),
            )
            fwd_vals = np.array(fwd_counts).sum(axis=0).astype(np.float32)

            rev_counts = bam.count_coverage(
                bam_chrom, ext_start, ext_stop,
                quality_threshold=0,
                read_callback=lambda r: (r.is_reverse and not r.is_secondary
                                         and not r.is_supplementary),
            )
            rev_vals = np.array(rev_counts).sum(axis=0).astype(np.float32)
        except Exception:
            continue

        # --- 5' end positions via fetch (need read-level info) ---
        fwd_5p = np.zeros(region_len, dtype=np.float32)
        rev_5p = np.zeros(region_len, dtype=np.float32)

        for read in bam.fetch(bam_chrom, ext_start, ext_stop):
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                continue

            # 5' end of the read
            if read.is_reverse:
                pos = read.reference_end  # 5' end for reverse reads
                if pos is None:
                    continue
            else:
                pos = read.reference_start  # 5' end for forward reads

            idx = pos - ext_start
            if 0 <= idx < region_len:
                if read.is_reverse:
                    rev_5p[idx] += 1
                else:
                    fwd_5p[idx] += 1

        # --- Mappability (if available) ---
        map_vals = None
        if map_bw:
            map_chrom = _resolve_chrom(bed_chrom, map_chroms)
            if map_chrom is None:
                map_vals = np.ones(region_len, dtype=np.float32)
            else:
                try:
                    map_vals = np.array(map_bw.values(map_chrom, ext_start, ext_stop), dtype=np.float32)
                    map_vals = np.nan_to_num(map_vals, 0.0)
                except Exception:
                    map_vals = np.ones(region_len, dtype=np.float32)

        # --- Orient to TE strand ---
        if te_strand == "+":
            sense_cov = fwd_vals
            antisense_cov = rev_vals
            sense_5p = fwd_5p
            antisense_5p = rev_5p
        else:
            sense_cov = rev_vals[::-1]
            antisense_cov = fwd_vals[::-1]
            sense_5p = rev_5p[::-1]
            antisense_5p = fwd_5p[::-1]
            if map_vals is not None:
                map_vals = map_vals[::-1]

        # Resize all to bins
        fam = row["te_family"]
        family_data[fam]["sense"].append(_resize_to_bins(sense_cov, total_bins))
        family_data[fam]["antisense"].append(_resize_to_bins(antisense_cov, total_bins))
        family_data[fam]["sense_5p"].append(_resize_to_bins(sense_5p, total_bins))
        family_data[fam]["antisense_5p"].append(_resize_to_bins(antisense_5p, total_bins))
        if map_vals is not None:
            family_data[fam]["mappability"].append(_resize_to_bins(map_vals, total_bins))

    bam.close()
    if map_bw:
        map_bw.close()

    result = {}
    for fam in families:
        s_list = family_data[fam]["sense"]
        if len(s_list) == 0:
            continue
        d = {
            "sense": np.stack(s_list),
            "antisense": np.stack(family_data[fam]["antisense"]),
            "sense_5p": np.stack(family_data[fam]["sense_5p"]),
            "antisense_5p": np.stack(family_data[fam]["antisense_5p"]),
            "bin_centers": bin_centers,
        }
        if "mappability" in family_data[fam] and len(family_data[fam]["mappability"]) > 0:
            d["mappability"] = np.stack(family_data[fam]["mappability"])
        result[fam] = d

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

    Background estimation (two modes):
    A. With mappability: learn coverage = f(mappability) from flanks,
       predict expected coverage inside TE body given its mappability.
    B. Without mappability: smooth flank signal, interpolate across TE body.

    In both cases, antisense signal serves as a second background.
    """
    footprints = {}

    for fam, cov in family_coverages.items():
        sense_mat = cov["sense"]
        antisense_mat = cov["antisense"]
        bin_centers = cov["bin_centers"]
        n_loci = sense_mat.shape[0]
        has_map = "mappability" in cov

        # Aggregate
        sense_mean = sense_mat.mean(axis=0)
        sense_median = np.median(sense_mat, axis=0)
        antisense_mean = antisense_mat.mean(axis=0)
        antisense_median = np.median(antisense_mat, axis=0)
        mappability_mean = cov["mappability"].mean(axis=0) if has_map else None

        # Background estimation
        if has_map:
            bg_sense = _estimate_background_mappability(
                sense_mean, mappability_mean, bin_centers, flank_frac)
            bg_antisense = _estimate_background_mappability(
                antisense_mean, mappability_mean, bin_centers, flank_frac)
        else:
            bg_sense = _estimate_background(sense_mean, bin_centers, flank_frac, bg_smooth_window)
            bg_antisense = _estimate_background(antisense_mean, bin_centers, flank_frac, bg_smooth_window)

        # 5' end data
        sense_5p_mean = None
        antisense_5p_mean = None
        bg_sense_5p = None
        bg_antisense_5p = None
        if "sense_5p" in cov:
            sense_5p_mean = cov["sense_5p"].mean(axis=0)
            antisense_5p_mean = cov["antisense_5p"].mean(axis=0)
            if has_map:
                bg_sense_5p = _estimate_background_mappability(
                    sense_5p_mean, mappability_mean, bin_centers, flank_frac)
                bg_antisense_5p = _estimate_background_mappability(
                    antisense_5p_mean, mappability_mean, bin_centers, flank_frac)
            else:
                bg_sense_5p = _estimate_background(sense_5p_mean, bin_centers, flank_frac, bg_smooth_window)
                bg_antisense_5p = _estimate_background(antisense_5p_mean, bin_centers, flank_frac, bg_smooth_window)

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
            sense_5p_mean=sense_5p_mean,
            antisense_5p_mean=antisense_5p_mean,
            bg_sense_5p=bg_sense_5p,
            bg_antisense_5p=bg_antisense_5p,
            mappability_mean=mappability_mean,
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


def _estimate_background_mappability(
    signal: np.ndarray,
    mappability: np.ndarray,
    bin_centers: np.ndarray,
    flank_frac: float,
    n_map_bins: int = 20,
) -> np.ndarray:
    """
    Estimate background using mappability regression learned from flanks.

    1. In flank regions, bin positions by mappability score
    2. For each mappability bin, compute mean coverage → empirical curve
    3. For TE body positions, predict expected coverage from their mappability
       using the learned curve

    This accounts for the fact that repetitive TE bodies have different
    mappability than unique flanking sequences.
    """
    flank_mask = (bin_centers < 0) | (bin_centers > 1)

    # Get flank data points
    flank_signal = signal[flank_mask]
    flank_map = mappability[flank_mask]

    if len(flank_signal) == 0 or flank_map.max() - flank_map.min() < 1e-6:
        # Fallback: constant background from flank mean
        return np.full_like(signal, flank_signal.mean() if len(flank_signal) > 0 else 0)

    # Bin flanks by mappability and compute mean coverage per bin
    map_bin_edges = np.linspace(flank_map.min() - 1e-6, flank_map.max() + 1e-6, n_map_bins + 1)
    map_bin_centers = (map_bin_edges[:-1] + map_bin_edges[1:]) / 2
    bin_means = np.zeros(n_map_bins)
    bin_counts = np.zeros(n_map_bins)

    bin_indices = np.digitize(flank_map, map_bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_map_bins - 1)

    for b in range(n_map_bins):
        mask = bin_indices == b
        if mask.any():
            bin_means[b] = flank_signal[mask].mean()
            bin_counts[b] = mask.sum()

    # Remove empty bins
    valid = bin_counts > 0
    if valid.sum() < 2:
        return np.full_like(signal, flank_signal.mean())

    valid_centers = map_bin_centers[valid]
    valid_means = bin_means[valid]

    # Predict background for ALL positions (flanks + TE body) using their mappability
    # Interpolate/extrapolate from the learned curve
    bg = np.interp(mappability, valid_centers, valid_means).astype(np.float32)

    return bg


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

        # Detect coordinate system: consensus bp (large values) vs relative [0,1]
        is_consensus = bc.max() > 10
        if is_consensus:
            # Consensus coords: TE body is between first positive value and
            # the point where flank starts on the right
            # Approximate: body starts at 1, ends at max - flank region
            te_start_x = 1
            te_end_x = bc[bc > 0].max() - (bc.max() - bc[bc > 0].max()) if bc[bc > 0].max() > 0 else bc.max()
            # Simpler: body bins have positive values, flanks are negative or > cons_len
            # Use the gap: flank bins are at extremes
            positive_bins = bc[bc > 0]
            if len(positive_bins) > 2:
                # Body is roughly the middle portion
                te_end_x = positive_bins[-1] - (bc.max() - positive_bins[-1])
                if te_end_x <= te_start_x:
                    te_end_x = positive_bins[int(len(positive_bins) * 0.67)]
            te_mask = (bc >= te_start_x) & (bc <= te_end_x)
            x_label = "Consensus position (bp)"
        else:
            te_start_x = 0
            te_end_x = 1
            te_mask = (bc >= 0) & (bc <= 1)
            x_label = "Relative position (0 = TE 5', 1 = TE 3')"

        has_5p = fp.sense_5p_mean is not None
        n_panels = 5 if has_5p else 3
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 4 * n_panels))

        # ---- Panel 1: Sense coverage ----
        ax = axes[0]
        ax.axvspan(te_start_x, te_end_x, alpha=0.06, color="#333")
        ax.axvline(te_start_x, color="#333", linewidth=0.8, alpha=0.4)
        ax.axvline(te_end_x, color="#333", linewidth=0.8, alpha=0.4)

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
        ax.text(te_start_x, 1.03, "5'", transform=ax.get_xaxis_transform(),
                fontsize=11, fontweight="bold", ha="center")
        ax.text(te_end_x, 1.03, "3'", transform=ax.get_xaxis_transform(),
                fontsize=11, fontweight="bold", ha="center")

        # ---- Panel 2: Antisense ----
        ax = axes[1]
        ax.axvspan(te_start_x, te_end_x, alpha=0.06, color="#333")
        ax.axvline(te_start_x, color="#333", linewidth=0.8, alpha=0.4)
        ax.axvline(te_end_x, color="#333", linewidth=0.8, alpha=0.4)

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
        ax.axvspan(te_start_x, te_end_x, alpha=0.06, color="#333")
        ax.axvline(te_start_x, color="#333", linewidth=0.8, alpha=0.4)
        ax.axvline(te_end_x, color="#333", linewidth=0.8, alpha=0.4)

        diff = fp.sense_mean - fp.antisense_mean
        bg_diff = fp.bg_sense - fp.bg_antisense

        ax.fill_between(bc, diff, alpha=0.4, color="#7570b3", label="Sense - Antisense")
        ax.plot(bc, diff, color="#7570b3", linewidth=1.0)
        ax.plot(bc, bg_diff, color="#7570b3", linewidth=2, linestyle="--",
                label="Expected (from flanks)")

        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Sense - Antisense (CPM)", fontsize=10)
        ax.set_title("Strand asymmetry (positive = sense-biased = autonomous)", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

        # ---- Panel 4: Sense 5' read ends ----
        if has_5p:
            ax = axes[3]
            ax.axvspan(0, 1, alpha=0.06, color="#333")
            ax.axvline(0, color="#333", linewidth=0.8, alpha=0.4)
            ax.axvline(1, color="#333", linewidth=0.8, alpha=0.4)

            ax.fill_between(bc, fp.sense_5p_mean, alpha=0.4, color="#2166ac",
                            label="Sense 5' ends")
            ax.plot(bc, fp.sense_5p_mean, color="#2166ac", linewidth=1.0)
            ax.plot(bc, fp.bg_sense_5p, color="#2166ac", linewidth=2, linestyle="--",
                    label="Expected bg")

            ax.fill_between(bc[te_mask],
                            fp.bg_sense_5p[te_mask],
                            np.maximum(fp.sense_5p_mean[te_mask], fp.bg_sense_5p[te_mask]),
                            alpha=0.35, color="#ff7f0e", label="Excess")

            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Mean 5' ends per locus", fontsize=10)
            ax.set_title("Sense read 5' ends (TSS proxy — strongest with long reads)", fontsize=11)
            ax.legend(fontsize=8, loc="upper right")

            # ---- Panel 5: Antisense 5' read ends ----
            ax = axes[4]
            ax.axvspan(0, 1, alpha=0.06, color="#333")
            ax.axvline(0, color="#333", linewidth=0.8, alpha=0.4)
            ax.axvline(1, color="#333", linewidth=0.8, alpha=0.4)

            ax.fill_between(bc, fp.antisense_5p_mean, alpha=0.4, color="#b2182b",
                            label="Antisense 5' ends")
            ax.plot(bc, fp.antisense_5p_mean, color="#b2182b", linewidth=1.0)
            ax.plot(bc, fp.bg_antisense_5p, color="#b2182b", linewidth=2, linestyle="--",
                    label="Expected bg")

            ax.fill_between(bc[te_mask],
                            fp.bg_antisense_5p[te_mask],
                            np.maximum(fp.antisense_5p_mean[te_mask], fp.bg_antisense_5p[te_mask]),
                            alpha=0.35, color="#ff7f0e", label="Excess")

            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Mean 5' ends per locus", fontsize=10)
            ax.set_xlabel(x_label, fontsize=10)
            ax.set_title("Antisense read 5' ends", fontsize=11)
            ax.legend(fontsize=8, loc="upper right")

        if not has_5p:
            axes[2].set_xlabel("Relative position (0 = TE 5', 1 = TE 3')", fontsize=10)

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
