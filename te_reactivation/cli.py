"""
CLI entry point for TE reactivation analysis.
"""

import argparse
import sys
import json
import numpy as np

from .data import load_bed, count_reads_from_bam, aggregate_by_family
from .inference import run_svi, summarize_results


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian detection of autonomous TE reactivation from RNA-seq data"
    )
    subparsers = parser.add_subparsers(dest="command")

    # ---- run: full pipeline from BAM + BED ----
    run_parser = subparsers.add_parser("run", help="Run reactivation analysis on BAM + BED")
    run_parser.add_argument("--bed", required=True, help="BED file: chrom start stop TE_family strand")
    run_parser.add_argument("--bam", required=True, help="Aligned BAM file (indexed)")
    run_parser.add_argument("--steps", type=int, default=3000, help="SVI steps (default: 3000)")
    run_parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005)")
    run_parser.add_argument("--threshold", type=float, default=0.5, help="Reactivation call threshold")
    run_parser.add_argument("--output", "-o", help="Output TSV path (optional)")
    run_parser.add_argument("--plots", help="Directory to save plots (optional)")

    # ---- synthetic: validate on synthetic data ----
    synth_parser = subparsers.add_parser("synthetic", help="Run on synthetic data to validate model")
    synth_parser.add_argument("--n-families", type=int, default=20)
    synth_parser.add_argument("--n-loci", type=int, default=50)
    synth_parser.add_argument("--frac-active", type=float, default=0.3)
    synth_parser.add_argument("--steps", type=int, default=3000)
    synth_parser.add_argument("--lr", type=float, default=0.005)
    synth_parser.add_argument("--seed", type=int, default=42)
    synth_parser.add_argument("--plots", help="Directory to save plots (optional)")

    # ---- footprint: bigWig-based footprinting ----
    fp_parser = subparsers.add_parser("footprint", help="Generate TE footprint plots")
    fp_parser.add_argument("--bed", required=True, help="BED file: chrom start stop TE_family strand")
    fp_input = fp_parser.add_mutually_exclusive_group(required=True)
    fp_input.add_argument("--bam", help="BAM file (indexed) — slower but no preprocessing needed")
    fp_input.add_argument("--fwd-bw", help="Forward-strand bigWig (use with --rev-bw)")
    fp_parser.add_argument("--rev-bw", help="Reverse-strand bigWig (use with --fwd-bw)")
    fp_parser.add_argument("--out-dir", "-o", required=True, help="Output directory for plots")
    fp_parser.add_argument("--n-bins", type=int, default=100, help="Bins across TE body (default: 100)")
    fp_parser.add_argument("--flank-frac", type=float, default=0.5, help="Flank size as fraction of TE length (default: 0.5)")
    fp_parser.add_argument("--mappability-bw", help="Mappability bigWig for background correction (e.g. k100.Umap.bw)")
    fp_parser.add_argument("--top-n", type=int, default=None, help="Only plot top N families by enrichment")

    args = parser.parse_args()

    if args.command == "footprint":
        from .footprint import load_bed as fp_load_bed, extract_coverage, extract_coverage_bam, compute_footprints, plot_footprints

        if args.fwd_bw and not args.rev_bw:
            parser.error("--fwd-bw requires --rev-bw")

        print(f"Loading BED: {args.bed}")
        bed_df = fp_load_bed(args.bed)
        n_fam = bed_df["te_family"].nunique()
        print(f"  {len(bed_df)} loci across {n_fam} families")

        if args.bam:
            map_bw = getattr(args, 'mappability_bw', None)
            if map_bw:
                print(f"Using mappability correction: {map_bw}")
            print(f"Extracting coverage from BAM: {args.bam}")
            coverages = extract_coverage_bam(
                args.bam, bed_df,
                n_bins=args.n_bins, flank_frac=args.flank_frac,
                mappability_bw_path=map_bw,
            )
        else:
            print(f"Extracting coverage from bigWigs...")
            coverages = extract_coverage(
                args.fwd_bw, args.rev_bw, bed_df,
                n_bins=args.n_bins, flank_frac=args.flank_frac,
            )
        print(f"  Extracted {sum(v['sense'].shape[0] for v in coverages.values())} loci across {len(coverages)} families")

        print("Computing footprints...")
        footprints = compute_footprints(coverages, flank_frac=args.flank_frac)

        print(f"Plotting to {args.out_dir}/...")
        plot_footprints(footprints, save_dir=args.out_dir, top_n=args.top_n)

    elif args.command == "run":
        print(f"Loading BED: {args.bed}")
        bed_df = load_bed(args.bed)
        print(f"  {len(bed_df)} loci across {bed_df['te_family'].nunique()} families")

        print(f"Counting reads from: {args.bam}")
        bed_df = count_reads_from_bam(args.bam, bed_df)

        data = aggregate_by_family(bed_df)
        print(f"Running SVI ({args.steps} steps)...")
        results = run_svi(data, n_steps=args.steps, lr=args.lr)
        summarize_results(results, threshold=args.threshold)

        if args.output:
            _write_output(results, args.output, args.threshold)
            print(f"\nResults written to {args.output}")

        if args.plots:
            from .data import extract_positional_profiles
            from .plotting import plot_summary
            print("Extracting positional profiles from BAM (for footprint plots)...")
            profiles = extract_positional_profiles(args.bam, bed_df)
            plot_summary(data, results, threshold=args.threshold,
                         save_dir=args.plots, profiles=profiles)

    elif args.command == "synthetic":
        from .data import generate_synthetic_data
        print("Generating synthetic data...")
        data, z_true = generate_synthetic_data(
            n_families=args.n_families,
            n_loci_per_family=args.n_loci,
            frac_reactivated=args.frac_active,
            seed=args.seed,
        )
        true_active = set(np.where(z_true == 1.0)[0])
        print(f"  {len(true_active)} / {data.n_families} families truly reactivated")
        print(f"  True active: {[data.family_names[i] for i in sorted(true_active)]}")

        print(f"\nRunning SVI ({args.steps} steps)...")
        results = run_svi(data, n_steps=args.steps, lr=args.lr)
        summarize_results(results)

        # Evaluate accuracy
        z_post = results["z_posterior"]
        called_active = set(np.where(z_post >= 0.5)[0])
        tp = len(called_active & true_active)
        fp = len(called_active - true_active)
        fn = len(true_active - called_active)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n{'='*60}")
        print("Synthetic Data Evaluation")
        print(f"{'='*60}")
        print(f"  TP={tp}  FP={fp}  FN={fn}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1:        {f1:.3f}")

        if args.plots:
            from .plotting import plot_summary
            plot_summary(data, results, save_dir=args.plots)

    else:
        parser.print_help()
        sys.exit(1)


def _write_output(results, path, threshold):
    """Write results to TSV."""
    import pandas as pd
    z = results["z_posterior"]
    ar_alpha = results["params"]["ar_alpha"]
    ar_beta = results["params"]["ar_beta"]
    ar_mean = ar_alpha / (ar_alpha + ar_beta)

    df = pd.DataFrame({
        "te_family": results["family_names"],
        "p_reactivated": z,
        "antisense_sense_ratio": ar_mean,
        "call": ["ACTIVE" if p >= threshold else "silent" for p in z],
    })
    df = df.sort_values("p_reactivated", ascending=False)
    df.to_csv(path, sep="\t", index=False)


if __name__ == "__main__":
    main()
