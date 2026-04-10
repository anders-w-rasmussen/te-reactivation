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

    # ---- footprint-consensus: consensus-coordinate footprinting ----
    fpc_parser = subparsers.add_parser("footprint-consensus",
        help="Footprint using consensus coordinates from RepeatMasker .out")
    fpc_parser.add_argument("--rm-out", required=True, help="RepeatMasker .out file")
    fpc_parser.add_argument("--bam", required=True, help="BAM file (indexed)")
    fpc_parser.add_argument("--out-dir", "-o", required=True, help="Output directory for plots")
    fpc_parser.add_argument("--families", help="Comma-separated family names to plot (e.g. L1HS,L1PA2,L1PA6)")
    fpc_parser.add_argument("--n-bins", type=int, default=200, help="Bins across consensus body (default: 200)")
    fpc_parser.add_argument("--flank-bp", type=int, default=2000, help="Flank size in bp (default: 2000)")
    fpc_parser.add_argument("--mappability-bw", help="Mappability bigWig")
    fpc_parser.add_argument("--top-n", type=int, default=None, help="Only plot top N families")

    # ---- footprint-compare: compare conditions ----
    cmp_parser = subparsers.add_parser("footprint-compare",
        help="Compare TE footprints between conditions (e.g. control vs IBD)")
    cmp_parser.add_argument("--samples", required=True,
        help="TSV file: bam_path<TAB>condition (e.g. control or alt)")
    cmp_parser.add_argument("--out-dir", "-o", required=True, help="Output directory")
    cmp_parser.add_argument("--families", help="Comma-separated family names")
    cmp_parser.add_argument("--mode", choices=["relative", "consensus"], default="consensus",
        help="relative: normalize to [0,1] (best for short TEs like Alus/SVAs). "
             "consensus: use RepeatMasker consensus coords (best for L1s/HERVs). Default: consensus")
    cmp_parser.add_argument("--rm-out", help="RepeatMasker .out file (required for consensus mode)")
    cmp_parser.add_argument("--bed", help="BED file (required for relative mode)")
    cmp_parser.add_argument("--n-bins", type=int, default=200, help="Bins across TE body (default: 200)")
    cmp_parser.add_argument("--flank-bp", type=int, default=2000, help="Flank size in bp for consensus mode (default: 2000)")
    cmp_parser.add_argument("--flank-frac", type=float, default=1.0, help="Flank as fraction of TE length for relative mode (default: 1.0)")
    cmp_parser.add_argument("--mappability-bw", help="Mappability bigWig")
    cmp_parser.add_argument("--top-n", type=int, default=None, help="Only plot top N families")

    # ---- polyA-background: build artifact footprint from genomic polyA sites ----
    pa_parser = subparsers.add_parser("polya-background",
        help="Build background footprint from genomic polyA sites (artifact profile)")
    pa_parser.add_argument("--bam", required=True, help="Aligned BAM file (indexed)")
    pa_parser.add_argument("--ref", required=True, help="Reference FASTA (indexed, e.g. hg38.fa)")
    pa_parser.add_argument("--out-dir", "-o", required=True, help="Output directory")
    pa_parser.add_argument("--min-a", type=int, default=6, help="Minimum consecutive A's to define a polyA site (default: 6)")
    pa_parser.add_argument("--upstream-bp", type=int, default=800, help="Window upstream of polyA to profile (default: 800)")
    pa_parser.add_argument("--n-bins", type=int, default=100, help="Number of bins (default: 100)")
    pa_parser.add_argument("--max-sites", type=int, default=5000, help="Max polyA sites to sample (default: 5000)")
    pa_parser.add_argument("--exclude-genes", help="Genes BED file to exclude (optional)")
    pa_parser.add_argument("--exclude-tes", help="RepeatMasker BED file to exclude (optional)")
    pa_parser.add_argument("--chroms", help="Comma-separated chromosomes to scan (default: chr1-22,X,Y)")
    pa_parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")

    args = parser.parse_args()

    if args.command == "polya-background":
        from .footprint import extract_polyA_background, plot_polyA_background
        import os, json

        exclude_beds = []
        if args.exclude_genes:
            exclude_beds.append(args.exclude_genes)
        if args.exclude_tes:
            exclude_beds.append(args.exclude_tes)

        chroms = None
        if args.chroms:
            chroms = [c.strip() for c in args.chroms.split(",")]

        bg = extract_polyA_background(
            bam_path=args.bam,
            ref_path=args.ref,
            min_a_count=args.min_a,
            upstream_bp=args.upstream_bp,
            n_bins=args.n_bins,
            max_sites=args.max_sites,
            exclude_beds=exclude_beds if exclude_beds else None,
            seed=args.seed,
            chroms=chroms,
        )

        os.makedirs(args.out_dir, exist_ok=True)

        # Save plot
        plot_path = os.path.join(args.out_dir, "polya_background_footprint.png")
        plot_polyA_background(bg, save_path=plot_path)

        # Save arrays for reuse
        npz_path = os.path.join(args.out_dir, "polya_background.npz")
        np.savez(npz_path,
                 sense_mean=bg["sense_mean"],
                 sense_median=bg["sense_median"],
                 antisense_mean=bg["antisense_mean"],
                 antisense_median=bg["antisense_median"],
                 sense_5p_mean=bg["sense_5p_mean"],
                 antisense_5p_mean=bg["antisense_5p_mean"],
                 bin_centers=bg["bin_centers"],
                 polyA_lengths=np.array(bg["polyA_lengths"]),
        )
        print(f"Background arrays saved to {npz_path}")

        # Save metadata
        meta = {"n_sites": bg["n_sites"], "min_a": args.min_a,
                "upstream_bp": args.upstream_bp, "n_bins": args.n_bins,
                "median_polyA_length": float(np.median(bg["polyA_lengths"]))}
        meta_path = os.path.join(args.out_dir, "polya_background_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"Metadata saved to {meta_path}")

    elif args.command == "footprint-compare":
        from .footprint import (load_repeatmasker_out, load_bed, load_samples_file,
                                extract_and_aggregate_by_condition,
                                extract_and_aggregate_by_condition_relative,
                                plot_footprint_comparison)

        if args.mode == "consensus" and not args.rm_out:
            parser.error("--rm-out is required for consensus mode")
        if args.mode == "relative" and not args.bed:
            parser.error("--bed is required for relative mode")

        print(f"Loading samples: {args.samples}")
        samples = load_samples_file(args.samples)
        for cond, bams in samples.items():
            print(f"  {cond}: {len(bams)} samples")

        family_filter = None
        if args.families:
            family_filter = [f.strip() for f in args.families.split(",")]
            print(f"  Filtering to families: {family_filter}")

        map_bw = getattr(args, 'mappability_bw', None)

        if args.mode == "consensus":
            print(f"Loading RepeatMasker .out: {args.rm_out}")
            rm_df = load_repeatmasker_out(args.rm_out)
            print(f"  {len(rm_df)} total TE loci")
            print(f"Mode: consensus coordinates")

            condition_footprints = extract_and_aggregate_by_condition(
                samples, rm_df,
                family_filter=family_filter,
                n_bins=args.n_bins,
                flank_bp=args.flank_bp,
                mappability_bw_path=map_bw,
            )
        else:
            print(f"Loading BED: {args.bed}")
            bed_df = load_bed(args.bed)
            print(f"  {len(bed_df)} loci across {bed_df['te_family'].nunique()} families")
            print(f"Mode: relative coordinates [0, 1]")

            condition_footprints = extract_and_aggregate_by_condition_relative(
                samples, bed_df,
                family_filter=family_filter,
                n_bins=args.n_bins,
                flank_frac=args.flank_frac,
                mappability_bw_path=map_bw,
            )

        print(f"Plotting to {args.out_dir}/...")
        plot_footprint_comparison(condition_footprints, save_dir=args.out_dir,
                                  top_n=args.top_n)

    elif args.command == "footprint-consensus":
        from .footprint import load_repeatmasker_out, extract_coverage_consensus, compute_footprints, plot_footprints

        print(f"Loading RepeatMasker .out: {args.rm_out}")
        rm_df = load_repeatmasker_out(args.rm_out)
        print(f"  {len(rm_df)} total TE loci, {rm_df['repeat_name'].nunique()} families")

        family_filter = None
        if args.families:
            family_filter = [f.strip() for f in args.families.split(",")]
            print(f"  Filtering to families: {family_filter}")

        print(f"Extracting coverage in consensus coordinates...")
        coverages = extract_coverage_consensus(
            args.bam, rm_df,
            family_filter=family_filter,
            n_bins=args.n_bins,
            flank_bp=args.flank_bp,
            mappability_bw_path=getattr(args, 'mappability_bw', None),
        )
        print(f"  Extracted {sum(v['sense'].shape[0] for v in coverages.values())} loci across {len(coverages)} families")

        print("Computing footprints...")
        footprints = compute_footprints(coverages)

        print(f"Plotting to {args.out_dir}/...")
        plot_footprints(footprints, save_dir=args.out_dir, top_n=args.top_n)

    elif args.command == "footprint":
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
