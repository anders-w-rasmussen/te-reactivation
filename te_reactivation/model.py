"""
Bayesian model for autonomous TE reactivation detection.

Analogy to ATAC-seq footprinting:
- Background (slab): read-through / genomic noise — SHARED across families
- Foreground (spike): autonomous TE transcription — family-specific, gated by z_f
- Spike-and-slab at the family level using discrete enumeration

Key insight: background read-through rate is a property of the genomic context,
not the TE family. Sharing it forces the model to use the foreground component
to explain excess expression in reactivated families.
"""

import torch
import pyro
import pyro.distributions as dist


def te_reactivation_model(sense_counts, antisense_counts, locus_lengths, n_families):
    """
    Generative model for TE family reactivation.

    Shared background across all families forces the foreground (z_f=1)
    to explain any excess expression.
    """

    # ---- Global priors ----
    pi = pyro.sample("pi", dist.Beta(2.0, 8.0))
    inv_disp = pyro.sample("inv_disp", dist.Gamma(3.0, 1.0))

    # SHARED background rates (per-kb) — same for all families
    # This is the key: read-through noise is a genomic property, not TE-specific
    log_bg_sense = pyro.sample("log_bg_sense", dist.Normal(0.0, 1.0))
    log_bg_antisense = pyro.sample("log_bg_antisense", dist.Normal(0.0, 1.0))

    # ---- Per-family parameters ----
    for f in range(n_families):
        n_loci = sense_counts[f].shape[0]
        lengths_kb = locus_lengths[f] / 1000.0

        # Discrete spike-and-slab
        z_f = pyro.sample(f"z_{f}", dist.Bernoulli(pi),
                          infer={"enumerate": "parallel"})

        # Family-specific foreground magnitude and antisense ratio
        log_fg = pyro.sample(f"log_fg_{f}", dist.Normal(2.0, 1.5))
        antisense_ratio = pyro.sample(f"antisense_ratio_{f}", dist.Beta(2.0, 3.0))

        # Compute rates
        bg_sense_rate = torch.exp(log_bg_sense) * lengths_kb
        bg_antisense_rate = torch.exp(log_bg_antisense) * lengths_kb
        fg_sense_rate = z_f * torch.exp(log_fg) * lengths_kb
        fg_antisense_rate = z_f * torch.exp(log_fg) * antisense_ratio * lengths_kb

        mu_sense = bg_sense_rate + fg_sense_rate + 1e-6
        mu_antisense = bg_antisense_rate + fg_antisense_rate + 1e-6

        with pyro.plate(f"loci_{f}", n_loci):
            pyro.sample(
                f"obs_sense_{f}",
                dist.GammaPoisson(
                    concentration=inv_disp,
                    rate=inv_disp / mu_sense,
                ),
                obs=sense_counts[f],
            )
            pyro.sample(
                f"obs_antisense_{f}",
                dist.GammaPoisson(
                    concentration=inv_disp,
                    rate=inv_disp / mu_antisense,
                ),
                obs=antisense_counts[f],
            )


def te_reactivation_guide(sense_counts, antisense_counts, locus_lengths, n_families):
    """Mean-field variational guide. Discrete z_f is enumerated (no guide needed)."""

    # Global params
    pi_alpha = pyro.param("pi_alpha", torch.tensor(2.0),
                          constraint=dist.constraints.positive)
    pi_beta = pyro.param("pi_beta", torch.tensor(8.0),
                         constraint=dist.constraints.positive)
    pyro.sample("pi", dist.Beta(pi_alpha, pi_beta))

    inv_disp_alpha = pyro.param("inv_disp_alpha", torch.tensor(3.0),
                                constraint=dist.constraints.positive)
    inv_disp_beta = pyro.param("inv_disp_beta", torch.tensor(1.0),
                               constraint=dist.constraints.positive)
    pyro.sample("inv_disp", dist.Gamma(inv_disp_alpha, inv_disp_beta))

    # Shared background
    loc_bg_s = pyro.param("loc_bg_s", torch.tensor(0.0))
    scale_bg_s = pyro.param("scale_bg_s", torch.tensor(0.5),
                            constraint=dist.constraints.positive)
    pyro.sample("log_bg_sense", dist.Normal(loc_bg_s, scale_bg_s))

    loc_bg_a = pyro.param("loc_bg_a", torch.tensor(0.0))
    scale_bg_a = pyro.param("scale_bg_a", torch.tensor(0.5),
                            constraint=dist.constraints.positive)
    pyro.sample("log_bg_antisense", dist.Normal(loc_bg_a, scale_bg_a))

    # Per-family foreground
    for f in range(n_families):
        loc_fg = pyro.param(f"loc_fg_{f}", torch.tensor(2.0))
        scale_fg = pyro.param(f"scale_fg_{f}", torch.tensor(0.5),
                              constraint=dist.constraints.positive)
        pyro.sample(f"log_fg_{f}", dist.Normal(loc_fg, scale_fg))

        ar_alpha = pyro.param(f"ar_alpha_{f}", torch.tensor(2.0),
                              constraint=dist.constraints.positive)
        ar_beta = pyro.param(f"ar_beta_{f}", torch.tensor(3.0),
                             constraint=dist.constraints.positive)
        pyro.sample(f"antisense_ratio_{f}", dist.Beta(ar_alpha, ar_beta))
