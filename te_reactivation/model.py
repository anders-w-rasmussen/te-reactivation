"""
Bayesian model for autonomous TE reactivation detection.

Analogy to ATAC-seq footprinting:
- Background (slab): read-through / genomic noise — SHARED across families
- Foreground (spike): autonomous TE transcription — family-specific, gated by z_f
- Continuous spike-and-slab via sigmoid(logit_z) with shared background

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

    Uses continuous relaxation of spike-and-slab (sigmoid of logit_z)
    to avoid the 64-dim limit of parallel enumeration. Shared background
    provides identifiability.
    """

    # ---- Global priors ----
    pi = pyro.sample("pi", dist.Beta(2.0, 8.0))
    inv_disp = pyro.sample("inv_disp", dist.Gamma(3.0, 1.0))

    # SHARED background rates (per-kb) — same for all families
    log_bg_sense = pyro.sample("log_bg_sense", dist.Normal(0.0, 1.0))
    log_bg_antisense = pyro.sample("log_bg_antisense", dist.Normal(0.0, 1.0))

    # ---- Per-family parameters ----
    with pyro.plate("families", n_families):
        # Continuous spike-and-slab: logit_z → sigmoid → soft gate
        logit_z = pyro.sample("logit_z", dist.Normal(
            torch.logit(pi).expand(n_families),
            torch.ones(n_families) * 1.5,
        ))
        z = torch.sigmoid(logit_z)

        # Foreground magnitude and antisense ratio
        log_fg = pyro.sample("log_fg", dist.Normal(
            torch.ones(n_families) * 2.0,
            torch.ones(n_families) * 1.5,
        ))
        antisense_ratio = pyro.sample("antisense_ratio", dist.Beta(
            torch.ones(n_families) * 2.0,
            torch.ones(n_families) * 3.0,
        ))

    # ---- Per-locus observations ----
    for f in range(n_families):
        n_loci = sense_counts[f].shape[0]
        lengths_kb = locus_lengths[f] / 1000.0

        bg_sense_rate = torch.exp(log_bg_sense) * lengths_kb
        bg_antisense_rate = torch.exp(log_bg_antisense) * lengths_kb
        fg_sense_rate = z[f] * torch.exp(log_fg[f]) * lengths_kb
        fg_antisense_rate = z[f] * torch.exp(log_fg[f]) * antisense_ratio[f] * lengths_kb

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
    """Mean-field variational guide for continuous spike-and-slab."""

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

    # Per-family params (vectorized)
    logit_z_loc = pyro.param("logit_z_loc", torch.zeros(n_families))
    logit_z_scale = pyro.param("logit_z_scale", torch.ones(n_families) * 0.5,
                                constraint=dist.constraints.positive)

    log_fg_loc = pyro.param("log_fg_loc", torch.ones(n_families) * 2.0)
    log_fg_scale = pyro.param("log_fg_scale", torch.ones(n_families) * 0.5,
                               constraint=dist.constraints.positive)

    ar_alpha = pyro.param("ar_alpha", torch.ones(n_families) * 2.0,
                           constraint=dist.constraints.positive)
    ar_beta = pyro.param("ar_beta", torch.ones(n_families) * 3.0,
                          constraint=dist.constraints.positive)

    with pyro.plate("families", n_families):
        pyro.sample("logit_z", dist.Normal(logit_z_loc, logit_z_scale))
        pyro.sample("log_fg", dist.Normal(log_fg_loc, log_fg_scale))
        pyro.sample("antisense_ratio", dist.Beta(ar_alpha, ar_beta))
