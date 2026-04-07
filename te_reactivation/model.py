"""
Bayesian hierarchical model for autonomous TE reactivation detection.

Background: per-locus from flanking regions (local genomic context).
Foreground: family-level magnitude * locus-level scale.

The locus-level scale w_fi ~ LogNormal(0, sigma_w_f) allows heterogeneity
within a family, but sigma_w_f has a tight prior so the family-level
effect log_fg_f has to do the heavy lifting. One hot locus can't drive
the family — you need a consistent pattern across loci.

sigma_w_f is itself informative: high = heterogeneous activation (few hot loci),
low = uniform activation (epigenetic derepression).
"""

import torch
import pyro
import pyro.distributions as dist


def te_reactivation_model(sense_counts, antisense_counts, locus_lengths, n_families,
                           flank_sense_rates=None, flank_antisense_rates=None):

    # ---- Global priors ----
    pi = pyro.sample("pi", dist.Beta(2.0, 8.0))
    inv_disp = pyro.sample("inv_disp", dist.Gamma(3.0, 1.0))
    bg_scale = pyro.sample("bg_scale", dist.LogNormal(0.0, 0.5))

    # ---- Per-family parameters ----
    with pyro.plate("families", n_families):
        # Spike-and-slab
        logit_z = pyro.sample("logit_z", dist.Normal(
            torch.logit(pi).expand(n_families),
            torch.ones(n_families) * 1.5,
        ))
        z = torch.sigmoid(logit_z)

        # Family-level foreground magnitude
        log_fg = pyro.sample("log_fg", dist.Normal(
            torch.ones(n_families) * 2.0,
            torch.ones(n_families) * 1.5,
        ))

        # Antisense ratio
        antisense_ratio = pyro.sample("antisense_ratio", dist.Beta(
            torch.ones(n_families) * 2.0,
            torch.ones(n_families) * 3.0,
        ))

        # Locus heterogeneity scale — tight prior keeps it small
        # so family-level log_fg has to explain the bulk of the signal
        sigma_w = pyro.sample("sigma_w", dist.HalfNormal(
            torch.ones(n_families) * 0.3,
        ))

    # ---- Per-locus observations ----
    for f in range(n_families):
        n_loci = sense_counts[f].shape[0]
        lengths_kb = locus_lengths[f] / 1000.0

        # Per-locus background from flanks
        if flank_sense_rates is not None:
            bg_sense_rate = bg_scale * flank_sense_rates[f] * lengths_kb + 1e-6
            bg_antisense_rate = bg_scale * flank_antisense_rates[f] * lengths_kb + 1e-6
        else:
            bg_sense_rate = bg_scale * lengths_kb + 1e-6
            bg_antisense_rate = bg_scale * lengths_kb + 1e-6

        with pyro.plate(f"loci_{f}", n_loci):
            # Locus-level scaling: centered at 0 in log-space (= scale of 1)
            # sigma_w controls how much individual loci can deviate from family mean
            log_w = pyro.sample(f"log_w_{f}", dist.Normal(
                torch.zeros(n_loci),
                sigma_w[f].expand(n_loci),
            ))
            w = torch.exp(log_w)

            # Foreground = family effect * locus scale
            fg_sense_rate = z[f] * torch.exp(log_fg[f]) * w * lengths_kb
            fg_antisense_rate = z[f] * torch.exp(log_fg[f]) * antisense_ratio[f] * w * lengths_kb

            mu_sense = bg_sense_rate + fg_sense_rate
            mu_antisense = bg_antisense_rate + fg_antisense_rate

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


def te_reactivation_guide(sense_counts, antisense_counts, locus_lengths, n_families,
                           flank_sense_rates=None, flank_antisense_rates=None):

    # Global
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

    bg_scale_loc = pyro.param("bg_scale_loc", torch.tensor(0.0))
    bg_scale_scale = pyro.param("bg_scale_scale", torch.tensor(0.3),
                                constraint=dist.constraints.positive)
    pyro.sample("bg_scale", dist.LogNormal(bg_scale_loc, bg_scale_scale))

    # Per-family
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

    sigma_w_loc = pyro.param("sigma_w_loc", torch.ones(n_families) * 0.2,
                              constraint=dist.constraints.positive)
    sigma_w_scale = pyro.param("sigma_w_scale", torch.ones(n_families) * 0.1,
                                constraint=dist.constraints.positive)

    with pyro.plate("families", n_families):
        pyro.sample("logit_z", dist.Normal(logit_z_loc, logit_z_scale))
        pyro.sample("log_fg", dist.Normal(log_fg_loc, log_fg_scale))
        pyro.sample("antisense_ratio", dist.Beta(ar_alpha, ar_beta))
        pyro.sample("sigma_w", dist.HalfNormal(sigma_w_loc))

    # Per-locus scaling
    for f in range(n_families):
        n_loci = sense_counts[f].shape[0]
        log_w_loc = pyro.param(f"log_w_loc_{f}", torch.zeros(n_loci))
        log_w_scale = pyro.param(f"log_w_scale_{f}", torch.ones(n_loci) * 0.2,
                                  constraint=dist.constraints.positive)
        with pyro.plate(f"loci_{f}", n_loci):
            pyro.sample(f"log_w_{f}", dist.Normal(log_w_loc, log_w_scale))
