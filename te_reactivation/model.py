"""
Bayesian model for autonomous TE reactivation detection.

Background: per-locus from flanking regions (local genomic context).
Foreground: family-level magnitude with robust NegBin dispersion to
handle locus heterogeneity without per-locus latent variables.

The NegBin dispersion (inv_disp) naturally handles the fact that some
loci are hotter than others — it models overdispersion at the locus
level. A separate family-level dispersion parameter (inv_disp_fg)
controls how variable the foreground is across loci within a family.
"""

import torch
import pyro
import pyro.distributions as dist


def te_reactivation_model(sense_counts, antisense_counts, locus_lengths, n_families,
                           flank_sense_rates=None, flank_antisense_rates=None):

    # ---- Global priors ----
    pi = pyro.sample("pi", dist.Beta(2.0, 8.0))

    # Separate dispersion for background and foreground
    # Low inv_disp_fg = high variance across loci (focal activation)
    # High inv_disp_fg = low variance (uniform activation)
    inv_disp_bg = pyro.sample("inv_disp_bg", dist.Gamma(3.0, 1.0))
    inv_disp_fg = pyro.sample("inv_disp_fg", dist.Gamma(2.0, 0.5))

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

        # Foreground: family-level rate, overdispersion handled by inv_disp_fg
        fg_sense_rate = z[f] * torch.exp(log_fg[f]) * lengths_kb
        fg_antisense_rate = z[f] * torch.exp(log_fg[f]) * antisense_ratio[f] * lengths_kb

        mu_sense = bg_sense_rate + fg_sense_rate
        mu_antisense = bg_antisense_rate + fg_antisense_rate

        # Use the lower (more overdispersed) of bg and fg dispersion
        # when foreground is active, to allow locus heterogeneity in activation
        effective_disp = torch.where(
            z[f] > 0.5,
            torch.minimum(inv_disp_bg, inv_disp_fg),
            inv_disp_bg,
        )

        with pyro.plate(f"loci_{f}", n_loci):
            pyro.sample(
                f"obs_sense_{f}",
                dist.GammaPoisson(
                    concentration=effective_disp,
                    rate=effective_disp / mu_sense,
                ),
                obs=sense_counts[f],
            )
            pyro.sample(
                f"obs_antisense_{f}",
                dist.GammaPoisson(
                    concentration=effective_disp,
                    rate=effective_disp / mu_antisense,
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

    inv_disp_bg_alpha = pyro.param("inv_disp_bg_alpha", torch.tensor(3.0),
                                    constraint=dist.constraints.positive)
    inv_disp_bg_beta = pyro.param("inv_disp_bg_beta", torch.tensor(1.0),
                                   constraint=dist.constraints.positive)
    pyro.sample("inv_disp_bg", dist.Gamma(inv_disp_bg_alpha, inv_disp_bg_beta))

    inv_disp_fg_alpha = pyro.param("inv_disp_fg_alpha", torch.tensor(2.0),
                                    constraint=dist.constraints.positive)
    inv_disp_fg_beta = pyro.param("inv_disp_fg_beta", torch.tensor(0.5),
                                   constraint=dist.constraints.positive)
    pyro.sample("inv_disp_fg", dist.Gamma(inv_disp_fg_alpha, inv_disp_fg_beta))

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

    with pyro.plate("families", n_families):
        pyro.sample("logit_z", dist.Normal(logit_z_loc, logit_z_scale))
        pyro.sample("log_fg", dist.Normal(log_fg_loc, log_fg_scale))
        pyro.sample("antisense_ratio", dist.Beta(ar_alpha, ar_beta))
