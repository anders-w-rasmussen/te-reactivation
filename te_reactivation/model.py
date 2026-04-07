"""
Bayesian model for autonomous TE reactivation detection.

Background is estimated per-locus from flanking regions (2kb up/downstream).
This captures local genomic context: a TE in a highly expressed gene will have
high flanks too, so only excess expression above the local background is signal.

Foreground (spike-and-slab at family level): autonomous TE transcription that
appears in the TE body but NOT in the flanks.
"""

import torch
import pyro
import pyro.distributions as dist


def te_reactivation_model(sense_counts, antisense_counts, locus_lengths, n_families,
                           flank_sense_rates=None, flank_antisense_rates=None):
    """
    Generative model with flank-based per-locus background.

    Per locus i in family f:
      expected_bg_sense_i = flank_sense_rate_i * length_kb_i
      expected_bg_antisense_i = flank_antisense_rate_i * length_kb_i

    The model asks: does the TE body have excess expression over what
    the flanking regions predict?
    """

    # ---- Global priors ----
    pi = pyro.sample("pi", dist.Beta(2.0, 8.0))
    inv_disp = pyro.sample("inv_disp", dist.Gamma(3.0, 1.0))

    # Background scaling factor — allows the model to learn that TE bodies
    # have slightly different baseline rates than flanks (e.g. mappability)
    bg_scale = pyro.sample("bg_scale", dist.LogNormal(0.0, 0.5))

    # ---- Per-family parameters ----
    with pyro.plate("families", n_families):
        # Continuous spike-and-slab
        logit_z = pyro.sample("logit_z", dist.Normal(
            torch.logit(pi).expand(n_families),
            torch.ones(n_families) * 1.5,
        ))
        z = torch.sigmoid(logit_z)

        # Foreground magnitude (excess over background)
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

        # Per-locus background from flanks
        if flank_sense_rates is not None:
            bg_sense_rate = bg_scale * flank_sense_rates[f] * lengths_kb + 1e-6
            bg_antisense_rate = bg_scale * flank_antisense_rates[f] * lengths_kb + 1e-6
        else:
            # Fallback: uniform background if no flanks available
            bg_sense_rate = bg_scale * lengths_kb + 1e-6
            bg_antisense_rate = bg_scale * lengths_kb + 1e-6

        # Foreground: excess from autonomous transcription
        fg_sense_rate = z[f] * torch.exp(log_fg[f]) * lengths_kb
        fg_antisense_rate = z[f] * torch.exp(log_fg[f]) * antisense_ratio[f] * lengths_kb

        mu_sense = bg_sense_rate + fg_sense_rate
        mu_antisense = bg_antisense_rate + fg_antisense_rate

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


def te_reactivation_guide(sense_counts, antisense_counts, locus_lengths, n_families,
                           flank_sense_rates=None, flank_antisense_rates=None):
    """Mean-field variational guide."""

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

    bg_scale_loc = pyro.param("bg_scale_loc", torch.tensor(0.0))
    bg_scale_scale = pyro.param("bg_scale_scale", torch.tensor(0.3),
                                constraint=dist.constraints.positive)
    pyro.sample("bg_scale", dist.LogNormal(bg_scale_loc, bg_scale_scale))

    # Per-family params
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
