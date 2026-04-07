"""
SVI inference for the TE reactivation model.

No per-locus latent variables — locus heterogeneity handled by NegBin
dispersion. Scales to 100k+ loci without ELBO divergence.
"""

import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam

from .model import te_reactivation_model, te_reactivation_guide
from .data import TEFamilyData


def prepare_tensors(data: TEFamilyData):
    """Convert numpy arrays to torch tensors."""
    sense = [torch.tensor(s, dtype=torch.float32) for s in data.sense_counts]
    antisense = [torch.tensor(a, dtype=torch.float32) for a in data.antisense_counts]
    lengths = [torch.tensor(l, dtype=torch.float32) for l in data.locus_lengths]

    flank_s = None
    flank_a = None
    if data.flank_sense_rates is not None:
        flank_s = [torch.tensor(f, dtype=torch.float32) for f in data.flank_sense_rates]
        flank_a = [torch.tensor(f, dtype=torch.float32) for f in data.flank_antisense_rates]

    return sense, antisense, lengths, flank_s, flank_a


def run_svi(
    data: TEFamilyData,
    n_steps: int = 3000,
    lr: float = 0.005,
    print_every: int = 500,
) -> dict:
    pyro.clear_param_store()

    sense, antisense, lengths, flank_s, flank_a = prepare_tensors(data)

    optimizer = ClippedAdam({"lr": lr, "betas": (0.9, 0.999)})
    svi = SVI(te_reactivation_model, te_reactivation_guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(n_steps):
        loss = svi.step(sense, antisense, lengths, data.n_families, flank_s, flank_a)
        losses.append(loss)
        if print_every and (step + 1) % print_every == 0:
            print(f"Step {step+1:>5d} | ELBO loss: {loss:>10.1f}")

    z_posterior = _compute_z_posteriors(sense, antisense, lengths, data.n_families,
                                        flank_s, flank_a)

    params = {}
    for name in pyro.get_param_store():
        params[name] = pyro.param(name).detach().numpy()

    return {
        "losses": losses,
        "z_posterior": z_posterior,
        "family_names": data.family_names,
        "params": params,
    }


@torch.no_grad()
def _compute_z_posteriors(sense, antisense, lengths, n_families,
                           flank_s, flank_a, n_mc=200):
    """
    Compute P(z_f=1 | data) via log-likelihood ratio.
    """
    log_odds_samples = np.zeros((n_mc, n_families))

    for s in range(n_mc):
        guide_trace = pyro.poutine.trace(te_reactivation_guide).get_trace(
            sense, antisense, lengths, n_families, flank_s, flank_a
        )

        pi = guide_trace.nodes["pi"]["value"]
        inv_disp_bg = guide_trace.nodes["inv_disp_bg"]["value"]
        inv_disp_fg = guide_trace.nodes["inv_disp_fg"]["value"]
        bg_scale = guide_trace.nodes["bg_scale"]["value"]
        log_fg = guide_trace.nodes["log_fg"]["value"]
        ar = guide_trace.nodes["antisense_ratio"]["value"]

        log_prior_odds = torch.log(pi + 1e-8) - torch.log(1 - pi + 1e-8)
        disp_active = torch.minimum(inv_disp_bg, inv_disp_fg)

        for f in range(n_families):
            lengths_kb = lengths[f] / 1000.0

            if flank_s is not None:
                bg_s = bg_scale * flank_s[f] * lengths_kb + 1e-6
                bg_a = bg_scale * flank_a[f] * lengths_kb + 1e-6
            else:
                bg_s = bg_scale * lengths_kb + 1e-6
                bg_a = bg_scale * lengths_kb + 1e-6

            fg_s = torch.exp(log_fg[f]) * lengths_kb
            fg_a = torch.exp(log_fg[f]) * ar[f] * lengths_kb

            mu_s_0 = bg_s
            mu_a_0 = bg_a
            mu_s_1 = bg_s + fg_s
            mu_a_1 = bg_a + fg_a

            # z=0: background dispersion only
            ll_0 = (
                dist.GammaPoisson(inv_disp_bg, inv_disp_bg / mu_s_0).log_prob(sense[f]).sum()
                + dist.GammaPoisson(inv_disp_bg, inv_disp_bg / mu_a_0).log_prob(antisense[f]).sum()
            )
            # z=1: potentially more overdispersed (active dispersion)
            ll_1 = (
                dist.GammaPoisson(disp_active, disp_active / mu_s_1).log_prob(sense[f]).sum()
                + dist.GammaPoisson(disp_active, disp_active / mu_a_1).log_prob(antisense[f]).sum()
            )

            log_odds_samples[s, f] = (ll_1 - ll_0 + log_prior_odds).item()

    mean_log_odds = log_odds_samples.mean(axis=0)
    z_posterior = 1.0 / (1.0 + np.exp(-mean_log_odds))
    return z_posterior


def summarize_results(results: dict, threshold: float = 0.5) -> None:
    print("\n" + "=" * 60)
    print("TE Family Reactivation Results")
    print("=" * 60)
    print(f"{'Family':<25s} {'P(reactivated)':>15s} {'Call':>8s}")
    print("-" * 60)

    z = results["z_posterior"]
    names = results["family_names"]

    order = np.argsort(-z)
    for idx in order:
        call = "ACTIVE" if z[idx] >= threshold else "silent"
        print(f"{names[idx]:<25s} {z[idx]:>15.3f} {call:>8s}")

    n_active = (z >= threshold).sum()
    print("-" * 60)
    print(f"Total reactivated: {int(n_active)} / {len(z)} families (threshold={threshold})")

    params = results["params"]
    ar_alpha = params.get("ar_alpha")
    ar_beta = params.get("ar_beta")
    if ar_alpha is not None:
        ar_mean = ar_alpha / (ar_alpha + ar_beta)
        print(f"\n{'Family':<25s} {'P(react)':>10s} {'AS/S':>8s} {'dsRNA?':>8s}")
        print("-" * 60)
        for idx in order:
            if z[idx] >= threshold:
                dsrna = "yes" if ar_mean[idx] > 0.3 else "low"
                print(f"{names[idx]:<25s} {z[idx]:>10.3f} {ar_mean[idx]:>8.3f} {dsrna:>8s}")
