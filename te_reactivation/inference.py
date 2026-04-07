"""
SVI inference for the TE reactivation model.

Strategy: fit continuous relaxation via Trace_ELBO to learn shared background,
foreground magnitudes, and dispersion. Then compute discrete P(z_f=1 | data)
post-hoc via log-likelihood ratios. This scales to any number of families.
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
    return sense, antisense, lengths


def run_svi(
    data: TEFamilyData,
    n_steps: int = 3000,
    lr: float = 0.005,
    print_every: int = 500,
) -> dict:
    """
    Run stochastic variational inference, then compute discrete posteriors.
    """
    pyro.clear_param_store()

    sense, antisense, lengths = prepare_tensors(data)

    optimizer = ClippedAdam({"lr": lr, "betas": (0.9, 0.999)})
    svi = SVI(te_reactivation_model, te_reactivation_guide, optimizer, loss=Trace_ELBO())

    losses = []
    for step in range(n_steps):
        loss = svi.step(sense, antisense, lengths, data.n_families)
        losses.append(loss)
        if print_every and (step + 1) % print_every == 0:
            print(f"Step {step+1:>5d} | ELBO loss: {loss:>10.1f}")

    # Compute discrete posteriors via log-likelihood ratio
    z_posterior = _compute_z_posteriors(sense, antisense, lengths, data.n_families)

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
def _compute_z_posteriors(sense, antisense, lengths, n_families, n_mc=200):
    """
    Compute P(z_f=1 | data) for each family via log-likelihood ratio.

    For each MC sample of continuous params from the guide:
      - Evaluate log p(data_f | z=0, theta) vs log p(data_f | z=1, theta)
      - Combine with prior odds
    Average log-odds across MC samples, convert to probability.
    """
    log_odds_samples = np.zeros((n_mc, n_families))

    for s in range(n_mc):
        guide_trace = pyro.poutine.trace(te_reactivation_guide).get_trace(
            sense, antisense, lengths, n_families
        )

        pi = guide_trace.nodes["pi"]["value"]
        inv_disp = guide_trace.nodes["inv_disp"]["value"]
        log_bg_s = guide_trace.nodes["log_bg_sense"]["value"]
        log_bg_a = guide_trace.nodes["log_bg_antisense"]["value"]
        log_fg = guide_trace.nodes["log_fg"]["value"]
        ar = guide_trace.nodes["antisense_ratio"]["value"]

        log_prior_odds = torch.log(pi + 1e-8) - torch.log(1 - pi + 1e-8)

        for f in range(n_families):
            lengths_kb = lengths[f] / 1000.0

            bg_s = torch.exp(log_bg_s) * lengths_kb
            bg_a = torch.exp(log_bg_a) * lengths_kb
            fg_s = torch.exp(log_fg[f]) * lengths_kb
            fg_a = torch.exp(log_fg[f]) * ar[f] * lengths_kb

            mu_s_0 = bg_s + 1e-6
            mu_a_0 = bg_a + 1e-6
            mu_s_1 = bg_s + fg_s + 1e-6
            mu_a_1 = bg_a + fg_a + 1e-6

            ll_0 = (
                dist.GammaPoisson(inv_disp, inv_disp / mu_s_0).log_prob(sense[f]).sum()
                + dist.GammaPoisson(inv_disp, inv_disp / mu_a_0).log_prob(antisense[f]).sum()
            )
            ll_1 = (
                dist.GammaPoisson(inv_disp, inv_disp / mu_s_1).log_prob(sense[f]).sum()
                + dist.GammaPoisson(inv_disp, inv_disp / mu_a_1).log_prob(antisense[f]).sum()
            )

            log_odds_samples[s, f] = (ll_1 - ll_0 + log_prior_odds).item()

    mean_log_odds = log_odds_samples.mean(axis=0)
    z_posterior = 1.0 / (1.0 + np.exp(-mean_log_odds))
    return z_posterior


def summarize_results(results: dict, threshold: float = 0.5) -> None:
    """Print a summary of reactivation calls."""
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

    # Report antisense ratios for active families
    params = results["params"]
    ar_alpha = params.get("ar_alpha")
    ar_beta = params.get("ar_beta")
    if ar_alpha is not None:
        ar_mean = ar_alpha / (ar_alpha + ar_beta)
        print(f"\n{'Family':<25s} {'P(react)':>10s} {'AS/S ratio':>12s} {'dsRNA?':>8s}")
        print("-" * 60)
        for idx in order:
            if z[idx] >= threshold:
                dsrna = "yes" if ar_mean[idx] > 0.3 else "low"
                print(f"{names[idx]:<25s} {z[idx]:>10.3f} {ar_mean[idx]:>12.3f} {dsrna:>8s}")
