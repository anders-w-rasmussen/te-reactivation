"""
SVI inference for the TE reactivation model.
Uses TraceEnum_ELBO for exact marginalization of discrete spike-and-slab variables.
Computes marginal P(z_f=1) by evaluating log-likelihood under z=0 vs z=1.
"""

import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
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
    Run stochastic variational inference with discrete enumeration.

    Returns dict with:
      - losses: ELBO loss per step
      - z_posterior: posterior P(reactivated) per family
      - params: all variational parameters
    """
    pyro.clear_param_store()

    sense, antisense, lengths = prepare_tensors(data)

    enum_model = config_enumerate(te_reactivation_model, default="parallel")

    optimizer = ClippedAdam({"lr": lr, "betas": (0.9, 0.999)})
    elbo = TraceEnum_ELBO(max_plate_nesting=1)
    svi = SVI(enum_model, te_reactivation_guide, optimizer, loss=elbo)

    losses = []
    for step in range(n_steps):
        loss = svi.step(sense, antisense, lengths, data.n_families)
        losses.append(loss)
        if print_every and (step + 1) % print_every == 0:
            print(f"Step {step+1:>5d} | ELBO loss: {loss:>10.1f}")

    # Compute posterior P(z_f = 1) by log-likelihood comparison
    z_posterior = _compute_z_posteriors_ll(sense, antisense, lengths, data.n_families)

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
def _compute_z_posteriors_ll(sense, antisense, lengths, n_families, n_mc=100):
    """
    Compute marginal P(z_f=1 | data) for each family.

    For each MC sample from the guide:
      - Fix continuous latents at guide sample
      - For each family, compute log p(data_f | z_f=0, theta) and log p(data_f | z_f=1, theta)
      - Combine with prior pi to get posterior
    Average over MC samples.
    """
    log_odds_samples = np.zeros((n_mc, n_families))

    for s in range(n_mc):
        # Sample continuous params from guide
        guide_trace = pyro.poutine.trace(te_reactivation_guide).get_trace(
            sense, antisense, lengths, n_families
        )

        pi = guide_trace.nodes["pi"]["value"]
        inv_disp = guide_trace.nodes["inv_disp"]["value"]
        # Shared background
        log_bg_s_global = guide_trace.nodes["log_bg_sense"]["value"]
        log_bg_a_global = guide_trace.nodes["log_bg_antisense"]["value"]

        for f in range(n_families):
            n_loci = sense[f].shape[0]
            lengths_kb = lengths[f] / 1000.0

            log_fg = guide_trace.nodes[f"log_fg_{f}"]["value"]
            ar = guide_trace.nodes[f"antisense_ratio_{f}"]["value"]

            bg_s = torch.exp(log_bg_s_global) * lengths_kb
            bg_a = torch.exp(log_bg_a_global) * lengths_kb
            fg_s = torch.exp(log_fg) * lengths_kb
            fg_a = torch.exp(log_fg) * ar * lengths_kb

            # z=0: only background
            mu_s_0 = bg_s + 1e-6
            mu_a_0 = bg_a + 1e-6
            # z=1: background + foreground
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

            # Log odds: log p(z=1|data) - log p(z=0|data) = (ll_1 - ll_0) + log(pi/(1-pi))
            log_prior_odds = torch.log(pi + 1e-8) - torch.log(1 - pi + 1e-8)
            log_odds_samples[s, f] = (ll_1 - ll_0 + log_prior_odds).item()

    # Average log-odds across MC samples, then convert to probability
    mean_log_odds = log_odds_samples.mean(axis=0)
    z_posterior = 1.0 / (1.0 + np.exp(-mean_log_odds))

    return z_posterior


def summarize_results(results: dict, threshold: float = 0.5) -> None:
    """Print a summary of reactivation calls."""
    print("\n" + "=" * 60)
    print("TE Family Reactivation Results")
    print("=" * 60)
    print(f"{'Family':<20s} {'P(reactivated)':>15s} {'Call':>8s}")
    print("-" * 60)

    z = results["z_posterior"]
    names = results["family_names"]

    order = np.argsort(-z)
    for idx in order:
        call = "ACTIVE" if z[idx] >= threshold else "silent"
        print(f"{names[idx]:<20s} {z[idx]:>15.3f} {call:>8s}")

    n_active = (z >= threshold).sum()
    print("-" * 60)
    print(f"Total reactivated: {int(n_active)} / {len(z)} families (threshold={threshold})")

    # Report antisense ratios for active families
    params = results["params"]
    print(f"\n{'Family':<20s} {'P(react)':>10s} {'AS/S ratio':>12s} {'dsRNA?':>8s}")
    print("-" * 60)
    for idx in order:
        if z[idx] >= threshold:
            ar_a = params.get(f"ar_alpha_{idx}", 2.0)
            ar_b = params.get(f"ar_beta_{idx}", 3.0)
            if isinstance(ar_a, np.ndarray):
                ar_a = ar_a.item()
                ar_b = ar_b.item()
            ar_mean = ar_a / (ar_a + ar_b)
            dsrna = "yes" if ar_mean > 0.3 else "low"
            print(f"{names[idx]:<20s} {z[idx]:>10.3f} {ar_mean:>12.3f} {dsrna:>8s}")
