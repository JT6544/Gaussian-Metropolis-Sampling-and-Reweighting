"""Metropolis sampling and histogram reweighting for a Gaussian system.

The target distribution is proportional to ``exp(-beta * x**2)``.  The
observable called energy throughout the study is therefore ``H(x) = x**2``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class StudyConfig:
    """Numerical settings for the default Gaussian-system study."""

    betas: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
    steps: int = 100_000
    burn_in: int = 5_000
    chains: int = 4
    proposal_scale: float = 2.0
    bin_size: int = 100
    max_lag: int = 100
    seed: int = 2025
    reference_beta: float = 1.0
    bridge_beta: float = 0.75
    reweight_routes: tuple[tuple[float, float], ...] = (
        (1.0, 0.75),
        (1.0, 1.5),
        (1.0, 2.0),
        (0.75, 0.5),
    )
    diagnostic_routes: tuple[tuple[float, float], ...] = ((1.0, 0.5),)

    def proposal_width(self, beta: float) -> float:
        """Return a temperature-scaled random-walk proposal half-width."""

        return self.proposal_scale / np.sqrt(beta)


def validate_config(config: StudyConfig) -> None:
    """Raise ``ValueError`` when a study configuration is invalid."""

    if not config.betas:
        raise ValueError("At least one beta value is required.")
    if len(set(config.betas)) != len(config.betas):
        raise ValueError("Beta values must be unique.")
    if any(beta <= 0 or not np.isfinite(beta) for beta in config.betas):
        raise ValueError("All beta values must be positive and finite.")
    if config.reference_beta not in config.betas:
        raise ValueError("reference_beta must be included in betas.")
    if config.bridge_beta not in config.betas:
        raise ValueError("bridge_beta must be included in betas.")
    if config.proposal_scale <= 0 or not np.isfinite(config.proposal_scale):
        raise ValueError("proposal_scale must be positive and finite.")
    if config.steps <= 0:
        raise ValueError("The number of Metropolis steps must be positive.")
    if not 0 <= config.burn_in < config.steps:
        raise ValueError("burn_in must satisfy 0 <= burn_in < steps.")
    if config.chains <= 0:
        raise ValueError("chains must be positive.")

    retained_per_chain = config.steps - config.burn_in
    retained = config.chains * retained_per_chain
    if config.bin_size <= 0 or retained // config.bin_size < 2:
        raise ValueError("bin_size must leave at least two complete bins.")
    if retained_per_chain % config.bin_size:
        raise ValueError("bin_size must divide the retained samples in each chain.")
    if not 1 <= config.max_lag <= retained_per_chain:
        raise ValueError(
            "max_lag must be between 1 and the retained samples per chain."
        )
    if not isinstance(config.seed, (int, np.integer)) or config.seed < 0:
        raise ValueError("seed must be a non-negative integer.")

    all_routes = config.reweight_routes + config.diagnostic_routes
    if not all_routes:
        raise ValueError("At least one reweighting route is required.")
    for beta_old, beta_new in all_routes:
        if beta_old not in config.betas or beta_new not in config.betas:
            raise ValueError("Every reweighting route must use configured beta values.")
        if beta_old == beta_new:
            raise ValueError("A reweighting route must connect different beta values.")

    primary_targets = [beta_new for _, beta_new in config.reweight_routes]
    if len(set(primary_targets)) != len(primary_targets):
        raise ValueError("Primary reweighting targets must be unique.")

    lowest_beta = min(config.betas)
    if (config.bridge_beta, lowest_beta) not in config.reweight_routes:
        raise ValueError("The primary routes must include bridge_beta -> lowest beta.")
    if (config.reference_beta, lowest_beta) not in config.diagnostic_routes:
        raise ValueError(
            "The diagnostic routes must include reference_beta -> lowest beta."
        )


def analytical_energy(beta: float) -> float:
    """Return the exact mean energy, ``<x^2> = 1 / (2 beta)``."""

    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be positive and finite.")
    return 1.0 / (2.0 * beta)


def analytical_density(x: np.ndarray, beta: float) -> np.ndarray:
    """Evaluate the normalized Gaussian probability density at ``x``."""

    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be positive and finite.")
    return np.sqrt(beta / np.pi) * np.exp(-beta * np.asarray(x) ** 2)


def metropolis_gaussian(
    beta: float,
    proposal_width: float,
    steps: int,
    burn_in: int,
    rng: np.random.Generator,
    initial_x: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Sample ``exp(-beta*x^2)`` using a random-walk Metropolis chain."""

    if beta <= 0 or not np.isfinite(beta):
        raise ValueError("beta must be positive and finite.")
    if proposal_width <= 0 or not np.isfinite(proposal_width):
        raise ValueError("proposal_width must be positive and finite.")
    if steps <= 0 or not 0 <= burn_in < steps:
        raise ValueError("Require steps > 0 and 0 <= burn_in < steps.")
    if not np.isfinite(initial_x):
        raise ValueError("initial_x must be finite.")

    chain = np.empty(steps, dtype=float)
    x = float(initial_x)
    accepted = 0

    for index in range(steps):
        proposal = x + proposal_width * (2.0 * rng.random() - 1.0)
        delta_energy = proposal**2 - x**2

        if delta_energy <= 0.0 or rng.random() <= np.exp(-beta * delta_energy):
            x = proposal
            accepted += 1

        chain[index] = x

    return chain[burn_in:], accepted / steps


def binned_mean_error(
    observations: np.ndarray,
    bin_size: int = 20,
) -> tuple[float, float]:
    """Estimate an observable mean and its standard error using binning."""

    values = np.asarray(observations, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("observations must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("observations must contain only finite values.")
    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    number_of_bins = values.size // bin_size
    if number_of_bins < 2:
        raise ValueError("At least two complete bins are required.")

    used = values[: number_of_bins * bin_size]
    bin_means = used.reshape(number_of_bins, bin_size).mean(axis=1)
    standard_error = bin_means.std(ddof=1) / np.sqrt(number_of_bins)
    return float(values.mean()), float(standard_error)


def autocorrelation(samples: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Calculate the original biased autocorrelation estimator up to ``max_lag``."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    if not 1 <= max_lag <= values.size:
        raise ValueError("max_lag must be between 1 and the sample count.")

    centered = values - values.mean()
    variance = centered.var()
    if variance == 0.0:
        raise ValueError("Autocorrelation is undefined for zero-variance samples.")

    normalization = variance * values.size
    result = np.empty(max_lag, dtype=float)
    result[0] = 1.0
    for lag in range(1, max_lag):
        result[lag] = np.dot(centered[:-lag], centered[lag:]) / normalization
    return result


def reweighted_energy(
    samples: np.ndarray,
    beta_new: float,
    beta_old: float,
    block_size: int = 20,
) -> tuple[float, float, float]:
    """Return a reweighted energy, blocked-jackknife error, and effective size."""

    values = np.asarray(samples, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("samples must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values.")
    if (
        beta_new <= 0
        or beta_old <= 0
        or not np.isfinite(beta_new)
        or not np.isfinite(beta_old)
    ):
        raise ValueError("beta_new and beta_old must be positive.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    number_of_blocks = values.size // block_size
    if number_of_blocks < 2:
        raise ValueError("At least two complete jackknife blocks are required.")

    energies = values**2
    log_weights = -(beta_new - beta_old) * energies
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)

    total_weight = weights.sum()
    total_weighted_energy = np.dot(weights, energies)
    estimate = total_weighted_energy / total_weight
    effective_sample_size = total_weight**2 / np.dot(weights, weights)

    used_count = number_of_blocks * block_size
    used_weights = weights[:used_count].reshape(number_of_blocks, block_size)
    used_energies = energies[:used_count].reshape(number_of_blocks, block_size)
    block_denominators = used_weights.sum(axis=1)
    block_numerators = (used_weights * used_energies).sum(axis=1)
    used_denominator = block_denominators.sum()
    used_numerator = block_numerators.sum()

    leave_one_out = (used_numerator - block_numerators) / (
        used_denominator - block_denominators
    )
    jackknife_mean = leave_one_out.mean()
    jackknife_error = np.sqrt(
        (number_of_blocks - 1)
        / number_of_blocks
        * np.sum((leave_one_out - jackknife_mean) ** 2)
    )

    return float(estimate), float(jackknife_error), float(effective_sample_size)


def has_finite_weight_variance(beta_new: float, beta_old: float) -> bool:
    """Return whether the Gaussian importance weights have a finite variance."""

    if (
        beta_new <= 0
        or beta_old <= 0
        or not np.isfinite(beta_new)
        or not np.isfinite(beta_old)
    ):
        raise ValueError("beta_new and beta_old must be positive.")
    return 2.0 * beta_new > beta_old


def run_study(
    config: StudyConfig | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Run the direct and reweighted Gaussian-energy comparison."""

    config = config or StudyConfig()
    validate_config(config)

    seed_sequence = np.random.SeedSequence(config.seed)
    child_sequences = iter(seed_sequence.spawn(len(config.betas) * config.chains))
    direct: dict[float, dict[str, Any]] = {}

    for beta in config.betas:
        chain_samples = []
        acceptance_rates = []
        for _ in range(config.chains):
            samples, acceptance_rate = metropolis_gaussian(
                beta=beta,
                proposal_width=config.proposal_width(beta),
                steps=config.steps,
                burn_in=config.burn_in,
                rng=np.random.default_rng(next(child_sequences)),
            )
            chain_samples.append(samples)
            acceptance_rates.append(acceptance_rate)

        combined_samples = np.concatenate(chain_samples)
        energy, error = binned_mean_error(combined_samples**2, config.bin_size)
        exact = analytical_energy(beta)
        direct[beta] = {
            "samples": combined_samples,
            "chain_samples": tuple(chain_samples),
            "energy": energy,
            "error": error,
            "exact": exact,
            "relative_error": abs(energy - exact) / exact,
            "acceptance_rate": float(np.mean(acceptance_rates)),
            "chain_acceptance_rates": tuple(acceptance_rates),
            "proposal_width": config.proposal_width(beta),
        }

    def calculate_routes(
        routes: tuple[tuple[float, float], ...],
    ) -> dict[tuple[float, float], dict[str, float | bool]]:
        route_results: dict[tuple[float, float], dict[str, float | bool]] = {}
        for beta_old, beta_new in routes:
            source_samples = direct[beta_old]["samples"]
            energy, error, effective_size = reweighted_energy(
                source_samples,
                beta_new=beta_new,
                beta_old=beta_old,
                block_size=config.bin_size,
            )
            exact = analytical_energy(beta_new)
            route_results[(beta_old, beta_new)] = {
                "energy": energy,
                "error": error,
                "exact": exact,
                "relative_error": abs(energy - exact) / exact,
                "effective_sample_size": effective_size,
                "effective_fraction": effective_size / source_samples.size,
                "finite_weight_variance": has_finite_weight_variance(
                    beta_new,
                    beta_old,
                ),
            }
        return route_results

    reweighted = calculate_routes(config.reweight_routes)
    diagnostic_reweighted = calculate_routes(config.diagnostic_routes)

    reference_chains = direct[config.reference_beta]["chain_samples"]
    correlations = np.vstack(
        [
            autocorrelation(chain**2, config.max_lag)
            for chain in reference_chains
        ]
    )
    correlation = correlations.mean(axis=0)

    results: dict[str, Any] = {
        "config": config,
        "direct": direct,
        "reweighted": reweighted,
        "diagnostic_reweighted": diagnostic_reweighted,
        "autocorrelation": correlation,
        "chain_autocorrelations": correlations,
        "figures": (),
    }
    if make_plots:
        results["figures"] = create_figures(results)
    return results


def create_figures(results: dict[str, Any]) -> tuple[plt.Figure, ...]:
    """Create sampling, correlation, comparison, and overlap figures."""

    config: StudyConfig = results["config"]
    direct: dict[float, dict[str, Any]] = results["direct"]
    reweighted = results["reweighted"]
    diagnostic_reweighted = results["diagnostic_reweighted"]

    columns = 2
    rows = int(np.ceil(len(config.betas) / columns))
    histogram_figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11, 4.2 * rows),
        squeeze=False,
    )
    flat_axes = axes.ravel()
    for axis, beta in zip(flat_axes, config.betas, strict=False):
        samples = direct[beta]["samples"]
        lower, upper = np.quantile(samples, [0.001, 0.999])
        x_grid = np.linspace(lower, upper, 500)
        axis.hist(
            samples,
            bins=100,
            density=True,
            alpha=0.65,
            color="tab:blue",
            edgecolor="black",
            linewidth=0.35,
            label="Metropolis samples",
        )
        axis.plot(
            x_grid,
            analytical_density(x_grid, beta),
            color="tab:red",
            linewidth=2,
            label="Exact density",
        )
        axis.set(
            title=rf"$\beta={beta:g}$",
            xlabel="$x$",
            ylabel="Probability density",
        )
        axis.legend()
    for axis in flat_axes[len(config.betas) :]:
        axis.remove()
    histogram_figure.suptitle("Gaussian Metropolis samples", fontsize=14)
    histogram_figure.tight_layout()

    autocorrelation_figure, autocorrelation_axis = plt.subplots(figsize=(8, 4.5))
    lags = np.arange(config.max_lag)
    autocorrelation_axis.plot(
        lags,
        results["autocorrelation"],
        marker=".",
        linewidth=1.2,
        color="tab:red",
    )
    autocorrelation_axis.axhline(0.0, color="black", linewidth=0.8)
    autocorrelation_axis.set(
        title=rf"Energy autocorrelation at $\beta={config.reference_beta:g}$",
        xlabel="Lag",
        ylabel=r"Autocorrelation of $x^2$",
    )
    autocorrelation_axis.grid(alpha=0.3)
    autocorrelation_figure.tight_layout()

    comparison_figure, comparison_axis = plt.subplots(figsize=(8, 5))
    beta_values = np.array(sorted(config.betas))
    exact_values = np.array([analytical_energy(beta) for beta in beta_values])
    comparison_axis.plot(beta_values, exact_values, "k-", label="Exact")
    comparison_axis.errorbar(
        beta_values,
        [direct[beta]["energy"] for beta in beta_values],
        yerr=[direct[beta]["error"] for beta in beta_values],
        fmt="o",
        capsize=4,
        color="tab:blue",
        label="Direct sampling",
    )
    primary_routes = sorted(reweighted, key=lambda route: route[1])
    comparison_axis.errorbar(
        [route[1] for route in primary_routes],
        [reweighted[route]["energy"] for route in primary_routes],
        yerr=[reweighted[route]["error"] for route in primary_routes],
        fmt="s",
        capsize=4,
        color="tab:orange",
        label="Overlap-safe reweighting",
    )
    comparison_axis.set(
        title="Mean Gaussian energy",
        xlabel=r"Inverse temperature $\beta$",
        ylabel=r"$\langle x^2\rangle$",
    )
    comparison_axis.grid(alpha=0.3)
    comparison_axis.legend()
    comparison_figure.tight_layout()

    bridge_route = (config.bridge_beta, min(config.betas))
    unstable_route = (config.reference_beta, min(config.betas))
    bridge_result = reweighted[bridge_route]
    unstable_result = diagnostic_reweighted[unstable_route]
    target_beta = min(config.betas)
    overlap_figure, overlap_axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(3)
    overlap_axis.errorbar(
        positions,
        [
            direct[target_beta]["energy"],
            bridge_result["energy"],
            unstable_result["energy"],
        ],
        yerr=[
            direct[target_beta]["error"],
            bridge_result["error"],
            unstable_result["error"],
        ],
        fmt="o",
        capsize=5,
        color="tab:blue",
    )
    overlap_axis.plot(positions[1], bridge_result["energy"], "s", color="tab:green")
    overlap_axis.plot(positions[2], unstable_result["energy"], "X", color="tab:red")
    overlap_axis.axhline(
        analytical_energy(target_beta),
        color="black",
        linewidth=1.5,
        label="Exact",
    )
    overlap_axis.set_xticks(
        positions,
        [
            rf"Direct $\beta={target_beta:g}$",
            rf"Bridge ${config.bridge_beta:g}\to{target_beta:g}$",
            rf"Unstable ${config.reference_beta:g}\to{target_beta:g}$",
        ],
    )
    overlap_axis.set(
        title=rf"Reweighting overlap at $\beta={target_beta:g}$",
        ylabel=r"$\langle x^2\rangle$",
    )
    overlap_axis.grid(axis="y", alpha=0.3)
    overlap_axis.legend()
    overlap_figure.tight_layout()

    return (
        histogram_figure,
        autocorrelation_figure,
        comparison_figure,
        overlap_figure,
    )


def print_summary(results: dict[str, Any]) -> None:
    """Print direct, primary reweighted, and diagnostic results."""

    config: StudyConfig = results["config"]
    direct: dict[float, dict[str, Any]] = results["direct"]

    print("Gaussian Metropolis sampling and reweighting")
    print(f"Seed: {config.seed}")
    print(f"Independent chains per beta: {config.chains}")
    print(f"Steps per chain: {config.steps:,}")
    retained_per_beta = config.chains * (config.steps - config.burn_in)
    print(f"Retained samples per beta: {retained_per_beta:,}")
    print(f"Jackknife block size: {config.bin_size}")
    print()
    print("Direct sampling")
    print(
        " beta |   exact |              estimate | rel error | proposal | accept"
    )
    print("-" * 78)
    for beta in config.betas:
        result = direct[beta]
        print(
            f"{beta:5.2f} | {result['exact']:7.4f} | "
            f"{result['energy']:9.6f} +/- {result['error']:8.6f} | "
            f"{100 * result['relative_error']:8.3f}% | "
            f"{result['proposal_width']:8.4f} | {result['acceptance_rate']:6.3f}"
        )

    print()
    print("Reweighting")
    print(
        "       route | role       |              estimate | rel error |"
        " weight ESS | finite Var(w)"
    )
    print("-" * 105)
    route_groups = (
        ("primary", results["reweighted"]),
        ("diagnostic", results["diagnostic_reweighted"]),
    )
    for role, route_results in route_groups:
        for (beta_old, beta_new), result in route_results.items():
            finite_variance = "yes" if result["finite_weight_variance"] else "no"
            print(
                f"{beta_old:4.2f}->{beta_new:<4.2f} | {role:10s} | "
                f"{result['energy']:9.6f} +/- {result['error']:8.6f} | "
                f"{100 * result['relative_error']:8.3f}% | "
                f"{100 * result['effective_fraction']:9.2f}% | "
                f"{finite_variance:^13s}"
            )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the direct-entrypoint workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="run the numerical study without creating or displaying plots",
    )
    return parser.parse_args()


def main() -> dict[str, Any]:
    """Run the default reproducible study and display its figures."""

    args = parse_args()
    results = run_study(make_plots=not args.no_show)
    print_summary(results)
    if not args.no_show:
        plt.show()
    return results


if __name__ == "__main__":
    main()
