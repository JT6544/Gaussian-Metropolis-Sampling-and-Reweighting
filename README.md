# Gaussian Metropolis Sampling and Reweighting

This project uses Markov chain Monte Carlo to study the one-dimensional
Gaussian system with Hamiltonian

```math
H(x)=x^2.
```

The program samples the corresponding Boltzmann distribution using a
random-walk Metropolis algorithm, estimates the mean energy with
correlation-aware uncertainties, and uses histogram reweighting to predict
results at other inverse temperatures.

The Gaussian model is exactly solvable, so it provides a useful controlled
test of:

- Metropolis sampling;
- thermalisation and acceptance rates;
- autocorrelation and binned errors;
- blocked-jackknife uncertainties for ratio estimators;
- importance-weight effective sample size;
- distributional overlap in reweighting.

A central result is that direct reweighting from $\beta=1$ to $\beta=0.5$
has divergent weight variance. The default study therefore uses an
intermediate ensemble at $\beta=0.75$ as an overlap-safe bridge while retaining
the unstable direct route as an explicit diagnostic.

## Repository Contents

```text
.
├── .gitignore
├── README.md
├── gaussian_metropolis.py
└── requirements.txt
```

The numerical study is self-contained and requires no external dataset. The
original coursework brief, generated figures, caches, and verification outputs
are deliberately excluded.

## Requirements

The project has been verified using:

| Component | Tested version |
|---|---:|
| Python | 3.12.13 |
| NumPy | 2.3.5 |
| Matplotlib | 3.10.8 |

The tested package versions are recorded in `requirements.txt`.

## Installation

From the repository directory, create and activate a virtual environment.

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Study

Run the full study from the repository root:

```bash
python gaussian_metropolis.py
```

This prints the direct and reweighted comparison tables and displays four
Matplotlib figures.

For a non-interactive numerical run without constructing figures:

```bash
python gaussian_metropolis.py --no-show
```

The default non-interactive run completed in approximately four seconds in the
verified environment. Runtime depends on the Python and NumPy builds in use.

## Physical Background

At inverse temperature $\beta>0$, the unnormalised Boltzmann distribution is

```math
p_\beta(x)\propto e^{-\beta H(x)}=e^{-\beta x^2}.
```

The partition function is

```math
Z(\beta)
=
\int_{-\infty}^{\infty}e^{-\beta x^2}\,dx
=
\sqrt{\frac{\pi}{\beta}}.
```

The normalized density is therefore

```math
p_\beta(x)
=
\sqrt{\frac{\beta}{\pi}}e^{-\beta x^2}.
```

The mean energy follows from either direct Gaussian integration or a
derivative of $\ln Z$:

```math
\left\langle H\right\rangle_\beta
=
\left\langle x^2\right\rangle_\beta
=
-\frac{\partial\ln Z}{\partial\beta}
=
\frac{1}{2\beta}.
```

This exact expression is used throughout the program to assess the numerical
estimates without influencing the sampling calculation.

## Main Aim

The numerical study tests whether:

1. direct Metropolis estimates reproduce
   $\langle x^2\rangle=1/(2\beta)$;
2. binned errors reflect the correlations within the Markov chains;
3. reweighting reproduces directly sampled results when the source and target
   distributions have adequate overlap;
4. an intermediate ensemble resolves the poor-overlap problem at
   $\beta=0.5$;
5. finite-sample diagnostics expose a reweighting route that is mathematically
   unstable.

The default inverse temperatures are

```python
betas = (0.5, 0.75, 1.0, 1.5, 2.0)
```

## What the Code Does

For each value of $\beta$, the program:

1. creates four deterministic but statistically independent random streams;
2. runs four Metropolis chains of 100,000 steps;
3. discards the first 5,000 steps of each chain;
4. combines 380,000 retained samples;
5. measures the acceptance rate;
6. estimates $\langle x^2\rangle$ and its binned standard error;
7. compares the result with $1/(2\beta)$;
8. evaluates overlap-safe reweighting routes;
9. calculates blocked-jackknife reweighting errors;
10. calculates weight-based effective sample sizes;
11. identifies whether each importance-weight variance is finite;
12. evaluates the unstable $1\to0.5$ route separately;
13. calculates the energy autocorrelation at $\beta=1$;
14. prints reproducible tables and optionally creates diagnostic figures.

## Metropolis Sampling

### Proposal distribution

From a current position $x$, the algorithm proposes

```math
x'=x+\delta,
```

where

```math
\delta\sim\mathcal{U}(-e,e).
```

The change in energy is

```math
\Delta H=H(x')-H(x)=x'^2-x^2.
```

The proposal is accepted with probability

```math
P_{\mathrm{accept}}
=
\min\left(1,e^{-\beta\Delta H}\right).
```

This rule accepts every downhill move and accepts uphill moves according to
their Boltzmann probability. Rejected proposals repeat the current state in the
chain, as required for a correct Markov process.

### Temperature-scaled proposal width

The characteristic width of the target Gaussian changes as
$1/\sqrt{\beta}$. The proposal half-width is therefore scaled using

```math
e(\beta)=\frac{2}{\sqrt{\beta}}.
```

This produces acceptance rates close to $0.514$ at all five temperatures. In a
controlled comparison at $\beta=1$, changing the proposal width from $1$ to
$2$ reduced the measured integrated energy-autocorrelation time from
approximately $6.84$ to $3.84$.

### Independent chains and reproducibility

The default seed is

```python
seed = 2025
```

`numpy.random.SeedSequence` spawns one independent child stream for every
chain and temperature. Repeating the default command therefore reproduces the
same results exactly, while the four chains do not reuse an identical random
sequence.

Changing the seed changes the finite Monte Carlo sample but should not change
the statistical conclusions.

## Correlated Statistical Analysis

### Thermalisation

Every chain starts at $x=0$. The first

```python
burn_in = 5_000
```

states are discarded before any observable is calculated. This prevents the
chosen initial condition from contributing directly to the reported sample.

### Binned mean and error

Successive Metropolis samples are correlated, so treating all retained states
as independent would underestimate the error.

For an observable $O$, the samples are divided into $B$ blocks of size 100.
The program calculates each block mean $\bar O_b$ and estimates the standard
error as

```math
\sigma_{\bar O}
=
\frac{s_{\mathrm{block}}}{\sqrt{B}},
```

where $s_{\mathrm{block}}$ is the sample standard deviation of the block
means. The block size is substantially larger than the measured correlation
scale of the default energy chains.

### Energy autocorrelation

The autocorrelation diagnostic is calculated for the energy observable
$H(x)=x^2$, because this is the quantity whose mean and uncertainty are
reported.

For lag $k$, the code uses the biased estimator

```math
C(k)
=
\frac{
\sum_{i=1}^{N-k}(H_i-\bar H)(H_{i+k}-\bar H)
}{
N\,\mathrm{Var}(H)
}.
```

The autocorrelation is evaluated only for the requested 100 lags rather than
calculating an unnecessary full $N\times N$ correlation. The four
$\beta=1$ chain estimates are averaged for the displayed diagnostic.

## Histogram Reweighting

Suppose samples were generated at $\beta_0$. An expectation value at a new
inverse temperature $\beta_1$ can be written as

```math
\langle O\rangle_{\beta_1}
=
\frac{
\left\langle O(x)w(x)\right\rangle_{\beta_0}
}{
\left\langle w(x)\right\rangle_{\beta_0}
},
```

with

```math
w(x)
=
e^{-(\beta_1-\beta_0)x^2}.
```

For the energy observable, the numerical estimator is

```math
\widehat{E}_{\beta_1}
=
\frac{\sum_i x_i^2w_i}{\sum_i w_i}.
```

The calculation subtracts the largest log weight before exponentiation. This
does not change the ratio, but prevents avoidable floating-point overflow and
underflow.

### Blocked-jackknife uncertainty

The reweighted estimate is a ratio of correlated sums. Its uncertainty is
therefore calculated with a delete-one-block jackknife rather than by applying
an independent-sample formula to the weights.

If $N_b$ and $D_b$ are the weighted numerator and denominator contributed by
block $b$, the leave-one-block-out estimate is

```math
\widehat{E}_{(b)}
=
\frac{N-N_b}{D-D_b}.
```

With $B$ blocks, the jackknife standard error is

```math
\sigma_{\mathrm{JK}}
=
\sqrt{
\frac{B-1}{B}
\sum_{b=1}^{B}
\left(
\widehat{E}_{(b)}-\overline{\widehat{E}}_{(\cdot)}
\right)^2
}.
```

When $\beta_1=\beta_0$, this implementation reduces numerically to the ordinary
binned standard error.

### Weight-based effective sample size

The program reports

```math
N_{\mathrm{eff},w}
=
\frac{\left(\sum_iw_i\right)^2}{\sum_iw_i^2}.
```

The displayed percentage is $N_{\mathrm{eff},w}/N$. This measures weight
concentration only; it is not an autocorrelation-adjusted count of independent
Markov samples.

## Distributional Overlap

Reweighting is reliable only when the source ensemble adequately samples the
regions that matter in the target ensemble.

For Gaussian reweighting, the second weight moment behaves as

```math
\left\langle w^2\right\rangle_{\beta_0}
\propto
\int_{-\infty}^{\infty}
e^{-(2\beta_1-\beta_0)x^2}\,dx.
```

It is finite only when

```math
2\beta_1>\beta_0.
```

### Unstable direct route

For direct reweighting from $\beta_0=1$ to $\beta_1=0.5$,

```math
2\beta_1-\beta_0=0.
```

Consequently,

```math
\left\langle w^2\right\rangle_{\beta=1}
\propto
\int_{-\infty}^{\infty}dx,
```

which diverges. A finite run can still produce a plausible point estimate and
a finite empirical effective sample size, but those observations do not remove
the theoretical instability. Rare tail samples can dominate as the run is
extended.

The program retains this calculation under `diagnostic_reweighted` and prints
`finite Var(w) = no`.

### Overlap-safe bridge

The primary estimate at $\beta=0.5$ instead uses samples generated at
$\beta=0.75$:

```math
0.75\longrightarrow0.5.
```

Here,

```math
2(0.5)-0.75=0.25>0,
```

so the weight variance is finite. The default primary routes are

```python
reweight_routes = (
    (1.0, 0.75),
    (1.0, 1.5),
    (1.0, 2.0),
    (0.75, 0.5),
)
```

## Default Configuration

| Setting | Default | Description |
|---|---:|---|
| `betas` | `(0.5, 0.75, 1.0, 1.5, 2.0)` | Directly sampled temperatures |
| `steps` | `100_000` | Metropolis steps per chain |
| `burn_in` | `5_000` | Discarded states per chain |
| `chains` | `4` | Independent chains per temperature |
| `proposal_scale` | `2.0` | Numerator in $e(\beta)=2/\sqrt{\beta}$ |
| `bin_size` | `100` | Binning and jackknife block size |
| `max_lag` | `100` | Number of autocorrelation lags |
| `seed` | `2025` | Reproducible master seed |
| `reference_beta` | `1.0` | Main reweighting source |
| `bridge_beta` | `0.75` | Source for the $\beta=0.5$ bridge |

A custom study can be run programmatically:

```python
from gaussian_metropolis import StudyConfig, print_summary, run_study


config = StudyConfig(
    steps=200_000,
    burn_in=5_000,
    chains=4,
    bin_size=100,
    seed=1234,
)

results = run_study(config=config, make_plots=False)
print_summary(results)
```

The retained samples in each chain must be divisible by `bin_size`, and every
reweighting route must use configured values of $\beta$.

## Verified Default Results

### Direct sampling

| $\beta$ | Exact $\langle x^2\rangle$ | Direct estimate | Relative error | Acceptance |
|---:|---:|---:|---:|---:|
| 0.50 | 1.000000 | $0.998090\pm0.004386$ | 0.191% | 0.514 |
| 0.75 | 0.666667 | $0.667636\pm0.002929$ | 0.145% | 0.515 |
| 1.00 | 0.500000 | $0.497741\pm0.002221$ | 0.452% | 0.513 |
| 1.50 | 0.333333 | $0.334588\pm0.001494$ | 0.376% | 0.515 |
| 2.00 | 0.250000 | $0.249580\pm0.001102$ | 0.168% | 0.514 |

### Primary reweighting

| Route | Exact | Reweighted estimate | Relative error | Weight ESS | Finite $\mathrm{Var}(w)$ |
|---:|---:|---:|---:|---:|:---:|
| $1.00\to0.75$ | 0.666667 | $0.666419\pm0.004903$ | 0.037% | 94.19% | Yes |
| $1.00\to1.50$ | 0.333333 | $0.331263\pm0.001073$ | 0.621% | 94.30% | Yes |
| $1.00\to2.00$ | 0.250000 | $0.248449\pm0.000770$ | 0.620% | 86.67% | Yes |
| $0.75\to0.50$ | 1.000000 | $1.000977\pm0.009543$ | 0.098% | 86.69% | Yes |

### Unstable route diagnostic

| Route | Exact | Estimate | Relative error | Weight ESS | Finite $\mathrm{Var}(w)$ |
|---:|---:|---:|---:|---:|:---:|
| $1.00\to0.50$ | 1.000000 | $1.004183\pm0.017768$ | 0.418% | 50.78% | **No** |

The diagnostic point estimate happens to be close to the exact answer for the
default seed. It is not promoted to a primary result because closeness in one
finite run does not repair its divergent theoretical weight variance.

## Independent Seed Check

The bridge was also tested in ten separate four-chain studies using different
master seeds. These verification runs produced:

| Diagnostic | Result |
|---|---:|
| Minimum bridge estimate | 0.985303 |
| Median bridge estimate | 1.004565 |
| Maximum bridge estimate | 1.012189 |
| Mean bridge estimate | 1.001860 |
| Between-study standard deviation | 0.007718 |
| Median reported standard error | 0.009884 |
| Weight ESS range | 85.8%–87.5% |
| Exact-value coverage within two reported errors | 10/10 |

These runs are verification evidence rather than a seed-selection procedure.
The default seed remained fixed before the comparison.

## Terminal Output

The program prints two tables.

### Direct table

For every directly sampled value of $\beta$, this reports:

- exact and measured energy;
- binned standard error;
- relative error;
- proposal width;
- mean acceptance rate across four chains.

### Reweighting table

For every primary and diagnostic route, this reports:

- source and target $\beta$;
- route role;
- reweighted estimate and jackknife error;
- relative error;
- weight-based effective sample fraction;
- whether the analytical weight variance is finite.

## Plots

The program creates four Matplotlib figures.

### Sample distributions

Five histogram panels compare the retained Metropolis samples with the exact
normalized Gaussian density at each value of $\beta$.

### Energy autocorrelation

The average $\beta=1$ energy autocorrelation is plotted for the first 100 lags.
The rapid decay demonstrates the improvement from the scaled proposal width and
supports the use of 100-sample analysis blocks.

### Energy comparison

The exact curve, direct measurements, and overlap-safe reweighted estimates are
displayed together with their reported errors.

### Reweighting-overlap diagnostic

At $\beta=0.5$, the plot compares:

- direct sampling;
- the primary $0.75\to0.5$ bridge;
- the unstable $1\to0.5$ route;
- the exact analytical energy.

The unstable route is shown with its wider uncertainty and is labelled
separately from the primary estimate.

Figures are displayed using `plt.show()` and are not saved automatically.

## Returned Results

`run_study()` returns a dictionary with these top-level keys:

| Key | Description |
|---|---|
| `config` | Validated `StudyConfig` instance |
| `direct` | Direct results keyed by $\beta$ |
| `reweighted` | Primary results keyed by `(beta_old, beta_new)` |
| `diagnostic_reweighted` | Unstable diagnostic routes |
| `autocorrelation` | Mean energy autocorrelation across reference chains |
| `chain_autocorrelations` | Individual reference-chain correlations |
| `figures` | Tuple of four figures, or an empty tuple when disabled |

Example:

```python
from gaussian_metropolis import run_study


results = run_study(make_plots=False)

direct_beta_one = results["direct"][1.0]
bridge = results["reweighted"][(0.75, 0.5)]
unstable = results["diagnostic_reweighted"][(1.0, 0.5)]

print(direct_beta_one["energy"], direct_beta_one["error"])
print(bridge["energy"], bridge["error"])
print(unstable["finite_weight_variance"])
```

Each direct result contains:

| Key | Description |
|---|---|
| `samples` | Combined retained samples |
| `chain_samples` | Retained samples from each independent chain |
| `energy` | Direct mean-energy estimate |
| `error` | Binned standard error |
| `exact` | Analytical mean energy |
| `relative_error` | Absolute relative deviation from exact |
| `acceptance_rate` | Mean acceptance rate across chains |
| `chain_acceptance_rates` | Individual-chain acceptance rates |
| `proposal_width` | Temperature-scaled proposal half-width |

Each reweighted result contains the estimate, jackknife error, exact value,
relative error, weight effective sample size, effective fraction, and
finite-weight-variance flag.

## Main Functions

| Function | Description |
|---|---|
| `StudyConfig` | Stores the numerical configuration |
| `validate_config()` | Validates temperatures, chains, blocks, and routes |
| `analytical_energy()` | Returns $1/(2\beta)$ |
| `analytical_density()` | Evaluates the exact normalized density |
| `metropolis_gaussian()` | Runs one random-walk Metropolis chain |
| `binned_mean_error()` | Estimates a correlated mean and error |
| `autocorrelation()` | Calculates a lag-limited biased estimator |
| `reweighted_energy()` | Calculates a stable ratio estimate and jackknife error |
| `has_finite_weight_variance()` | Applies the Gaussian overlap criterion |
| `run_study()` | Runs all direct chains and reweighting routes |
| `create_figures()` | Creates the four diagnostic figures |
| `print_summary()` | Prints the reproducible result tables |

## Repository Preparation and Verification

This repository develops the original coursework implementation into a
reproducible portfolio project while preserving its physical model and core
Metropolis method. The maintenance and numerical-validation work includes:

- renaming the script to an importable filename;
- separating simulation, analysis, plotting, and entrypoint behaviour;
- adding configuration validation and descriptive errors;
- adding deterministic independent random streams;
- scaling proposals with the target Gaussian width;
- replacing a full autocorrelation calculation with a lag-limited equivalent;
- adding blocked-jackknife errors and stable weights;
- adding analytical overlap and effective-size diagnostics;
- introducing the $\beta=0.75$ bridge;
- retaining the unstable direct route as a disclosed limitation;
- declaring the tested dependencies.

Verification included:

- syntax and direct-entrypoint checks;
- exact reproduction under the default seed;
- parity between the lag-limited and original autocorrelation estimators;
- parity between shifted and unshifted weights in their safe range;
- a jackknife identity check at unchanged temperature;
- analytical agreement within three reported errors;
- acceptance-rate and chain-structure checks;
- explicit finite-variance tests for every route;
- ten independent four-chain bridge studies;
- visual inspection of all four figures.

The original source and assignment brief remain unchanged and are not included
in this repository.

## Numerical Considerations

### Exactly solvable benchmark

The exact result is known before sampling. This makes the project useful for
validating numerical methodology, but it is not an application to an unknown
physical observable.

### Markov-chain dependence

Multiple chains, proposal tuning, autocorrelation analysis, and blocking reduce
the risk of underestimated errors. They do not make individual retained states
independent.

### Block-size dependence

The default block size is conservative relative to the measured energy
correlation scale. Custom configurations with slower mixing should re-evaluate
the block size rather than assuming 100 is always sufficient.

### Effective sample size

The reported effective size measures importance-weight concentration. It does
not incorporate the Markov autocorrelation and must not be interpreted as a
complete count of independent samples.

### Finite-sample overlap diagnostics

A high empirical effective fraction cannot prove that the theoretical weight
variance is finite. The program therefore reports both the finite-sample metric
and the analytical Gaussian criterion.

### Bridge dependence

The $0.75\to0.5$ route is stable for this model because its weight variance is
finite and its distributions overlap well. More distant targets may require
additional intermediate ensembles or a multihistogram method.

### Random-seed dependence

The fixed seed makes the documented results reproducible. Statistical validity
is supported by independent-seed checks rather than by selecting a seed that
produces the closest point estimate.

## Expected Behaviour

A successful default run should show:

- acceptance rates close to 0.51 at every temperature;
- histograms that follow the exact Gaussian densities;
- rapidly decaying energy autocorrelation;
- direct results consistent with $1/(2\beta)$;
- primary reweighted results with finite weight variance;
- a bridge estimate near the direct $\beta=0.5$ result;
- high weight effective fractions for the primary routes;
- the direct $1\to0.5$ route explicitly marked as unstable;
- identical output when the same seed and configuration are reused.

## Coursework Context

The original program was written by Jack Turner for PH-353 Computational
Physics coursework at Swansea University in 2025. This repository presents the
numerical study as an academic and portfolio project with additional
reproducibility, validation, diagnostics, and documentation.

The lecturer-provided assignment sheet and the separate written submission are
not distributed with the repository.

## Licence

No open-source licence has currently been applied.

Copyright remains with Jack Turner. The absence of a licence means that
permission to copy, modify, or redistribute the code should not be assumed.
