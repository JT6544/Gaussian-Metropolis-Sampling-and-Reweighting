# Improvement Record: Gaussian Metropolis Sampling and Reweighting — Gate 3

## Archive represented

This document describes the changes made when producing:

```text
Gaussian-Metropolis-Sampling-and-Reweighting-Gate3.zip
```

The accompanying `CA2.py` is the untouched original source used as the baseline. Its SHA-256 checksum is:

```text
01fa02ca4e880d5cefcd78a79d7ab7350f562695d6604934900010e6d2c72d76  CA2.py
```

## Executive summary

The original coursework script demonstrated random-walk Metropolis sampling of

$$
p_\beta(x)\propto e^{-\beta x^2}
$$

and basic histogram reweighting. The rebuilt repository preserves that physical model while turning the script into a reproducible numerical study with independent chains, correlation-aware errors, stable ratio estimation, overlap diagnostics, analytical validation, configurable execution, and publication-quality documentation.

The most important scientific change was the treatment of reweighting from $\beta=1$ to $\beta=0.5$. The original script reported this estimate without diagnosing that its importance weights have divergent variance. The rebuild retains the direct route as a warning example and introduces a stable bridge through $\beta=0.75$.

## Original implementation limitations

The baseline script had the following limitations:

- one global NumPy random stream and no fixed seed;
- one chain per temperature;
- plotting embedded inside the sampler;
- proposal widths selected manually for individual temperatures;
- a full autocorrelation calculation when only the first 100 lags were used;
- one small fixed bin size without an explicit correlation-scale check;
- reweighted point estimates without uncertainty estimates;
- no effective-sample-size or overlap diagnostics;
- no warning that the $1\to0.5$ weight variance is infinite;
- top-level execution on import;
- no configuration object, input validation, CLI, dependency declaration, or detailed README.

## Improvement summary

| Area | Improvement | Why it was made | Impact |
|---|---|---|---|
| Structure | Split simulation, statistics, reweighting, plotting, reporting, and entrypoint logic into functions | The original sampler mixed calculation and display behaviour | Functions can be imported, tested, and reused independently |
| Configuration | Added a validated `StudyConfig` data class | Hard-coded values made controlled experiments difficult | Temperatures, chain count, burn-in, block size, seed, and routes are explicit and reproducible |
| Randomness | Added a fixed master seed and independent `SeedSequence` child streams | Reusing one implicit stream obscured reproducibility and chain independence | Default output is exactly repeatable while chains remain statistically separate |
| Sampling | Ran four chains at each temperature | One chain gives weak evidence about mixing and seed dependence | The reference estimate combines 380,000 retained samples per temperature |
| Proposal scale | Used $e(\beta)=2/\sqrt{\beta}$ | The target Gaussian width varies as $1/\sqrt{\beta}$ | Acceptance rates remain close to 0.51 across all temperatures |
| Error analysis | Replaced the minimal bin analysis with correlation-aware blocking | Markov samples are not independent | Reported uncertainties reflect within-chain correlation more realistically |
| Autocorrelation | Calculated only the requested lags for the energy observable | The original full correlation calculation was unnecessarily expensive | Memory and runtime scale with the requested lag range rather than the full chain length |
| Reweighting | Added log-shifted weights and blocked jackknife errors | Direct exponentiation and point estimates alone can be unstable or misleading | Ratio estimates are numerically stable and carry uncertainty estimates |
| Overlap | Added weight effective sample size and an analytical finite-variance test | A plausible point estimate can conceal poor distributional overlap | Stable and unstable routes are distinguished explicitly |
| Bridge ensemble | Added direct sampling at $\beta=0.75$ and used $0.75\to0.5$ as the primary route | The original $1\to0.5$ route lies on the divergent-variance boundary | The primary low-temperature estimate has finite weight variance and high empirical overlap |
| Validation | Compared every direct result with $\langle x^2\rangle=1/(2\beta)$ | The model is exactly solvable | Numerical bias and uncertainty can be checked quantitatively |
| Packaging | Added `.gitignore`, `requirements.txt`, an importable filename, CLI flags, and a detailed README | The coursework file was not repository-ready | The project can be cloned, reproduced, reviewed, and maintained |

## Detailed numerical improvements

### Exact analytical reference

For

$$
H(x)=x^2,
$$

the normalized density is

$$
p_\beta(x)
=
\sqrt{\frac{\beta}{\pi}}e^{-\beta x^2}.
$$

The exact mean energy is

$$
\langle H\rangle_\beta
=
\langle x^2\rangle_\beta
=
\frac{1}{2\beta}.
$$

The rebuilt program evaluates this formula for every simulated temperature and reports the relative numerical error. This was added to distinguish sampling accuracy from visual agreement in a histogram.

### Reproducible independent chains

The original script used `np.random.rand()` directly. Results changed with process history and only one trajectory was available at each temperature.

The rebuild uses a fixed master seed and spawns independent generators. Four chains are run for each value of $\beta$. This improves reproducibility and provides better protection against a misleading single-chain result.

### Temperature-scaled proposals

The Metropolis proposal remains

$$
x'=x+\delta,
\qquad
\delta\sim\mathcal U(-e,e),
$$

with acceptance probability

$$
P_{\mathrm{accept}}
=
\min\left(1,e^{-\beta[x'^2-x^2]}\right).
$$

The rebuilt proposal width is

$$
e(\beta)=\frac{2}{\sqrt{\beta}}.
$$

This follows the scale of the Gaussian target. The impact is a consistent acceptance rate of approximately $0.514$ across the five default temperatures rather than requiring a manually chosen width for every run.

### Correlation-aware uncertainty

Successive Markov states are correlated. The rebuild divides retained observables into blocks and estimates

$$
\sigma_{\bar O}
=
\frac{s_{\mathrm{block}}}{\sqrt{B}},
$$

where $B$ is the number of blocks and $s_{\mathrm{block}}$ is the sample standard deviation of the block means.

The energy autocorrelation is evaluated for the requested lags:

$$
C(k)
=
\frac{
\sum_{i=1}^{N-k}(H_i-\bar H)(H_{i+k}-\bar H)
}{
N\operatorname{Var}(H)
}.
$$

The block size is then interpreted relative to the measured energy-correlation scale.

### Stable reweighting and jackknife errors

For samples drawn at $\beta_0$, the target expectation is

$$
\langle O\rangle_{\beta_1}
=
\frac{\langle O(x)w(x)\rangle_{\beta_0}}
{\langle w(x)\rangle_{\beta_0}},
$$

where

$$
w(x)=e^{-(\beta_1-\beta_0)x^2}.
$$

The rebuild shifts the log weights before exponentiation. Multiplying all weights by a common factor leaves the ratio unchanged but reduces overflow and underflow risk.

Because this is a ratio estimator, the uncertainty is calculated with a blocked jackknife rather than by treating the numerator and denominator as independent.

### Effective sample size

Weight concentration is summarized using

$$
N_{\mathrm{eff}}
=
\frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}.
$$

The fraction $N_{\mathrm{eff}}/N$ is reported for every route. This provides a finite-sample overlap diagnostic that the original script did not have.

### Analytical finite-variance criterion

Under the source density $p_{\beta_0}$,

$$
\mathbb E_{\beta_0}[w^2]
\propto
\int_{-\infty}^{\infty}
e^{-(2\beta_1-\beta_0)x^2}\,dx.
$$

The second moment is finite only if

$$
2\beta_1-\beta_0>0.
$$

For the original direct route $1\to0.5$,

$$
2(0.5)-1=0,
$$

so the theoretical weight variance diverges. The point estimate can still appear accurate for one finite sample, but its stability is not guaranteed.

The rebuilt primary route uses $0.75\to0.5$:

$$
2(0.5)-0.75=0.25>0.
$$

This route has finite weight variance and strong empirical overlap.

## Measured impact

The verified default run produced direct estimates within $0.46\%$ of the exact result at all five temperatures. Acceptance rates were between `0.513` and `0.515`.

The primary bridge result was

$$
\langle x^2\rangle_{\beta=0.5}
=
1.000977\pm0.009543,
$$

compared with the exact value $1$. Its empirical weight effective fraction was approximately $86.69\%$, and the analytical variance criterion was satisfied.

The direct $1\to0.5$ route was retained only as an unstable diagnostic. This changes the scientific interpretation: the repository no longer promotes a numerically convenient but theoretically fragile route as a primary result.

## Repository and documentation impact

The archive adds:

- an importable `gaussian_metropolis.py` module;
- a non-interactive `--no-show` mode;
- declared NumPy and Matplotlib dependencies;
- a `.gitignore` suitable for Python numerical work;
- a detailed README containing derivations, settings, verified results, diagnostics, and limitations.

These changes make the work suitable for GitHub review while keeping the original coursework source available beside this explanation.

## Remaining limitations

- The Gaussian model is exactly solvable and one-dimensional.
- Blocking reduces correlation bias but does not create independent samples.
- The empirical weight ESS does not include Markov autocorrelation.
- More distant reweighting targets would require additional bridge ensembles or a multihistogram method.
- No open-source licence was applied to the produced archive.
