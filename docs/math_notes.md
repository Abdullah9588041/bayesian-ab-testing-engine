# Mathematical Notes: Bayesian A/B Testing

These notes derive every formula implemented in `src/bayes_ab/`.
Notation: variant A (control) and B (test) have unknown conversion rates
$p_A, p_B \in [0,1]$. Observed data: $n_A, n_B$ trials (impressions),
$s_A, s_B$ successes (purchases).

---

## 1. Beta–Binomial conjugacy (posterior derivation)

**Prior.** $p \sim \mathrm{Beta}(\alpha_0, \beta_0)$ with density
$$f(p) = \frac{p^{\alpha_0-1}(1-p)^{\beta_0-1}}{B(\alpha_0,\beta_0)},
\qquad 0 \le p \le 1.$$

**Likelihood.** $s \mid p \sim \mathrm{Binomial}(n, p)$:
$$L(p \mid s) = \binom{n}{s} p^{s}(1-p)^{n-s} \propto p^{s}(1-p)^{n-s}.$$

**Posterior** (Bayes' theorem, dropping $p$-free constants):
$$\begin{aligned}
f(p \mid s) &\propto p^{s}(1-p)^{n-s}\cdot p^{\alpha_0-1}(1-p)^{\beta_0-1} \\
&= p^{(\alpha_0+s)-1}(1-p)^{(\beta_0+n-s)-1},
\end{aligned}$$
which is the kernel of $\mathrm{Beta}(\alpha_0+s,\ \beta_0+n-s)$.
Hence, with $\alpha = \alpha_0 + s$, $\beta = \beta_0 + n - s$:

$$p \mid s \sim \mathrm{Beta}(\alpha, \beta), \qquad
  \mathbb{E}[p \mid s] = \frac{\alpha}{\alpha+\beta}.$$

The posterior mean is a precision-weighted compromise between the prior
mean $\alpha_0/(\alpha_0+\beta_0)$ and the MLE $s/n$.

---

## 2. Posterior probability that B beats A

The posteriors of $p_A, p_B$ are independent, so with densities
$f_A, f_B$ and $F_B$ the CDF of $p_B$:

$$P(p_B > p_A \mid \text{data})
= \int_0^1 f_A(x)\,[1 - F_B(x)]\,dx.$$

This is evaluated by adaptive quadrature (`scipy.integrate.quad`) and
cross-validated against Monte Carlo in the test suite. For identical
posteriors the integral is exactly $1/2$ by symmetry.

---

## 3. Expected-loss decision rule

Choosing variant $X \in \{A, B\}$ incurs the *opportunity loss*
$\max(p_{\text{other}} - p_X,\, 0)$ — the conversion rate we forfeit if the
other variant was actually better. The Bayes decision minimizes posterior
expected loss:

$$\text{Choose B} \iff \mathbb{E}[\max(p_A - p_B, 0)] < \mathbb{E}[\max(p_B - p_A, 0)].$$

The key identity used for computation: for $X \sim \mathrm{Beta}(\alpha,\beta)$,

$$\mathbb{E}[(X - c)_+]
= \frac{\alpha}{\alpha+\beta}\,[1 - F_{\alpha+1,\beta}(c)]
- c\,[1 - F_{\alpha,\beta}(c)],$$

where $F_{a,b}$ is the $\mathrm{Beta}(a,b)$ CDF. *Proof sketch:*
$\int_c^1 x f_{\alpha,\beta}(x)\,dx
= \frac{B(\alpha+1,\beta)}{B(\alpha,\beta)}[1 - F_{\alpha+1,\beta}(c)]$
and $B(\alpha+1,\beta)/B(\alpha,\beta) = \alpha/(\alpha+\beta)$.
The outer expectation over the chosen variant's rate is then a single
one-dimensional quadrature:

$$\mathbb{E}[\max(p_A - p_B, 0)]
= \int_0^1 \mathbb{E}[(p_A - c)_+]\, f_B(c)\,dc.$$

**Stopping rule.** Stop the experiment when either
$P(p_B > p_A \mid \text{data}) \ge 0.95$ (or $\le 0.05$), or the expected
loss of the current leader falls below a business-negligible threshold
(e.g. $5\times10^{-4}$ = 0.05 percentage points of conversion rate).

---

## 4. Why sequential Bayesian monitoring avoids the peeking problem

A frequentist p-value is defined *relative to a sampling plan*: the
probability, under $H_0$, of data at least as extreme *in hypothetical
repetitions of the whole procedure, including the stopping rule*.
Peeking at the data and stopping when $p < 0.05$ changes that reference
set, so the nominal $\alpha$ no longer holds — the simulation in
`sequential.simulate_peeking_bias` shows the false-positive rate roughly
tripling with 5 peeks.

A posterior probability $P(p_B > p_A \mid \text{data observed})$ conditions
only on the data actually seen; the likelihood principle implies it does
not depend on the experimenter's *intentions* about when they would have
stopped. Repeatedly recomputing it as data arrive is therefore coherent —
no alpha-spending correction is required.

**Honest caveat (important).** This does *not* mean a Bayesian sequential
rule controls the frequentist Type I error rate at 5%. The threshold
$P(B>A) > 0.95$ is a *posterior* statement ("given our prior and these
data, B wins with 95% probability"), not a long-run error guarantee.
Under $H_0$ the observed decision rate is typically *below* the nominal
level (the simulation reports it rather than assuming it), but it is not
a theorem. If a stakeholder demands strict frequentist error control,
use a group-sequential design (e.g. O'Brien–Fleming boundaries) instead.

---

## 5. Prior sensitivity

The posterior parameters $\alpha = \alpha_0 + s$ show the prior acts as
$\alpha_0$ pseudo-successes and $\beta_0$ pseudo-failures. With
$n \sim 10^6$ observations, any reasonable prior is swamped
($\alpha_0 \ll s$); with small $n$ the prior dominates. The
`priors` module recomputes every decision quantity under:

- **Uniform** $\mathrm{Beta}(1,1)$ — maximum-entropy, adds 1 pseudo-count
  of each type;
- **Jeffreys** $\mathrm{Beta}(1/2,1/2)$ — the objective prior for the
  Binomial model, invariant under reparameterization;
- **Historical** $\mathrm{Beta}(5,995)$ — encodes past campaigns at
  $\approx 0.5\%$ conversion with the strength of 1000 observations.

---

## 6. Sample-size planning

**Frequentist (power).** Per-variant $n$ for a two-sided $z$-test of
$H_0: p_A = p_B$ at level $\alpha$ and power $1-\gamma$ (normal
approximation):

$$n = \frac{\left(z_{1-\alpha/2}\sqrt{2\bar p\bar q}
+ z_{1-\gamma}\sqrt{p_A q_A + p_B q_B}\right)^2}{(p_B - p_A)^2},
\qquad \bar p = \tfrac{p_A+p_B}{2}.$$

**Bayesian (assurance).** Average over the sampling distribution:
simulate $M$ experiments of size $n$ from assumed true rates, apply the
*actual* decision rule ($P(B>A) \ge 0.95$), and report the fraction that
decide correctly. Assurance answers "what is the probability this
experiment, run with our stopping rule, will give us an answer?" —
power answers a narrower question about a test statistic crossing a
threshold under fixed $n$.

## References

- Gelman, A. et al. *Bayesian Data Analysis*, 3rd ed., Ch. 2–3 (conjugacy).
- Berger, J.O. *Statistical Decision Theory and Bayesian Analysis* (expected loss).
- Kruschke, J.K. *Doing Bayesian Data Analysis*, Ch. 11 (Bayesian A/B decisions).
- Efron, B. "Bayesians, Frequentists, and Scientists" (peeking / stopping-rule principle).
