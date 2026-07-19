---
title: "Dynamic Discrete Choice: Example of Bus Engine Replacement Problem"
date: 2026-07-18
description: "Introduce dynamic discrete choice models using the classic example of a bus engine replacement problem."
categories:
  - Structural Modeling
  - Discrete Choice
  - Dynamic Logit
---

In the real world, choices are sequential, forward-looking, and dynamic. Every choice decision you make today alters the choice set available to you tomorrow. To study the sequence of an individual's choices, economists turn to the **Dynamic Discrete Choice** framework.

The classic entry point to study this framework is Rust (1987)'s **Optimal Replacement of GMC Bus Engines**.

## Bus Engine Dilemma
Harold Zurcher is a maintenance superintendent of the Madison Metropolitan Bus Company. Every month, Zurcher inspects the fleet of buses and faces a binary choice for each one: **replace the engine now, or keep it running for another month?**

- **Replace the engine:** It is expensive, incurring an upfront fixed cost ($RC$), but it resets the bus to like-new condition.
- **Keep it running:** It is cheap today, but maintainance costs rise with mileage, and there is always a chance of engine failure.

Zurcher is **forward-looking**. He wants to not only maintain a low cost today but also to minimize the expected *discounted stream* of all future costs. This tension between immediate expenses and future risks is the heart of dynamic optimization.

![](bus-eng.png){width="90%" fig-align="center"}

## Model Setup
### State and Control Variables

- **State variable $x_t$**: Cumulative mileage since last replacement (discretized into bins)
  - [Conditional Independence Assumption] Future cost only depends on the cumulative mileage ofthe current engine
- **Control variable $a_t \in \{0, 1\}$**: Binary decision at each month
  - $a_t = 0$: keep the current engine
  - $a_t = 1$: replace the engine

### Utility Function

$$
u(x_t, a_t, \theta) = \begin{cases}
-c(x_t, \theta_1) & \text{if } a_t = 0 \text{ (keep)} \\
-[RC + c(0, \theta_1)] & \text{if } a_t = 1 \text{ (replace)}
\end{cases}
$$

- $c(x, \theta_1)$: Deterministic maintenance cost, increasing in mileage. Rust uses a simple linear form $c(x, \theta_1) = \theta_1 x$. Thus, when $a=1$, $c(0, \theta_1) = 0$ and the utility is $-RC$.

Moreover, Zurcher observes an additive **private** shock $\varepsilon_t(a)$ for each choice. The total per-period payoff is hence given by $u(x_t, a_t, \theta) + \varepsilon_t(a_t)$.

- $\varepsilon_t(a)$: The private shock for the action $a$ at time $t$ is assumed to be an i.i.d. **Gumbel** (Type I extreme value) error, which the econometrician doesn't observe.

### State Transition
The transition process captures how buses accumulate mileage. It's exogenous to the replacement decision.

- If $a_t = 0$ (keep): mileage increases stochastically.
$$x_{t+1} \sim F(\cdot | x_t, \theta_2)$$
- If $a_t = 1$ (replace): mileage resets to 0, then increases.
$$x_{t+1} \sim F(\cdot | 0, \theta_2)$$

Rust discretizes mileage into bins and estimates the transition matrix $\theta_2$ directly from the data. This is a key simplification, **separating** $\theta_2$ from $\theta_1$.

### Discount Factor

$\beta \in (0, 1)$: how much Zurcher weights future costs relative to today. Typically fixed (e.g., $\beta = 0.966$) rather than estimated, because it's weakly identified in practice.

> **Example**: Anthropic subscription plan at \$200 if billed up front for a year or \$20 if billed monthly:
> $$
> 20\times (1 + \beta +\beta^2 + ... + \beta^{11})=20\times \frac{1-\beta^{12}}{1-\beta} = 200 \implies \beta \approx 0.966
> $$

### Value Function

The **value function** $V(x_t, \varepsilon_t)$ captures the optimized total expected cost from state $(x_t, \varepsilon_t)$ onward:

$$
V(x_t, \varepsilon_t) = \max_{{\{a_t\}}^\infty_{t=0}} \mathbb{E} \left[ \sum_{s=0}^{\infty} \beta^s \Big( u(x_{t+s}, a_{t+s}, \theta) + \varepsilon_{t+s}(a_{t+s}) \Big) \,\Big|\, x_t, \varepsilon_t \right]
$$

## Bellman Equation

The optimization problem with infinite-horizon can be rewritten as the following recursive definition in terms of value function, known as the **Bellman equation**:

$$
V(x_t, \varepsilon_t) = \max_{a_t \in \{0,1\}} \left\{ u(x_t, a_t, \theta) + \varepsilon_t(a_t) + \beta \mathbb{E}[V(x_{t+1}, \varepsilon_{t+1}) | x_t, a_t] \right\}
$$

**Conditional Independence (CI) Assumption** implies (1) $\varepsilon_{t+1}$ is an i.i.d. error in each period, and (2) $x_{t+1}$ depends only on $(x_t, a_t)$, not on $\varepsilon_t$. That is,
$$
p(x_{t+1}, \varepsilon_{t+1} | x_t, \varepsilon_t, a_t) = p(\varepsilon_{t+1}) \cdot p(x_{t+1} | x_t, a_t).
$$

This assumption allows us to integrate out $\varepsilon$ and work with a lower-dimensional object.

### Integrated Value Function
To compute the expected future value, we average over next period's mileage (via the transition matrix) and next period's shocks (via the Gumbel distribution).

Define the **expected value function** (integrated over both $\varepsilon'$ and $x'$):

$$
EV(x, a) = \sum_{x'} \mathbb{E}_{\varepsilon'} \left[ V(x', \varepsilon') \right] \cdot F(x' | x, a)
$$

- Since replacement always resets mileage to 0, $EV(x, 1) = EV(0, 0)$ for all $x$.

Let $\bar{u}(x, a) \equiv u(x, a, \theta)$ denote the deterministic part. Under the **Gumbel** assumption on $\varepsilon_t(a)$, the expectation over $\varepsilon$ has a closed-form (the log-sum formula):

$$
\mathbb{E}_\varepsilon \left[ \max_a \{ \bar{u}(x, a) + \varepsilon(a) + \beta \, EV(x, a) \} \right] = \log \left( \sum_a \exp\left( \bar{u}(x, a) + \beta \, EV(x, a) \right) \right) + \gamma_E
$$

where $\gamma_E$ is Euler's constant (~0.5772). This gives us a **fixed-point equation in $EV$**:

$$
EV(x, a) = \sum_{x'} \left[ \log \left( \sum_{a'} \exp(\bar{u}(x', a') + \beta \, EV(x', a')) \right) + \gamma_E \right] \cdot F(x' | x, a)
$$

**Key:** The Gumbel assumption not only simplifies the computation of $\mathbb{E}_{\varepsilon}$ but also **connects this dynamic problem to the choice probabilities** we observe in the data. Particularly, the log-sum formula is the bridge between the value function and the choice probabilities.

## Choice Probabilities

Given the Gumbel shocks, the probability of replacing the engine at state $x_t$ is:

$$
P(a_t = 1 | x_t, \theta) = \frac{\exp(\bar{u}(x_t, 1) + \beta \, EV(x_t, 1))}{\exp(\bar{u}(x_t, 0) + \beta \, EV(x_t, 0)) + \exp(\bar{u}(x_t, 1) + \beta \, EV(x_t, 1))}
$$

This is a **logit formula**, similar to a static discrete choice model, except the inclusion of the continuation value $\beta \, EV(x, a)$.

**Key:** The dynamic problem reduces to a sequence of static logit choices, where the future is summarized by $EV$. The hard part is the estimation of $EV$.

## Estimation - Nested Fixed Point (NFXP)


### Outer Loop: Maximum Likelihood

The likelihood of the observed data $\{x_t^i, a_t^i\}$ for bus $i$ over periods $t$:

$$
\mathcal{L}(\theta) = \prod_i \prod_t P(a_t^i | x_t^i, \theta) \cdot F(x_{t+1}^i | x_t^i, a_t^i, \theta_2)
$$

The optimizer proposes $\theta = (\theta_1, RC)$, evaluates the likelihood, and iterates.

### Inner Loop: Solve the Fixed Point

For each candidate $\theta$, solve the Bellman equation to get $EV(\cdot; \theta)$:

1. Start with an initial guess $EV^{(0)}(x) = 0$ for all $x$.
2. Apply the contraction mapping:
$$
EV^{(k+1)}(x) = T(EV^{(k)})(x) = \sum_{x'} \left[ \log \sum_{a'} \exp(\bar{u}(x', a') + \beta \, \widetilde{EV}^{(k)}(x', a')) + \gamma_E \right] F(x' | x, 0)
$$
3. Repeat until $\| EV^{(k+1)} - EV^{(k)} \| < \text{tol}$.
4. Compute choice probabilities from the converged $EV$.

**Why we only solve for $EV(x, 0)$:** As noted above, $EV(x, 1) = EV(0, 0)$ for all $x$. The future expected value, $EV(x, 0)$ (if "keep") and $EV(0, 0)$ (if "replace"), can be denoted as $EV(x)$. So we only iterate on $EV(x)$ across the $K$ mileage states, that is,

$$
{EV}^{(k)}(x, a) = \begin{cases}
EV^{(k)}(x) & \text{if } a = 0 \\
EV^{(k)}(0) & \text{if } a = 1
\end{cases}.
$$

**Why "nested":** Every time the outer loop tries a new $\theta$, the inner loop must re-solve the entire dynamic programming problem from scratch. This is computationally expensive, as the inner loop runs to convergence for *each* likelihood evaluation.