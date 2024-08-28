+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Tools

> #### Neumann series
>
> Let $A \in \Cnn$ with spectral radius $\rho(A) < 1$ be given.
> Then $I - A$ is invertible and
> $${(I - A)}^{-1} = \sum_{k = 0}^\infty A^k.$$

Will be important to identify the actual shape of Markov parameters and system
moments.

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

> #### (Polynomial) Krylov subspace
>
> Given an invertible matrix $A \in \Rnn$ and a vector $b \in \Rn$ the
> ***$k$-dimensional (polynomial) Krylov subspace*** is defined as
> $$\cK_{k}(A, b)
  := \myspan\!\left(b, A b, A^{2} b, \ldots, A^{k - 1} b\right).$$

> #### Rational Krylov subspace
>
> Given an invertible matrix $A \in \Rnn$ a vector $b \in \Rn$ and a vector of
> shifts $s \in \Rk$ the ***$k$-dimensional rational Krylov subspace*** is
> defined as
> $$\cK_{k}(A, b, s)
  := \myspan\!\left({(s_1 I - A)}^{-1} b, {(s_2 I - A)}^{-1} b,
     \ldots, {(s_k I - A)}^{-1} b\right).$$

Orthonormal bases of these spaces should be computed via the
[*Arnoldi iteration*](https://en.wikipedia.org/wiki/Arnoldi_iteration).

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Padé-type approximations

> #### Goal
>
> Match the coefficients $M_k(s_0)$ or $M_k(\infty)$ in
> $$
  H(s) = \sum_{k = 0}^{\infty} {(s - s_0)}^k M_k(s_0), \qquad
  H(s) = \sum_{k = 0}^{\infty} s^{-k} M_k(\infty).
  $$

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

> #### Motivation (assume: $m = p = 1$,  $s$ large enough)
>
> $$
  \begin{align*}
    H(s)
    & = C {(s E - A)}^{-1} B
    = \tfrac{1}{s} C
    \underbrace{{\left( I - \tfrac{1}{s} E^{-1} A \right)}^{-1}}%
    _{= \sum_{k = 0}^\infty \frac{1}{s^k}{(E^{-1} A)}^k} E^{-1} B
    = \sum_{k = 1}^\infty C {\left( E^{-1} A \right)}^{k - 1} E^{-1} B
      \tfrac{1}{s^{k}}.
  \end{align*}
  $$

+++ {"editable": true, "slideshow": {"slide_type": "fragment"}}

> Therefore, we have
>
> $$
  M_k(\infty) =
  \begin{cases}
    0, & \text{if } k = 0, \\
    C {\left( E^{-1} A \right)}^{k - 1} E^{-1} B, & \text{if } k \ge 1.
  \end{cases}
  \qquad \leadsto \text{ use } V = \cK_{r}(E^{-1} A, E^{-1} B)
  $$

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}

### Padé-type approximations

> #### Approximation at $\infty$
>
> $$
  V = \cK_{r}\!\left(E^{-1} A, E^{-1} B\right), \qquad
  W = V \text{ or }
  W = \cK_{r}\!\left(A^{\tran} E^{-\tran}, C^{\tran}\right)
  $$

> #### Approximation at $s_0 = 0$
>
> $$
  V = \cK_{r}\!\left(A^{-1} E, A^{-1} B\right), \qquad
  W = V \text{ or }
  W = \cK_{r}\!\left(E^{\tran} A^{-\tran}, C^{\tran}\right)
  $$

> #### Approximation at $s_0 \in \bbC$
>
> $$
  V = \cK_{r}\!\left({(s_0 E - A)}^{-1} E, {(s_0 E - A)}^{-1} B\right), \qquad
  W = V
  $$
>
> or
>
> $$
  W = \cK_{r}\!\left(E^{\tran} {\left(s_0 E^{\tran}- A^{\tran}\right)}^{-1},
    C^{\tran}\right)
  $$

+++ {"editable": true, "slideshow": {"slide_type": "subslide"}}
