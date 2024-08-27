---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"slideshow": {"slide_type": "slide"}}

# System-theoretic Methods

<h2>
August 27, 2024<br/>
pyMOR School and User Meeting 2024
</h2>

+++ {"slideshow": {"slide_type": "subslide"}}

# What this actually is all about:

<center>
<img src="/files/figures/system_fom.svg" alt="system" width="60%"/>
</center>

+++ {"slideshow": {"slide_type": "subslide"}}

<center>
<img src="/files/figures/mor_system_fo_v2.svg" alt="mor" width="40%">
</center>

+++ {"slideshow": {"slide_type": "subslide"}}

# Outline

<h2>
1. Linear Time-Invariant (LTI) Systems<br/>
2. Transfer Function and Realizations<br/>
3. Projection-based Model order Reduction<br />
4. System Analysis<br/>
5. A Selection of MOR Methods<br/>
</h2>

+++ {"slideshow": {"slide_type": "subslide"}}

# Restrictions for this lecture

- Only continuous-time systems
  - Discrete-time is treated in
    [[Antoulas '05]](https://doi.org/10.1137/1.9780898718713)
- Only unstructured systems
- No differential-algebraic systems
  - For DAE aspects see
    [[Voigt '19]](https://www.math.uni-hamburg.de/home/voigt/Modellreduktion_SoSe19/Notes_ModelReduction.pdf),
    [[Gugercin/Stykel/Wyatt '13]](https://doi.org/10.1137/130906635),
    [[Mehrmann/Stykel '05]](https://doi.org/10.1007/3-540-27909-1_3),
    [[Stykel '04]](https://doi.org/10.1007/s00498-004-0141-4)
- No non-linearities
- No parameter dependencies

+++ {"slideshow": {"slide_type": "slide"}}

# Linear Time-Invariant (LTI) Systems

## Setting for this course

### First-order State-space Systems (pyMOR: [`LTIModel`](https://docs.pymor.org/2023-1-0/autoapi/pymor/models/iosys/index.html#pymor.models.iosys.LTIModel))

$$
\begin{equation}\tag{$\Sigma$}
  \begin{aligned}
    E \dot{x}(t) & = A x(t) + B u(t), \\
    y(t) & = C x(t) + D u(t).
  \end{aligned}
\end{equation}
$$

Here

- $x(t) \in \mathbb{R}^{n}$ is called the *state*,
- $u(t) \in \mathbb{R}^{m}$ is called the *input*,
- $y(t) \in \mathbb{R}^{p}$ is called the *output*

of the LTI system.
Correspondingly, we have

$$
\begin{align*}
  E, A \in \mathbb{R}^{n \times n}, \qquad
  B \in \mathbb{R}^{n \times m}, \qquad
  C \in \mathbb{R}^{p \times n}, \quad\text{and}\quad
  D \in \mathbb{R}^{p \times m}.
\end{align*}
$$

We assume
$t \in [0, \infty)$,
$x(0) = 0$,
$E$ is invertible,
$E^{-1} A$ is Hurwitz, and
$D = 0$.

+++ {"slideshow": {"slide_type": "subslide"}}

## Examples

### Heat Equation ([MORWiki thermal block](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Thermal_Block))

<table>
<tr>
<td>

For $t \in (0, T)$, $\xi \in \Omega$ and initial values

$$
\theta(0, \xi) = 0,\text{ for } \xi \in \Omega,
$$

consider

$$
\begin{align*}
  \partial_t \theta(t, \xi)
  + \nabla \cdot (-\sigma(\xi) \nabla \theta(t, \xi))
  & = 0,
\end{align*}
$$

with boundary conditions

$$
\begin{align*}
  \sigma(\xi) \nabla \theta(t, \xi) \cdot n(\xi) & = u(t)
  & t \in (0, T),
  & \ \xi \in \Gamma_{\text{in}}, \\
  \sigma(\xi) \nabla \theta(t, \xi) \cdot n(\xi) & = 0
  & t \in (0, T),
  & \ \xi \in \Gamma_{\text{N}}, \\
  \theta(t, \xi) & = 0
  & t \in (0, T),
  & \ \xi \in \Gamma_{\text{D}},
\end{align*}
$$

and outputs

$$
y_i(t) = \int_{\Omega_i} \theta(t, \xi) \operatorname{d}\!{\xi}, \quad
i = 1, 2, 3, 4.
$$

</td>
<td>
<center>
<img src="/files/figures/cookie.svg" alt="cookie domain" width="30%">
<img src="/files/figures/Euler_100_Tf.png" alt="cookie snapshot" width="30%">
</center>
</td>
</tr>
</table>

+++ {"slideshow": {"slide_type": "subslide"}}

#### Finite element semi-discretization in space

- pairwise inner products of ansatz functions $\leadsto E$
- discretized spatial operator + Dirichlet boundary condition $\leadsto A$
- discretized non-zero Neumann boundary condition $\leadsto B$
- average temperatures on the inclusions $\leadsto C$

---

- $n = 7\,488$
- $m = 1$
- $p = 4$

+++ {"slideshow": {"slide_type": "subslide"}}

### Penzl Example ([MORWiki](https://morwiki.mpi-magdeburg.mpg.de/morwiki/index.php/Penzl%27s_FOM))

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---
import numpy as np
import scipy.sparse as sps
from pymor.models.iosys import LTIModel

A1 = np.array([[-1, 100], [-100, -1]])
A2 = np.array([[-1, 200], [-200, -1]])
A3 = np.array([[-1, 400], [-400, -1]])
A4 = sps.diags(np.arange(-1, -1001, -1))
A = sps.block_diag((A1, A2, A3, A4), format='csc')
B = np.ones((1006, 1))
B[:6] = 10
C = B.T

fom = LTIModel.from_matrices(A, B, C)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
fom
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
print(fom)
```

+++ {"slideshow": {"slide_type": "subslide"}}

We can perform time-domain simulation, but the final time and the time stepper
need to be specified in the `Model`.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
from pymor.algorithms.timestepping import ImplicitEulerTimeStepper

T = 2
nt = 10000
fom = fom.with_(T=T, time_stepper=ImplicitEulerTimeStepper(nt))
```

+++ {"slideshow": {"slide_type": "subslide"}}

We first simulate the impulse response, i.e.,
the output in response to $x(0) = 0$ and $u(t) = \delta(t)$.

$$
\begin{align*}
  y(t)
  & =
    C e^{t E^{-1} A} x(0)
    + \int_0^T C e^{\tau E^{-1} A} E^{-1} B u(t - \tau) \operatorname{d\!}\tau
    + D u(t) \\
  & =
    C e^{t E^{-1} A} E^{-1} B
    + D \delta(t)
\end{align*}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_impulse = fom.impulse_resp()
print(y_impulse.shape)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
import matplotlib.pyplot as plt

_ = plt.plot(np.linspace(0, T, nt + 1), y_impulse[:, 0, 0])
```

+++ {"slideshow": {"slide_type": "subslide"}}

Next we simulate the response to a sinusoidal input.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
y_sin_100 = fom.output(input='sin(100 * t)')
print(y_sin_100.shape)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = plt.plot(np.linspace(0, T, nt + 1), y_sin_100[:, 0])
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
y_sin_50 = fom.output(input='sin(50 * t)')
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = plt.plot(np.linspace(0, T, nt + 1), y_sin_50[:, 0])
```

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise

- Use the time stepper specified above and simulate the model using the input
  function $u(t) = e^{-t}$.
- Change the number of timesteps `nt` to `50000`. Repeat the simulation of the
  model using $u(t) = e^{-t}$.

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---

```

+++ {"slideshow": {"slide_type": "slide"}}

# Transfer Function

+++ {"slideshow": {"slide_type": "subslide"}}

## Laplace Transform

> ### Definition
>
> Let $f \colon [0, \infty) \to \mathbb{R}^{n}$ be exponentially bounded with
> bounding exponent $\alpha$.
> Then
> $$\mathcal{L}\{f\}(s) := \int_0^\infty f(\tau) e^{-s \tau} \operatorname{d}\!{\tau}$$
> for $\operatorname{Re}(s) > \alpha$ is called the ***Laplace transform*** of $f$.
> The process of forming the Laplace transform is called
> ***Laplace transformation***.

It can be shown that the integral converges uniformly in a domain with
$\operatorname{Re}(s) \ge \beta$ for all $\beta > \alpha$.

+++ {"slideshow": {"slide_type": "fragment"}}

> Allows us to map time signals to frequency signals.

+++ {"slideshow": {"slide_type": "subslide"}}

> ### Theorem
>
> Let $f, g, h \colon [0, \infty) \to \mathbb{R}^n$ be given.
> Then the following two statements hold true:
>
> 1. The Laplace transformation is linear, i.e.,
>    if $f$ and $g$ are exponentially bounded,
>    then $h := \gamma f + \delta g$ is also exponentially bounded and
>
>    $$
     \mathcal{L}\left\{h\right\} = \gamma\mathcal{L}\left\{f\right\} +
     \delta\mathcal{L}\left\{g\right\}
     $$
>
>    holds for all $\gamma, \delta \in \mathbb{C}$.
> 2. If $f \in \mathcal{PC}^1([0, \infty), \mathbb{R}^{n})$ and $\dot{f}$ is
>    exponentially bounded, then $f$ is exponentially bounded and
>
>    $$
     \mathcal{L}\bigl\{\dot{f}\bigr\}(s) = s \mathcal{L}\{f\}(s) - f(0).
     $$

+++ {"slideshow": {"slide_type": "fragment"}}

- $X(s) := \mathcal{L}\{x\}(s)$,
  $U(s) := \mathcal{L}\{u\}(s)$, and
  $Y(s) := \mathcal{L}\{y\}(s)$
- $A x(t) + B u(t) \leadsto A X(s) + B U(s)$
- $y(t) = C x(t) \leadsto Y(s) = C X(s)$
- $s X(s) := \mathcal{L}\{\dot{x}\}(s)$ (since $x(0) = 0$)

+++ {"slideshow": {"slide_type": "subslide"}}

## Transfer Function

In summary we have:

- $s E X(s) = A X(s) + B U(s)$
- $Y(s) = C X(s)$

Thus the mapping from inputs to outputs in frequency domain can be expressed as

$$
H(s) = C {\left(s E - A\right)}^{-1} B.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$$
H \text{ is analytic in } \mathbb{C} \setminus \Lambda(E, A).
$$

+++ {"slideshow": {"slide_type": "fragment"}}

### Pole-residue Form

Let $(\lambda_{i}, w_{i}, v_{i})$ be the eigentriplets of the pair $(E, A)$
with no degenerate eigenspaces.
Then the ***poles*** of $H$ are given by the eigenvalues
$\lambda_1,\ldots,\lambda_n$ and we have

$$
H(s) = \sum_{i = 1}^{n} \frac{R_{i}}{s - \lambda_{i}},
$$

where $R_{i} = (C v_{i}) (w_{i}^{\operatorname{H}} B)$,
assuming $w_{i}^{\operatorname{H}} E v_{i} = 1$.

+++ {"slideshow": {"slide_type": "subslide"}}

### Example

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fom.transfer_function
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fom.transfer_function.eval_tf(0)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
fom.transfer_function.eval_tf(10j)
```

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise

- Use the [`poles`](https://docs.pymor.org/2023-1-0/autoapi/pymor/models/iosys/index.html#pymor.models.iosys.LTIModel.poles) method of the [`LTIModel`](https://docs.pymor.org/2023-1-0/autoapi/pymor/models/iosys/index.html#pymor.models.iosys.LTIModel) class to compute the poles of the transfer function of `fom`.
  Compute the imaginary parts of the poles.
- Evaluate the transfer function for several values on the imaginary axis.
  Select some values that correspond to the imaginary parts of the poles and
  others which are close by.

```{code-cell} ipython3
---
slideshow:
  slide_type: ''
---

```

+++ {"slideshow": {"slide_type": "subslide"}}

### Frequency-Domain Analysis

#### Bode Plots

The Bode plot for $H$ consists of a ***magnitude plot*** and a ***phase plot***.

> ##### Bode magnitude plot
>
> - component-wise graph of the function $\lvert H(\boldsymbol{\imath} \omega) \rvert$
>   for frequencies $\omega \in [\omega_{\min}, \omega_{\max}] \subset \mathbb{R}$.
> - $\omega$-axis is logarithmic.
> - magnitude is given in decibels, i.e., $\lvert H(\boldsymbol{\imath} \cdot) \rvert$ is
>   plotted as $20 \log_{10}(\lvert H(\boldsymbol{\imath} \cdot) \rvert)$.

> ##### Bode phase plot
>
> - component-wise graph of the function $\arg{H(\boldsymbol{\imath} \omega)}$
>   for frequencies $\omega \in [\omega_{\min}, \omega_{\max}] \subset \mathbb{R}$.
> - $\omega$-axis is logarithmic.
> - phase is given in degrees on a linear scale.

+++ {"slideshow": {"slide_type": "subslide"}}

#### Bode Plot for the Thermal Block Example

<center>
<img src="/files/figures/cookie_bode.svg" alt="cookie bode" width="60%">
</center>

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
# w = (1e-1, 1e5)
w, _ = fom.transfer_function.freq_resp((1e-1, 1e5))
```

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = fom.transfer_function.bode_plot(w)
```

+++ {"slideshow": {"slide_type": "subslide"}}

> #### (Sigma) Magnitude Plots
>
> - 2-norm-wise graph of the function $H(\boldsymbol{\imath} \omega)$
>   for frequencies $\omega \in [\omega_{\min}, \omega_{\max}] \subset \mathbb{R}$.
> - $\omega$-axis is logarithmic.

The name is due to the fact that for a given matrix $M$ the norm
$\lVert M \rVert_2$ is given by its largest singular value.

The real sigma magnitude plot depicts all singular values as functions of
$\omega$.

```{code-cell} ipython3
---
slideshow:
  slide_type: fragment
---
_ = fom.transfer_function.mag_plot(w)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Projection-based MOR

### Ritz/Petrov-Galerkin Projection

$$
\begin{align*}
  E \dot{x}(t) - A x(t) - B u(t) & = 0, \\
  y(t) - C x(t) - D u(t) & = 0.
\end{align*}
$$

+++ {"slideshow": {"slide_type": "subslide"}}

**Step I: Use truncated state transformation**

Replace

$$
x(t) \approx V \hat{x}(t)
$$

with $V \in \mathbb{R}^{n \times r}$ and $\hat{x}(t) \in \mathbb{R}^{r}$.

$$
\begin{align*}
  E V \dot{\hat{x}}(t) - A V \hat{x}(t) - B u(t) & = e_{\text{res}}(t), \\
  y(t) - C V \hat{x}(t) - D u(t) & = e_{\text{output}}(t).
\end{align*}
$$

+++ {"slideshow": {"slide_type": "subslide"}}

**Step II: Mitigate transformation error**

Suppress truncation residual through left projection.

- one-sided method: use $V$ again.

  $$
  \begin{align*}
    V^{\operatorname{T}} E V \dot{\hat{x}}(t)
    - V^{\operatorname{T}} A V \hat{x}(t)
    - V^{\operatorname{T}} B u(t)
    & = 0, \\
    y(t) - C V \hat{x}(t) - D u(t) & = e_{\text{output}}(t).
  \end{align*}
  $$

+++ {"slideshow": {"slide_type": "subslide"}}

- two-sided method: find $W \in \mathbb{R}^{n \times r}$.

  $$
  \begin{align*}
    W^{\operatorname{T}} E V \dot{\hat{x}}(t)
    - W^{\operatorname{T}} A V \hat{x}(t)
    - W^{\operatorname{T}} B u(t)
    & = 0, \\
    y(t) - C V \hat{x}(t) - D u(t) & = e_{\text{output}}(t).
  \end{align*}
  $$

+++ {"slideshow": {"slide_type": "subslide"}}

<center>
<img src="/files/figures/compress_A.svg" alt="compress A" width="80%">
</center>

+++ {"slideshow": {"slide_type": "subslide"}}

### Reduced order model (ROM) (pyMOR: [`LTIPGReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/basic/index.html?highlight=ltipgred#pymor.reductors.basic.LTIPGReductor))

Define
$\hat{E} = W^{\operatorname{T}} E V$,
$\hat{A} = W^{\operatorname{T}} A V \in \mathbb{R}^{r \times r}$,
$\hat{B} = W^{\operatorname{T}} B \in \mathbb{R}^{r \times m}$, and
$\hat{C} = C V \in \mathbb{R}^{p \times r}$.
Then

$$
\begin{equation}\tag{ROM}
  \begin{aligned}
    \hat{E} \dot{\hat{x}}(t) & = \hat{A} \hat{x}(t) + \hat{B} u(t), \\
    \hat{y}(t) & = \hat{C} \hat{x}(t) + D u(t)
  \end{aligned}
\end{equation}
$$

approximates the dynamics of the full-order model $\Sigma$ with output error

$$
y(t) - \hat{y}(t) = e_{\text{output}}(t).
$$

- We call the corresponding transfer function $\hat{H}$.
- Model order reduction (MOR) $\leadsto$
  Find $W, V \in \mathbb{R}^{n \times r}$ such that $e_{\text{output}}(t)$ is
  small in a suitable sense.
- We will focus on eigenvalue-based, energy-based and
  interpolation-based methods today.

+++ {"slideshow": {"slide_type": "subslide"}}

### Example

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
from pymor.reductors.basic import LTIPGReductor

V = fom.solution_space.random(10)
pg = LTIPGReductor(fom, V, V)
rom_pg = pg.reduce()
```

+++ {"slideshow": {"slide_type": "subslide"}}

The resulting model `rom_pg` will again be an
[`LTIModel`](https://docs.pymor.org/2023-1-0/autoapi/pymor/models/iosys/index.html?highlight=ltimodel#pymor.models.iosys.LTIModel).

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
rom_pg
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_pg.transfer_function.mag_plot(w, label='Random PG')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise

- Simulate the state of the `rom_pg` with $u(t) = e^{-t}$ from $t_{start}=0$ to
  $t_{end}=2$ using its
  [`solve`](https://docs.pymor.org/2023-1-0/autoapi/pymor/models/interface/index.html#pymor.models.interface.Model.solve)
method.
- The solve method computes a
  [`VectorArray`](https://docs.pymor.org/2023-1-0/autoapi/pymor/vectorarrays/interface/index.html?highlight=vectorarray#pymor.vectorarrays.interface.VectorArray)
  with all computed state vectors from the time-domain simulation.
  Consider the last state (i.e., at time $t_{end}$) from the previous
  computation.
  Use the
  [`reconstruct`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/basic/index.html?highlight=reconstruct#pymor.reductors.basic.LTIPGReductor.reconstruct)
  method of `pg` to obtain the reconstructed state vector in the full-order
  model state space.
- Perform the same simulation with `fom`.
  Compare the final state with the state that has been reconstructed using the
  ROM simulation.

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---

```

+++ {"slideshow": {"slide_type": "slide"}}

# System Analysis

## System Norms and Hardy Spaces

We have $$Y(s) = H(s) U(s)$$ and $$\hat{Y}(s) = \hat{H}(s) U(s).$$

> ### Question
>
> What are suitable norms such that
>
> $$
  \lVert y - \hat{y} \rVert
  \le
  \left\lVert H - \hat{H} \right\rVert
  \lVert u \rVert?
  $$

+++ {"slideshow": {"slide_type": "subslide"}}

### The Banach Space $\mathcal{H}_\infty^{p \times m}$

$$
\mathcal{H}_\infty^{p \times m}
:=
\left\{
  G \colon \mathbb{C}^+ \to \mathbb{C}^{p \times m} :
  G \text{ is analytic in $\mathbb{C}^+$ and }
  \sup_{s \in \mathbb{C}^+} \left\lVert G(s) \right\rVert_2 < \infty
\right\}.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$\mathcal{H}_\infty^{p \times m}$ is a Banach space equipped with the
***$\mathcal{H}_\infty$-norm***

$$
\left\lVert G \right\rVert_{\mathcal{H}_\infty}
:= \sup_{\omega \in \mathbb{R}}
\left\lVert G(\boldsymbol{\imath} \omega) \right\rVert_2.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

> Can show:
>
> $$
  \lVert y - \hat{y} \rVert_{\mathcal{L}_{2}}
  \le
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_{\infty}}
  \lVert u \rVert_{\mathcal{L}_{2}}.
  $$

This bound can even be shown to be sharp.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
fom.hinf_norm()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### The Hilbert Space $\mathcal{H}_2^{p \times m}$

$$
\mathcal{H}_2^{p \times m}
:= \left\{
  G \colon \mathbb{C}^+ \to \mathbb{C}^{p \times m} :
  G \text{ is analytic in $\mathbb{C}^+$ and }
  \sup_{\xi > 0}
  \int_{-\infty}^\infty
  \left\lVert
  G(\xi + \boldsymbol{\imath} \omega)
  \right\rVert_{\operatorname{F}}^2
  \operatorname{d}\!{\omega}
  < \infty
\right\}.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

$\mathcal{H}_2^{p \times m}$ is a Hilbert space with the inner product

$$
\langle F, G \rangle_{\mathcal{H}_2}
:=
\frac{1}{2 \pi}
\int_{-\infty}^\infty
\operatorname{tr}\!\left(
  {F(\boldsymbol{\imath} \omega)}^{\operatorname{H}}
  G(\boldsymbol{\imath} \omega)
\right)
\operatorname{d}\!{\omega}
$$

and induced norm

$$
\left\lVert G \right\rVert_{\mathcal{H}_2}
:= \langle G, G \rangle_{\mathcal{H}_2}^{1/2}
= {
  \left(
    \frac{1}{2 \pi}
    \int_{-\infty}^\infty
    \left\lVert G(\boldsymbol{\imath} \omega) \right\rVert_{\operatorname{F}}^2
    \operatorname{d}\!{\omega}
  \right)
}^{1/2}.
$$

+++ {"slideshow": {"slide_type": "fragment"}}

> Can show:
>
> $$
  \lVert y - \hat{y} \rVert_{\mathcal{L}_{\infty}}
  \le
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_{2}}
  \lVert u \rVert_{\mathcal{L}_{2}}.
  $$

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
fom.h2_norm()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### System Gramians and $\mathcal{H}_{2}$ trace formula

A system $\Sigma$ with $\Lambda(E, A) \subset \mathbb{C}^{-}$ is called
***asymptotically stable***.
Then, all state trajectories decay exponentially as $t \to \infty$ and

- the infinite controllability and observability ***Gramians*** exist:

  $$
  \begin{align*}
    P
    & =
      \int_0^{\infty}
      e^{E^{-1} A t}
      E^{-1}
      B B^{\operatorname{T}}
      E^{-\operatorname{T}}
      e^{A^{\operatorname{T}} E^{-\operatorname{T}} t}
      \operatorname{d}\!{t} \\
    E^{\operatorname{T}} Q E
    & =
      \int_0^{\infty}
      e^{A^{\operatorname{T}} E^{-\operatorname{T}} t}
      C^{\operatorname{T}} C
      e^{E^{-1} A t}
      \operatorname{d}\!{t}.
  \end{align*}
  $$
- $P$, $Q$ solve the two ***Lyapunov equations***

  $$
  A P E^{\operatorname{T}} + E P A^{\operatorname{T}} + B B^{\operatorname{T}} = 0, \qquad
  A^{\operatorname{T}} Q E + E^{\operatorname{T}} Q A + C^{\operatorname{T}} C = 0
  $$
<!-- - If $(A, B)$ is controllable and $(A, C)$ is observable, -->
<!--   it moreover holds that $P = P^{\operatorname{T}} \succ 0$ and $Q = Q^{\operatorname{T}} \succ 0$. -->
<!--   (Otherwise we just have $P = P^{\operatorname{T}} \succcurlyeq 0$ and -->
<!--   $Q = Q^{\operatorname{T}} \succcurlyeq 0$.) -->
- the $\mathcal{H}_{2}$-norm can be expressed as

  $$
  \lVert H \rVert_{\mathcal{H}_{2}}^{2}
  = \operatorname{tr}\!\left(C P C^{\operatorname{T}}\right)
  = \operatorname{tr}\!\left(B^{\operatorname{T}} Q B\right).
  $$

+++ {"slideshow": {"slide_type": "slide"}}

# A Selection of MOR Methods

+++ {"slideshow": {"slide_type": "subslide"}}

## Modal Methods

+++ {"slideshow": {"slide_type": "subslide"}}

### Modal Coordinates

Assume that the pair $(E, A)$ is simultaneously diagonalizable in
$\mathbb{C}^{n \times n}$.

> #### Classic Modal Truncation
>
> - Compute diagonal realization from an eigendecomposition.
> - State-space transformation matrices contain eigenvectors (modes).
> - Use $W = V$.
> - Populate $V$ with modes corresponding to eigenvalues closest to
>   $\boldsymbol{\imath} \mathbb{R}$.
> - Add a few domain-specific or "anxiety" modes.

> #### Problem
>
> - Does not take inputs and outputs into account!
> - How many "anxiety" modes are necessary?

+++ {"slideshow": {"slide_type": "subslide"}}

### Dominant Poles Approximation (pyMOR: [`MTReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/mt/index.html?highlight=mtreductor#pymor.reductors.mt.MTReductor))

Recall the pole residue form of the transfer function

$$
H(s) = \sum_{i = 1}^{n} \frac{R_{i}}{s - \lambda_{i}},
$$

where $R_{i} = (C v_{i})(w_{i}^{\operatorname{H}} B)$, assuming
$w_{i}^{\operatorname{H}} E v_{i} = 1$.

+++ {"slideshow": {"slide_type": "fragment"}}

Suppose the modes are sorted based on the magnitude of the
$\lVert R_{i} \rVert / \operatorname{Re}(\lambda_{i})$.
Then we use the truncated pole residue form

$$
H(s) = \sum_{i = 1}^{r} \frac{R_{i}}{s - \lambda_{i}},
$$

as our ROM. This is motivated by the following

> #### Error bound
>
> $$
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_\infty}
  \le
  \sum_{i = r + 1}^{n}
  \frac{\lVert R_{i} \rVert}{\lvert \operatorname{Re}(\lambda_{i}) \rvert}
  $$

+++ {"slideshow": {"slide_type": "fragment"}}

Computation is feasible via *subspace accelerated MIMO dominant pole algorithm*
(SAMDP).

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.mt import MTReductor

mt = MTReductor(fom)
rom_mt = mt.reduce(10)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_mt.transfer_function.mag_plot(w, label='MT')
_ = plt.legend()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
err_mt = fom - rom_mt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = err_mt.transfer_function.mag_plot(w, label='MT')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise

- Compute the $\mathcal{H}_2$-norm of the error system `err_mt`.
  Note that the
  [`LTIModel`](https://docs.pymor.org/2023-1-0/autoapi/pymor/models/iosys/index.html?highlight=ltimodel#pymor.models.iosys.LTIModel)
  class has an
  [`h2_norm`](iosys/index.html?highlight=ltimodel#pymor.models.iosys.LTIModel.h2_norm)
  method.
- Consider the input $u(t) = e^{-t}$ which has the $\mathcal{L}_{2}$-norm
  $\lVert u \rVert_{\mathcal{L}_{2}} = \frac{\sqrt{2}}{2}$.
  Simulate the error system `err_mt` with the input $u(t) = e^{-t}$ and verify
  the input-output error bound $\lVert y - \hat{y} \rVert_{\mathcal{L}_{\infty}}
  \le
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_{2}}
  \lVert u \rVert_{\mathcal{L}_{2}}$.

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---

```

+++ {"slideshow": {"slide_type": "subslide"}}

## Balancing-based MOR

### Balanced Truncation aka. Lyapunov Balancing

#### Idea

- The system $\Sigma$, in realization $(E = I, A, B, C)$,
  is called ***balanced***, if the solutions $P, Q$ of the Lyapunov equations

  $$
  A P + P A^{\operatorname{T}} + B B^{\operatorname{T}} = 0, \qquad
  A^{\operatorname{T}} Q + Q A + C^{\operatorname{T}} C = 0,
  $$

  satisfy:
  $P = Q = \operatorname{diag}(\sigma_1, \ldots, \sigma_n)$
  where
  $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_n > 0$.

+++ {"slideshow": {"slide_type": "fragment"}}

- $\{\sigma_1, \ldots, \sigma_n\}$ are the *Hankel singular values (HSVs)* of
  $\Sigma$.

+++ {"slideshow": {"slide_type": "fragment"}}

- A so-called ***balanced realization*** is computed via state-space transformation

  $$
  \begin{align*}
    \mathcal{T} \colon (I, A, B, C) \mapsto {} & (I, T A T^{-1}, T B, C T^{-1}) \\
    & =
      \left(
        I,
        \begin{bmatrix}
          A_{11} & A_{12} \\
          A_{21} & A_{22}
        \end{bmatrix},
        \begin{bmatrix}
          B_{1} \\
          B_{2}
        \end{bmatrix},
        \begin{bmatrix}
          C_{1} & C_{2}
        \end{bmatrix}
      \right).
  \end{align*}
  $$

+++

- In a balanced realization the state variables are sorted based on their
  contribution to the input-output mapping.

+++ {"slideshow": {"slide_type": "fragment"}}

- Truncation removes state variables which are not important for input-output
  behavior $\leadsto$ reduced order model:
  $(I, \hat{A}, \hat{B}, \hat{C}) = (I, A_{11}, B_{1}, C_{1})$.

+++ {"slideshow": {"slide_type": "subslide"}}

### Implementation: The Square Root Method

#### The SR Method (pyMOR: [`BTReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.BTReductor))

1. Compute (Cholesky) factors of the solutions to the Lyapunov equation,

   $$
   P = S^{\operatorname{T}} S, \quad
   Q = R^{\operatorname{T}} R.
   $$

+++ {"slideshow": {"slide_type": "fragment"}}

2. Compute singular value decomposition

   $$
   S R^{\operatorname{T}}
   =
   \begin{bmatrix}
     U_1 & U_2
   \end{bmatrix}
   \begin{bmatrix}
     \Sigma_1 & 0 \\
     0 & \Sigma_2
   \end{bmatrix}
   \begin{bmatrix}
     V_1^{\operatorname{T}} \\
     V_2^{\operatorname{T}}
   \end{bmatrix}.
   $$

+++ {"slideshow": {"slide_type": "fragment"}}

3. Define

   $$
   W := R^{\operatorname{T}} V_1 \Sigma_1^{-1/2}, \quad
   V := S^{\operatorname{T}} U_1 \Sigma_1^{-1/2}.
   $$
4. Then the reduced-order model is
   $(W^{\operatorname{T}} A V, W^{\operatorname{T}} B, C V)$.

+++ {"slideshow": {"slide_type": "subslide"}}

#### Properties

- Lyapunov balancing **preserves asymptotic stability**.
- We have the **a priori error bound**:
  $$
  \left\lVert H - \hat{H} \right\rVert_{\mathcal{H}_{\infty}}
  \le
  2 \sum\limits_{k = r + 1}^{n} \sigma_{k}
  $$

+++ {"slideshow": {"slide_type": "subslide"}}

#### Variants (pyMOR: [`PRBTReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.PRBTReductor), [`BRBTReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.BRBTReductor), [`LQGBTReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.LQGBTReductor))

Other versions for special classes of systems or applications exist, such as

- **positive-real balancing** (passivity-preserving),
- **bounded-real balancing** (contractivity-preserving),
- **linear-quadratic Gaussian balancing**
  (stability preserving, aims at low-order output feedback controllers).

The given ones all compute $P, Q$ as solutions of ***algebraic Riccati
equations*** of the form:

$$
\begin{align*}
  0
  & =
    \tilde{A} P \tilde{E}^{\operatorname{T}}
    + \tilde{E} P \tilde{A}^{\operatorname{T}}
    + \tilde{B} \tilde{B}^{\operatorname{T}}
    \pm \tilde{E} P \tilde{C}^{\operatorname{T}} \tilde{C} P \tilde{E}^{\operatorname{T}} \\
  0
  & =
    \tilde{A}^{\operatorname{T}} Q \tilde{E}
    + \tilde{E}^{\operatorname{T}} Q \tilde{A}
    + \tilde{C}^{\operatorname{T}} \tilde{C}
    \pm \tilde{E}^{\operatorname{T}} Q \tilde{B} \tilde{B}^{\operatorname{T}} Q \tilde{E}.
\end{align*}
$$

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.bt import BTReductor

bt = BTReductor(fom)
rom_bt = bt.reduce(10)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_mt.transfer_function.mag_plot(w, label='MT')
_ = rom_bt.transfer_function.mag_plot(w, label='BT')
_ = plt.legend()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
err_bt = fom - rom_bt
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = err_mt.transfer_function.mag_plot(w, label='MT')
_ = err_bt.transfer_function.mag_plot(w, label='BT')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise

- Aside from a desired order, the
  [`reduce`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.GenericBTReductor.reduce)
  method of the
  [`BTReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.BTReductor)
  allows for specifying a truncation tolerance based on the a priori error
  bound.
  Use the `bt` instance of the
  [`BTReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.BTReductor)
  to compute a ROM based on a specified tolerance `tol=1e-5`.
- Use the
  [`LQGBTReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/bt/index.html?highlight=btreductor#pymor.reductors.bt.LQGBTReductor)
  to reduce `fom` using a truncation tolerance of `tol=1e-5`.
  Check the dimension of the ROM.
- Compare the $\mathcal{H}_{2}$-norms and orders of the ROMs obtained by both BT
  variants.

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---

```

+++ {"slideshow": {"slide_type": "subslide"}}

## Transfer Function Approximation

The transfer function $H$ is a degree-$n$ rational function

$$
H(s) = \frac{P(s)}{Q(s)}, \quad P,Q \text{ polynomials}
$$

such that deg$(P) \leq n$ and deg$(Q) \leq n$.

+++ {"slideshow": {"slide_type": "fragment"}}

MOR via rational approximation:
Find a degree-$r$ rational function $\hat{H}$ with $r \ll n$ such that

$$
H \approx \hat{H}
$$

+++ {"slideshow": {"slide_type": "subslide"}}

### Rational Interpolation

Pick some complex values $\sigma_1, \ldots, \sigma_r$ and enforce interpolation

$$
\hat{H}(\sigma_j) = H(\sigma_j) \quad \text{for } j = 1, \ldots, r.
$$

+++ {"slideshow": {"slide_type": "subslide"}}

Can also pick tangential directions $b_1, \ldots, b_r \in \mathbb{C}^m$ and
$c_1, \ldots, c_r \in \mathbb{C}^p$ and enforce bitangential Hermite
interpolation

$$
\begin{align*}
  \hat{H}(\sigma_j)b_j &= H(\sigma_j) b_j, \\
  c_j^* \hat{H}(\sigma_j) &= c_j^* H(\sigma_j), \\
  c_j^* \hat{H}'(\sigma_j) b_j &= c_j^* H'(\sigma_j) b_j, \\
\end{align*}
\quad \text{for } j=1,\ldots,r.
$$

+++ {"slideshow": {"slide_type": "subslide"}}

### Interpolation via Projection

Given $E,A,B,C$, how to enforce interpolation?

> ### Theorem
>
> Let $\hat{H}$ be the transfer function of the ROM obtained from a Petrov
> Galerkin projection using $V$ and $W$.
> For $b \in \mathbb{C}^m$, $c \in \mathbb{C}^p$ and $\sigma \in \mathbb{C}$ we
> have
>
> 1. $(\sigma E - A)^{-1} B b \in \mathrm{Range}(V)$ implies $H(\sigma) b = \hat{H}(\sigma) b$
> 2. $(\sigma E - A)^{-*} C^* c \in \mathrm{Range}(W)$ implies $c^* H(\sigma) = c^* \hat{H}(\sigma)$
> 3. If 1. and 2. are satisfied, then $c^* H'(\sigma) b = c^* \hat{H}'(\sigma) b$

Using bases $V$ and $W$ for rational Krylov subspaces allows for interpolatory
MOR [[Antoulas/Beattie/Güğercin '20]](https://doi.org/10.1137/1.9781611976083).

In pyMOR
[`LTIBHIReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/interpolation/index.html?highlight=ltibhired#pymor.reductors.interpolation.LTIBHIReductor)
is based on projection and
[`TFBHIReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/interpolation/index.html?highlight=tfbhi#pymor.reductors.interpolation.TFBHIReductor)
only uses evaluations of the transfer function $H$.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.interpolation import LTIBHIReductor

interp = LTIBHIReductor(fom)
sigma = np.array([50, 100, 200, 400, 800])
sigma = np.concatenate((1j * sigma, -1j * sigma))
b = np.ones((len(sigma), fom.dim_input))
c = np.ones((len(sigma), fom.dim_output))
rom_interp = interp.reduce(sigma, b, c)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_mt.transfer_function.mag_plot(w, label='MT')
_ = rom_bt.transfer_function.mag_plot(w, label='BT')
_ = rom_interp.transfer_function.mag_plot(w, label='Interpolation')
_ = plt.legend()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
err_interp = fom - rom_interp
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = err_mt.transfer_function.mag_plot(w, label='MT')
_ = err_bt.transfer_function.mag_plot(w, label='BT')
_ = err_interp.transfer_function.mag_plot(w, label='Interpolation')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

### Iterative Rational Krylov Algorithm (IRKA)

> #### $\mathcal{H}_2$-optimal MOR problem
>
> Find a stable $\hat{H}$ of order $r$ such that
> $\lVert H - \hat{H} \rVert_{\mathcal{H}_{2}}$
> is minimized.

+++ {"slideshow": {"slide_type": "fragment"}}

> #### Interpolatory necessary $\mathcal{H}_2$-optimality conditions
>
> Let $\hat{H}(s) = \sum_{i = 1}^r \frac{\phi_i}{s - \lambda_i}$ be an
> $\mathcal{H}_2$-optimal reduced-order model for $H$.
> Then
> \begin{align*}
    H\!\left(-\overline{\lambda_i}\right)
    & = \hat{H}\!\left(-\overline{\lambda_i}\right), \\
    H'\!\left(-\overline{\lambda_i}\right)
    & = \hat{H}'\!\left(-\overline{\lambda_i}\right),
  \end{align*}
> for $i = 1, 2, \ldots, r$.

+++ {"slideshow": {"slide_type": "subslide"}}

> ***Hermite interpolation*** of the transfer function is necessary for
> $\mathcal{H}_2$-optimality.

+++ {"slideshow": {"slide_type": "fragment"}}

> #### IRKA (pyMOR: [`IRKAReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/h2/index.html?highlight=irkared#pymor.reductors.h2.IRKAReductor))
>
> Fixed point iteration based on interpolatory necessary optimality conditions.

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
from pymor.reductors.h2 import IRKAReductor

irka = IRKAReductor(fom)
rom_irka = irka.reduce(10)
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
_ = fom.transfer_function.mag_plot(w, label='FOM')
_ = rom_mt.transfer_function.mag_plot(w, label='MT')
_ = rom_bt.transfer_function.mag_plot(w, label='BT')
_ = rom_interp.transfer_function.mag_plot(w, label='Interpolation')
_ = rom_irka.transfer_function.mag_plot(w, label='IRKA')
_ = plt.legend()
```

```{code-cell} ipython3
---
slideshow:
  slide_type: subslide
---
err_irka = fom - rom_irka
```

```{code-cell} ipython3
---
slideshow:
  slide_type: '-'
---
_ = err_mt.transfer_function.mag_plot(w, label='MT')
_ = err_bt.transfer_function.mag_plot(w, label='BT')
_ = err_interp.transfer_function.mag_plot(w, label='Interpolation')
_ = err_irka.transfer_function.mag_plot(w, label='IRKA')
_ = plt.legend()
```

+++ {"slideshow": {"slide_type": "subslide"}}

#### Exercise

- Compute 5 random values `sigma` in the interval $[0,1000]$ using
  `np.random.rand`.
  Then use the
  [`TFBHIReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/interpolation/index.html?highlight=tfbhi#pymor.reductors.interpolation.TFBHIReductor)
  to compute a ROM which interpolates `fom.transfer_function` at the random
  values and its complex conjugates along the imaginary axis (i.e., use
  interpolation points `np.concatenate((1j * sigma, -1j * sigma))`).
  Compute the relative $\mathcal{H}_{2}$-error of the resulting ROM.
  Compare it to the relative $\mathcal{H}_{2}$-error of the interpolation points
  chosen in the example above where we chose
  `sigma = np.array([50, 100, 200, 400, 800])`.
- Repeat the previous computations with 10 random values
  (i.e., a total of 20 interpolation points).
- Use the `sigma` specified below as an initial guess for the
  [`reduce`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/h2/index.html?highlight=irkared#pymor.reductors.h2.IRKAReductor.reduce)
  method of the
  [`IRKAReductor`](https://docs.pymor.org/2023-1-0/autoapi/pymor/reductors/h2/index.html?highlight=irkared#pymor.reductors.h2.IRKAReductor):

  ```
  sigma = np.array([50, 100, 200, 400, 800]);
  sigma = np.concatenate((1j * sigma, -1j * sigma))
  ```

```{code-cell} ipython3

```

+++ {"slideshow": {"slide_type": "subslide"}}

<center>Questions?</center>
