# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: school_venv
#     language: python
#     name: school_venv
# ---

# + editable=true slideshow={"slide_type": ""}
# enable logging widget
# %load_ext pymor.discretizers.builtin.gui.jupyter

# + editable=true slideshow={"slide_type": ""} language="html"
# <style>
# .rise-enabled .jp-RenderedHTMLCommon table {
#          font-size: 150%;
# }
#
# .rise-enabled .jp-RenderedHTMLCommon p {
#     font-size: 1.5rem;
# }
#
# .rise-enabled .jp-RenderedHTMLCommon li {
#     font-size: 1.5rem;
# }
#
#
# .rise-enabled .jp-RenderedHTMLCommon h2 {
#     font-size: 2.9rem;
#     font-weight: bold;
# }
#
# .rise-enabled .jp-RenderedHTMLCommon h3 {
#     font-size: 2.0rem;
#     font-weight: bold;
# }
#
# .rise-enabled .jupyter-widget-Collapse-header {
#     font-size: 1rem;
# }
#
# .rise-enabled .jupyter-widget-Collapse-header i{
#     font-size: 1rem;
# }
#
# .rise-enabled .cm-editor {
#     font-size: 1.25rem;
# }
# </style>

# + [markdown] editable=true slideshow={"slide_type": "slide"}
# # Reduced Basis Methods with pyMOR

# + [markdown] editable=true slideshow={"slide_type": ""}
# ## Our Goal
#
# We want to do model order reduction (MOR) for parametric problems.
#
# This means:
#
# - We are given a full-order model (FOM), usually a PDE model, which depends on some set of parameters $\mu \in \mathbb{R}^Q$.
# - We can simulate/solve the FOM for any given $\mu$. But this is costly.
# - We want to simulate the model for many different $\mu$.
#
# **Task:**
#
# - Replace the FOM by a surrogate reduced-order model (ROM).
# - The ROM should be much faster to simulate/solve.
# - The error between the ROM and FOM solution should be small and controllable.
#
# Note: In this tutorial we will only cover the mere basics of reduced basis (RB) methods. The approach has been extended to other types of models (systems, non-linear, inf-sup stable, outputs, ...) and is largely independent of the specific choice of discretization method.

# + [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Building the FOM

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Thermal-block problem
#
# Find $u(x,\mu)$ for $\mu\in\mathcal{P}$ such that
#
# $$
# \begin{align*}
# -\nabla \cdot [d(x, \mu) \nabla u(x,\mu)] &= f(x) & x &\in \Omega, \\
# u(x,\mu) &= 0 & x &\in \partial \Omega,
# \end{align*}
# $$
#
# where $\Omega := [0,1]^2 = \Omega_1 \cup \Omega_2 \cup \Omega_3 \cup \Omega_4$, $f \in L^2(\Omega)$,
#
#
# $$
# d(x, \mu) \equiv \mu_i \quad x \in \Omega_i
# $$
#
# and $\mu \in [\mu_{\min}, \mu_{\max}]^4$.
#
#
# ```
#         (0,1)-----------------(1,1)
#         |            |            |
#         |            |            |
#         |     μ_2    |     μ_3    |
#         |            |            |
#         |            |            |
#         |--------------------------
#         |            |            |
#         |            |            |
#         |     μ_0    |     μ_1    |
#         |            |            |
#         |            |            |
#         (0,0)-----------------(1,0)
# ```

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Setting up an analytical description of the thermal block problem
#
# The thermal block problem already comes with pyMOR:

# + editable=true slideshow={"slide_type": ""}
from pymor.basic import *
p = thermal_block_problem([2,2])

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# Our problem is parameterized:

# + editable=true slideshow={"slide_type": ""}
p.parameters

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Looking at the definition
#
# We can easily look at the definition of `p` by printing its `repr`:

# + editable=true slideshow={"slide_type": ""}
p

# + [markdown] editable=true slideshow={"slide_type": ""}
# It is easy to [build custom problem definitions](https://docs.pymor.org/latest/tutorial_builtin_discretizer.html).

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Weak formulation
#
# Find $u(\mu) \in H^1_0(\Omega)$ such that
#
# $$
# \underbrace{\int_\Omega d(x, \mu(x)) \nabla u(x, \mu) \cdot \nabla v(x) \,dx}
#     _{=:a(u(\mu), v; \mu)}
# = \underbrace{\int_\Omega f(x)v(x) \,dx}
#     _{=:\ell(v)}
#     \qquad \forall v \in H^1_0(\Omega).
# $$

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# ### Galerkin projection onto finite-element space
#
# Let $\mathcal{T}_h$ be an admissible triangulation of $\Omega$ and $V_h:=\mathcal{S}_{h,0}^1(\mathcal{T}_h)$ the corresponding space of piece-wise linear finite-element functions over $\mathcal{T}_h$ which vanish at $\partial\Omega$.
# The finite-element approximation $u_h(\mu) \in V_h$ is then given by
#
#
# $$
#     a(u_h(\mu), v;\mu) = \ell(v_h)
#     \qquad \forall v_h \in V_h.
# $$
#
# Céa's Lemma states that $u_h(\mu)$ is a quasi-best approximation of $u(\mu)$ in $V_h$:
#
# $$
#     \|\nabla u(\mu) - \nabla u_h(\mu)\|_{L^2(\Omega)}
#     \leq \frac{\mu_{max}}{\mu_{min}} \inf_{v_h \in V_h} \|\nabla u(\mu) - \nabla v_h\|_{L^2(\Omega)}.
# $$

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Linear system assembly
#
# Let $\varphi_{h,1}, \ldots, \varphi_{h,n}$ be the finite-element basis of $\mathcal{S}_{h,0}^1(\mathcal{T}_h)$.
# Let $A(\mu) \in \mathbb{R}^{n\times n}$, $\underline{\ell} \in \mathbb{R}^n$ be given by
#
# $$
#     A(\mu)_{j,i} := a(\varphi_{h,i}, \varphi_{h,j};\mu) \qquad
#     \underline \ell_j := \ell(\varphi_{h,j}).
# $$
#
# Then with
# $$
#     u_h(\mu) = \sum_{i=1}^{n} \underline{u}_h(\mu)_i \cdot \varphi_{h,i},
# $$
#
# we get
#
# $$
#     A(\mu) \cdot \underline{u}_h(\mu) = \underline{\ell}.
# $$
#
# Note that $A(\mu)$ is a sparse matrix.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### FOM assembly with pyMOR
#
# We use the builtin discretizer `discretize_stationary_cg` to compute a finite-element discretization of the problem:

# + editable=true slideshow={"slide_type": ""}
fom, data = discretize_stationary_cg(p, diameter=1/100)

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# `fom` is a `Model`. It has the same `Parameters` as `p`:

# + editable=true slideshow={"slide_type": ""}
fom.parameters

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Solving the FOM
#
# To `solve` the FOM, we need to specify values for those parameters:

# + editable=true slideshow={"slide_type": ""}
U = fom.solve({'diffusion': [1., 0.01, 0.1, 1]})

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# `U` is a `VectorArray`, an ordered collection of vectors of the same dimension:

# + editable=true slideshow={"slide_type": ""}
U

# + [markdown] editable=true slideshow={"slide_type": ""}
# `U` only contains a single vector:

# + editable=true slideshow={"slide_type": ""}
len(U)

# + [markdown] editable=true slideshow={"slide_type": ""}
# For a time-dependent problem, `U` would have contained a time-series of vectors. `U` corresponds to the coefficient vector $\underline{u}_h(\mu)$.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Looking at the solution
#
# We can use the `visualize` method to plot the solution:

# + editable=true slideshow={"slide_type": ""}
fom.visualize(U)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Your turn
#
# - Define a 2x3 thermal-block problem.
# - Build the FOM using pyMOR's builtin discretization toolkit over a mesh with element diameter 1/20.
# - Solve the FOM for some parameter and visualize the solution.

# + editable=true slideshow={"slide_type": ""}
p23 = ...
fom23 = ...
U23 = ...

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Parameter separability
#
# Remember the special form of $a(\cdot, \cdot; \mu)$:
#
# $$
# \begin{align}
#     a(u, v; \mu) &:= \int_\Omega d(x, \mu) \nabla u(x) \cdot \nabla v(x) \,dx \\
#     &:=\int_\Omega \Bigl(\sum_{q=1}^Q \mu_q \mathbb{1}_q(x)\Bigr) \nabla u(x) \cdot \nabla v(x) \,dx \\
#     &:=\sum_{q=1}^Q  \ \underbrace{\mu_q}_{:=\theta_q(\mu)} \ \ 
#         \underbrace{\int_\Omega \mathbb{1}_q(x) \nabla u(x) \cdot \nabla v(x) \,dx}_{=:a_q(u,v)}.
# \end{align}
# $$
#
# Hence, $a(\cdot, \cdot; \mu)$ admits the affine decomposition
#
# $$
#     a(u, v; \mu) = \sum_{q=1}^Q \theta_q(\mu) \cdot a_q(u,v).
# $$
#
# Consequently, for $A(\mu)$ we have the same structure:
#
# $$
#     A(\mu) = \sum_{q=1}^Q \theta_q(\mu) \cdot A_q,
# $$
#
# where $(A_q)_{j,i} := a_q(\varphi_{h,i}, \varphi_{h,j})$.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Parameter-separable FOM
#
# Remember that our problem definition encoded the affine decomposition of $d(x, \mu)$ using a `LincombFunction`:

# + editable=true slideshow={"slide_type": ""}
p.diffusion

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# pyMOR's builtin `discretizer` automatically preserves this structure when assembling the system matrices. Let's look at the `fom` in more detail. The system matrix $A(\mu)$ is stored in the `Model`'s `operator` attribute:

# + editable=true slideshow={"slide_type": ""}
fom.operator

# + [markdown] editable=true slideshow={"slide_type": ""}
# We see that the `LincombFunction` has become a `LincombOperator` of `NumpyMatrixOperators`.
# pyMOR always interprets matrices as linear `Operators`.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# The right-hand side vector $\underline{\ell}$ is stored in the `rhs` attribute:

# + editable=true slideshow={"slide_type": ""}
fom.rhs

# + [markdown] editable=true slideshow={"slide_type": ""}
# `fom.rhs` is not a `VectorArray` but a vector-like operator in order to support parameter-dependent right-hand sides. Only `Operators` can depend on a parameter in `pyMOR`, not `VectorArrays`.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Other ways of obtaining the FOM
#
# > Using an `analyticalproblem` and a `discretizer` is just one way
#   to build the FOM.
# >  
# > Everything that follows works the same for a FOM that is built using an external PDE solver.

# + [markdown] editable=true slideshow={"slide_type": "slide"}
# ## VectorArrays and Operators

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Some words about VectorArrays
#
# Each `VectorArray` has a length, i.e, the number of vectors in the array:

# + editable=true slideshow={"slide_type": ""}
len(U)

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# > There is not the notion of a single vector in pyMOR! Don't try to get hold of one!

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# Its dimension is the *uniform* size of each vector in the array:

# + editable=true slideshow={"slide_type": ""}
U.dim

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# When using pyMOR's builtin discretizations, we use `NumpyVectorArrays`:

# + editable=true slideshow={"slide_type": ""}
type(U)

# + [markdown] editable=true slideshow={"slide_type": ""}
# These arrays internally store their vectors using a 2d `NumPy` array:

# + editable=true slideshow={"slide_type": ""}
U.impl._array

# + [markdown] editable=true slideshow={"slide_type": ""}
# When using an external PDE solver for the FOM, we usually use `ListVectorArrays`, which manage a Python list of vector objects that directly correspond to vector data in the PDE solvers memory.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Supported Operations
#
# |                  |                                                         |
# | :-               | :-                                                      |
# | `append`         | append vectors from another array                       |
# | `+`/`-`/`*`      | element-wise addition/subtraction/scalar multiplication |
# | `inner`          | matrix of inner products between all vectors            |
# | `pairwise_inner` | list of pairwise inner products                         |
# | `norm`           | list of norms                                           |
# | `lincomb`        | linear combination of the vectors in the array          |
# | `scal`           | in-place scalar multiplication                          |
# | `axpy`           | in-place BLAS axpy operation                            |
# | `dofs`           | return a few degrees of freedom as NumPy array          |

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Playing a bit with VectorArrays
#
# It is important to note that `VectorArrays` are never instantiated directly. All `VectorArrays` are created by their `VectorSpace`:

# + editable=true slideshow={"slide_type": ""}
V = fom.solution_space.empty()

# + [markdown] editable=true slideshow={"slide_type": ""}
# Let's accumulate some solutions:

# + editable=true slideshow={"slide_type": ""}
for mu in p.parameter_space.sample_randomly(10):
    V.append(fom.solve(mu))

# + [markdown] editable=true slideshow={"slide_type": ""}
# Indeed, `V` now contains 10 vectors:

# + editable=true slideshow={"slide_type": ""}
len(V)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# We can visualize all the solutions as a time series:

# + editable=true slideshow={"slide_type": ""}
fom.visualize(V)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Your turn
#
# - Compute the (Euclidean) norms of the vectors in `V` using the `norm` method.
# - Compute all pairwise inner products between the vectors in `V` using the `inner` method.
# - Compute the sum of all vectors in `V` using the `lincomb` method.

# + editable=true slideshow={"slide_type": ""}
# your code here

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Indexing
# We can index a `VectorArray` using numbers, sequences of numbers, or slices, e.g.:

# + editable=true slideshow={"slide_type": ""}
V_indexed = V[3:6]

# + [markdown] editable=true slideshow={"slide_type": ""}
# Indexing **always** creates a **view** into the original array:

# + editable=true slideshow={"slide_type": ""}
print(V_indexed.is_view)
V_indexed *= 0
V.norm()

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Operators can be applied to VectorArrays
#
# To apply a pyMOR `Operator` to a given input `VectorArray`, we use the `Operator`'s apply method:

# + editable=true slideshow={"slide_type": ""}
fom.operator.apply(V, mu=[1,2,3,4])

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# We did something wrong here. pyMOR complains that `mu`, the values for the parameters, is not a `Mu` instance. Except for high-level interface methods like `solve`, parameter values *always* need to be passed as `Mu` objects. We follow pyMOR's advice:

# + editable=true slideshow={"slide_type": ""}
W = fom.operator.apply(V, mu=fom.parameters.parse([1,2,3,4]))

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# `apply` loops over all vectors in `V` and applies the operator individually to each vector. For a matrix operator, this corresponds to a matrix-vector product. The result is a new `VectorArray` of the same length from the `range` `VectorSpace` of the `Operator`:

# + editable=true slideshow={"slide_type": ""}
print(len(V))
print(V in fom.operator.range)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Is the solution really a solution?
#
# We compute the residual:

# + editable=true slideshow={"slide_type": ""}
mu = fom.parameters.parse([1., 0.01, 0.1, 1])
U = fom.solve(mu)
(fom.operator.apply(U, mu=mu) - fom.rhs.as_vector(mu)).norm()

# + [markdown] editable=true slideshow={"slide_type": ""}
# We used `as_vector` here to convert the right-hand side operator of the `Model` to a corresponding `VectorArray`.
#
# > If you implement a new `Model`, make sure that `solve` really returns solutions with zero residual!

# + [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Reduced basis methods

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Projection-based MOR
#
# Going back to the definition of the FOM
#
# $$
#     a(u_h(\mu), v; \mu) = \ell(v_h) \qquad \forall v_h \in V_h,
# $$
#
# our MOR approach is based on the idea of replacing the generic finite-element space $V_h$ by a problem-adapted reduced space $V_N$ of low dimension. I.e., we simply define our ROM by a Galerkin projection of the solution onto the reduced space $V_N$. So the reduced approximation $u_N(\mu) \in V_N$ of $u_h(\mu)\in V_h$ is given as the solution of
#
# $$
#     a(u_N(\mu), v_N; \mu) = \ell(v_N) \qquad \forall v_N \in V_N.
# $$
#
# Again, we can apply Céa's Lemma:
#
# $$
#     \|\nabla u_h(\mu) - \nabla u_N(\mu)\|_{L^2(\Omega)}
#     \leq \frac{\mu_{max}}{\mu_{min}} \inf_{\color{red}v_N \in V_N} \|\nabla u_h(\mu) - \nabla v_N\|_{L^2(\Omega)}.
# $$

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Does a good reduced space $V_N$ exist?
#
# Thanks to Céa's lemma, our only job is to come up with a good low-dimensional approximation space $V_N$. In RB methods, our definition of 'good' is usually that we want to miminize the worst-case best-approximation error over all parameters $\mu \in \mathcal{P}$. I.e.,
#
# $$
#     \sup_{\mu \in \mathcal{P}} \inf_{v_N \in V_N} \|\nabla u_h(\mu) - \nabla v_N\|_{L^2(\Omega)}
# $$
#
# should not be much larger than the Kolmogorov $N$-width
#
# $$
#     d_N:=\inf_{\substack{V'_N \subset V_h\\ \dim V'_N \leq N}}\sup_{\mu \in \mathcal{P}} \inf_{v'_N \in V'_N} \|\nabla u_h(\mu) - \nabla v'_N\|_{L^2(\Omega)}.
# $$
#
# We won't go into details here, but it can be shown that for parameter-separable coercive problems like the thermal-block problem, the Kolmogorov $N$-widths decay at a subexponential rate, so good reduced spaces $V_N$ of small dimension $N$ do exist.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Snapshot-based MOR
#
# The question remains how to find a good $V_N$ algorithmically. RB methods are snapshot based which means that $V_N$ is constructed from 'solution snapshots' $u_{h}(\mu_i)$ of the FOM, i.e.
#
# $$
#     V_N := \operatorname{span} \{u_h(\mu_1), \ldots, u_h(\mu_N)\}.
# $$
#
# We will start by just randomly picking some snapshot parameters $\mu_i\in\mathcal{P}$:

# + editable=true slideshow={"slide_type": ""}
snapshot_parameters = p.parameter_space.sample_randomly(10)
snapshots = fom.solution_space.empty()
for mu in snapshot_parameters:
    snapshots.append(fom.solve(mu))

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# For numerical stability, it's a good idea to orthonormalize the basis:

# + editable=true slideshow={"slide_type": ""}
basis = gram_schmidt(snapshots)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Is our basis any good?
#
# Let's see if we actually constructed a good approximation space by computing the best-approximation error in this space for some further random solution snapshot. We can do so via orthogonal projection:

# + editable=true slideshow={"slide_type": ""}
U_test = fom.solve(p.parameter_space.sample_randomly())
coeffs = U_test.inner(basis)
U_test_proj = basis.lincomb(coeffs)
fom.visualize((U_test, U_test_proj, U_test-U_test_proj),
              legend=('U', 'projection', 'error'),
              separate_colorbars=True)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# Let's also compute the norm of the error:

# + editable=true slideshow={"slide_type": ""}
(U_test - U_test_proj).norm().item() / U_test.norm().item()

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Your turn
#
# - Generate a plot 'projection error' vs. 'basis size'.
# - Use `VectorArray` slicing to project onto the first $k$ vectors in `basis`. (There are other ways.)
# - Use `matplotlib.pyplot.semilogy`.

# + editable=true slideshow={"slide_type": ""}
# your code here ...

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Assembling the reduced system matrix
#
# In order to compute a reduced solution, we need to choose a reduced basis $\psi_{1}, \ldots, \psi_{N}$ of $V_N$ and assemble the reduced system matrix $A_{N}(\mu) \in \mathbb{R}^{N\times N}$ and right-hand side vector $\underline{\ell}_N \in \mathbb{R}^N$ given by
#
# $$
#     A_N(\mu)_{j,i} := a(\psi_i, \psi_j; \mu) \qquad
#     \underline{\ell}_{N,j} := \ell(\psi_j).
# $$
#
# Expanding each basis vector $\psi_i$ w.r.t. the finite-element basis $\varphi_{h,i}$,
#
# $$
#     \psi_i = \sum_{k=1}^N \underline{\psi}_{i,k} \varphi_{h,k},
# $$
#
# we get
#
# $$
#     A_N(\mu)_{i,j} = \underline{\psi}_i^{\operatorname{T}} \cdot A(\mu) \cdot \underline{\psi}_j.
# $$

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# Thus, we could compute $A_N(\mu)$ in pyMOR using `W = fom.operator.apply(basis, mu=mu)` (multiplication from the right) and then using `basis.inner(W)` to multiply the basis from the left. We can use the `apply2` method as a (potentially more efficient) shorthand:

# + editable=true slideshow={"slide_type": ""}
mu = p.parameter_space.sample_randomly()
A_N = fom.operator.apply2(basis, basis, mu=mu)
A_N.shape

# + [markdown] editable=true slideshow={"slide_type": ""}
# Note that, contrary to the finite-element system matrix $A(\mu)$, the reduced matrix $A_N(\mu)$ is a dense but small matrix.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Assembling the reduced right-hand side
#
# For the right-hand side we have
#
# $$
#     \underline{\ell}_{N,j} = \underline{\psi}_j^{\operatorname{T}} \cdot \underline{\ell},
# $$
#
# which we compute using `inner`:

# + editable=true slideshow={"slide_type": ""}
l_N = basis.inner(fom.rhs.as_vector())
l_N.shape

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Solving the reduced system
#
# Finally, writing
#
# $$
#     u_N(\mu) = \sum_{i=1}^N \underline{u}_N(\mu)_i \cdot \psi_i
# $$
#
# we have
#
# $$
#     A_N(\mu) \cdot \underline{u}_N(\mu) = \underline{\ell}_N
# $$

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# So, let's solve the linear system and compare the reduced solution to the FOM solution:

# + editable=true slideshow={"slide_type": ""}
import numpy as np
u_N = np.linalg.solve(A_N, l_N)
U_N = basis.lincomb(u_N.ravel())
U = fom.solve(mu)
fom.visualize((U, U_N, U-U_N),
              legend=('FOM', 'ROM', 'Error'),
              separate_colorbars=True)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Automatic structure-preserving operator projection
#
# For each new parameter $\mu$ we want to solve the ROM for, we have to assemble a new $A_N(\mu)$, which requires $\mathcal{O}(N^2)$ high-dimensional operations. This can significantly diminish the efficiency of our ROM. However, we can avoid this issue by exploiting the parameter separability of $A(\mu)$,
#
# $$
#     A(\mu) = \sum_{q=1}^Q \theta_q(\mu) \cdot A_q,
# $$
#
# which is inherited by $A_N(\mu)$:
#
# $$
#     A_N(\mu) = \sum_{q=1}^Q \theta_q(\mu) \cdot A_{N,q},
# $$
# where $(A_{N,q})_{i,j} = \underline{\psi}_i^{\operatorname{T}} \cdot A_q \cdot \underline{\psi}_j$.
#
# Thus, we have to project all operators in `fom.operator.operators` individually and then later form a linear combination of these matrices.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# This is getting tedious, so we let pyMOR do the work for us:

# + editable=true slideshow={"slide_type": ""}
op_N = project(fom.operator, basis, basis)
op_N

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# Similarly, we can project the right-hand side:

# + editable=true slideshow={"slide_type": ""}
rhs_N = project(fom.rhs, basis, None)
rhs_N

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# Now, we could assemble a matrix operator from `op_N` for a specific `mu` using the `assemble` method:

# + editable=true slideshow={"slide_type": ""}
op_N_mu = op_N.assemble(mu)
op_N_mu

# + [markdown] editable=true slideshow={"slide_type": ""}
# Then, we can extract it's system matrix:

# + editable=true slideshow={"slide_type": ""}
op_N_mu.matrix.shape

# + [markdown] editable=true slideshow={"slide_type": ""}
# From that, we can proceed as before. However, it is more convenient, to use the operator's `apply_inverse` method to invoke an (`Operator`-dependent) linear solver with a given input `VectorArray` as right-hand side:

# + editable=true slideshow={"slide_type": ""}
u_N_new = op_N.apply_inverse(rhs_N.as_vector(), mu=mu)
u_N_new

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# Note that the result is a `VectorArray`. For `NumpyVectorArray` and some other `VectorArray` types, we can extract the internal data using the `to_numpy` method. We use it to check whether we arrived at the same solution:

# + editable=true slideshow={"slide_type": ""}
np.linalg.norm(u_N.ravel() - u_N_new.to_numpy().ravel())

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Projecting the entire Model
#
# In pyMOR, ROMs are built using a `Reductor` which appropriately projects all of the `Models` operators and returns a reduced `Model` comprised of the projected `Operators`. Let's pick the most basic `Reductor`
# available for a `StationaryModel`:

# + editable=true slideshow={"slide_type": ""}
reductor = StationaryRBReductor(fom, basis)

# + [markdown] editable=true slideshow={"slide_type": ""}
# Every reductor has a `reduce` method, which builds the ROM:

# + editable=true slideshow={"slide_type": ""}
rom = reductor.reduce()

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# Let's compare the structure of the FOM and of the ROM

# + editable=true slideshow={"slide_type": ""}
fom

# + editable=true slideshow={"slide_type": "subslide"}
rom

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Solving the ROM
#
# To solve the ROM, we just use `solve` again,

# + editable=true slideshow={"slide_type": ""}
u_rom = rom.solve(mu)

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# to get the reduced coefficients:

# + editable=true slideshow={"slide_type": ""}
u_rom

# + [markdown] editable=true slideshow={"slide_type": ""}
# It is the same coefficient vector we have computed before:

# + editable=true slideshow={"slide_type": ""}
(u_rom - u_N_new).norm()

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# A high-dimensional representation is obtained from the `reductor`:

# + editable=true slideshow={"slide_type": ""}
U_rom = reductor.reconstruct(u_rom)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Computing the MOR error
#
# Let's compute the error again:

# + editable=true slideshow={"slide_type": ""}
U = fom.solve(mu)
ERR = U - U_rom
ERR.norm() / U.norm()

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# and look at it:

# + editable=true slideshow={"slide_type": ""}
fom.visualize(ERR)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Your turn
#
# - Verify that the MOR error vanishes for the snapshot parameters $\mu_i$ used to build $V_N$ (this is also called *snapshot reproduction*).

# + editable=true slideshow={"slide_type": ""}
# your code here

# + [markdown] editable=true slideshow={"slide_type": "slide"}
# ## Certified Reduced Basis Method

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Error estimator
# Model order reduction introduces an additional approximation error which we need to control in order to be able to use a ROM as a reliable surrogate for a given FOM. While Céa's lemma provides a rigorous a priori bound, this error bound is not computable in general. Instead, we use a residual-based a posteriori error estimator. As in a posteriori theory for finite-element methods, we have:
#
# $$
#     \|\nabla u_h(\mu) - \nabla u_N(\mu)\|_{L^2(\Omega)}
#     \leq \frac{1}{\mu_{min}} \sup_{v_h\in V_h} \frac{\ell(v_h) - a(u_N(\mu), v_h; \mu)}{\|\nabla v_h\|_{L^2(\Omega)}}.
# $$
#
# For this estimate to hold, it is crucial that we use the right norms. I.e., instead of the Euclidean norm of the coefficient vectors, which we have used so far, we need to use the $H^1$-seminorm. 
#
# The inner product matrix of the $H^1$-seminorm is automatically assembled by pyMOR's builtin discretizer and available as `fom.h1_0_semi_product`. We can pass it as the `product`-argument to methods like `norm`, `inner` or `gram_schmidt` to perform these operations w.r.t. the correct inner product/norm. Further, we need a lower bound for the coercivity constant of $a(\cdot, \cdot; \mu)$.

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# Using this information, we can replace `StationaryRBReductor` by `CoerciveRBReductor`, which will add a reduction-error estimator to our ROM:

# + editable=true slideshow={"slide_type": ""}
basis = gram_schmidt(snapshots, product=fom.h1_0_semi_product)
reductor = CoerciveRBReductor(
   fom, basis,
   product=fom.h1_0_semi_product,
   coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)
rom = reductor.reduce()

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# We won't go into details here, but an 'offline-online decomposition' of the error estimator is possible similar to what we did for the projection of the system operator:

# + editable=true slideshow={"slide_type": ""}
rom.error_estimator.residual

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# Let's check if the estimator works:

# + editable=true slideshow={"slide_type": ""}
U = fom.solve(mu)
u_N = rom.solve(mu)
est = rom.estimate_error(mu).item()
err = (U - reductor.reconstruct(u_N)).norm(product=fom.h1_0_semi_product).item()
print(f'error: {err}, estimate: {est}')

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Greedy basis generation

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# So far, we have built the reduced space $V_N$ by just randomly picking snapshot parameters. A theoretically well-founded approach which leads to quasi-optimal approximation spaces it the so-called weak greedy algorithm. In the weak greedy algorithm, $V_N$ is constructed iteratively by enlarging $V_N$ by an element $u_h(\mu_{N+1})$ such that
#
# $$ \inf_{v_N \in V_N} \|\nabla u_h(\mu_{N+1}) - \nabla v_N\|_{L^2(\Omega)}
# \geq C \cdot \sup_{\mu \in \mathcal{P}}\inf_{v_N \in V_N} \|\nabla u_h(\mu) - \nabla v_N\|_{L^2(\Omega)}, $$
#
# for some fixed constant $0 < C \leq 1$.

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# In RB methods, we find such a $\mu_{N+1}$ by picking the parameter for which the estimated reduction error is maximized. 
#
# In order to make this maximization procedure computationally feasible, the infinite set $\mathcal{P}$ is replaced by a finite subset of training parameters:

# + editable=true slideshow={"slide_type": ""}
training_set = p.parameter_space.sample_uniformly(4)
len(training_set)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# Given this training set, we can use `rb_greedy` to compute $V_N$. In order to start with an empty basis, we create a new reductor that, by default, is initialized with an empty basis:

# + editable=true slideshow={"slide_type": ""}
reductor = CoerciveRBReductor(
   fom,
   product=fom.h1_0_semi_product,
   coercivity_estimator=ExpressionParameterFunctional('min(diffusion)', fom.parameters)
)
greedy_data = rb_greedy(fom, reductor, training_set, max_extensions=20)
print(greedy_data.keys())
rom = greedy_data['rom']

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Testing the ROM
#
# Let's compute the error again:

# + editable=true slideshow={"slide_type": ""}
mu = p.parameter_space.sample_randomly()
U = fom.solve(mu)
u_rom = rom.solve(mu)
ERR = U - reductor.reconstruct(u_rom)
ERR.norm(fom.h1_0_semi_product)

# + [markdown] editable=true slideshow={"slide_type": "fragment"}
# and compare it with the estimated error:

# + editable=true slideshow={"slide_type": ""}
rom.estimate_error(mu)

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Is it actually faster?
#
# Finally, we check if our ROM is really any faster than the FOM:

# + editable=true slideshow={"slide_type": ""}
from time import perf_counter
mus = p.parameter_space.sample_randomly(10)
tic = perf_counter()
for mu in mus:
    fom.solve(mu)
t_fom = perf_counter() - tic
tic = perf_counter()
for mu in mus:
    rom.solve(mu)
t_rom = perf_counter() - tic
print(f'Speedup: {t_fom/t_rom}')

# + [markdown] editable=true slideshow={"slide_type": "subslide"}
# ### Some more exercises
#
# - Plot the MOR error vs. the dimension of the reduced space. (Use `reductor.reduce(N)` to project onto a sub-basis of dimension `N`.)
#  
# - Plot the speedup vs. the dimension of the reduced space.
#
# - Compute the maximum/minimum efficiency of the error estimator over the parameter space.
#
# - Try different numbers of subdomains.

# +
# your code here
