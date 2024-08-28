# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# # pyMOR Architecture

# + [markdown] slideshow={"slide_type": "fragment"}
# pyMOR's architecture is centered around ***generic*** implementations of model order reduction algorithms.
#
# Algorithms are formulated in terms of ***abstract interfaces*** such that external PDE/LA backends can easily be integrated.

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Motivation
#
# Consider PDE solvers
# - FEniCS
# - deal.II
# - ...
# - proprietary code?
#
# with specific backend for discretized operators and numerical linear algebra operations.

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Can we implement MOR algorithms once in a way such that they are compatible with any desired PDE backend?
#
# For the most part, yes! We only need notions of
# - ***Operators***
# - (Collections of) Vectors $\rightarrow$ ***VectorArrays***
#
# and possible operations that involve these objects.

# + [markdown] slideshow={"slide_type": "slide"}
# ## `VectorArrays`
#
# - Represent a (short) sequence/array of (high-dimensional) vectors.
# - Can represent a tall-and-skinny or short-and-fat matrix.
# - Each vector (and thus the entire collection) lives in an associated `VectorSpace`.
# - `VectorArrays` are never instantiated directly, but rather through a `VectorSpace`.
# - Vectors in pyMOR are just `VectorArrays` of length one.
# - pyMOR uses inheritence:
#   - `pymor.vectorarrays.interface.VectorArray` defines the `VectorArray` interface
#   - all concrete vector array classes need to inherit from `VectorArray`

# + slideshow={"slide_type": "subslide"}
from pymor.basic import NumpyVectorSpace

space = NumpyVectorSpace(10)
print(space.dim)

# + slideshow={"slide_type": "fragment"}
V = space.zeros(3)
print(V.dim)
print(len(V))
print(V)

# + slideshow={"slide_type": "fragment"}
V = space.random(2)
print(V.dim)
print(len(V))
print(V)

# + slideshow={"slide_type": "fragment"}
V = space.ones(1)
print(V.dim)
print(len(V))
print(V)

# + slideshow={"slide_type": "subslide"}
import numpy as np

a = np.arange(30).reshape(10, 3)
print(a)

# + slideshow={"slide_type": "fragment"}
V = space.make_array(a.T)
print(V.dim)
print(len(V))
print(V)

# + [markdown] slideshow={"slide_type": "subslide"}
# Internally, `VectorArrays` appear to be stored as row vectors, but
#
# - `VectorArrays` in pyMOR are not considered ***column or row vectors***, they are simply lists of vectors.
# - Mathematically there is no notion of row/column vectors from e.g. $\mathbb{R}^n$ either.
# - Consequence: There is no `transpose` method for `VectorArrays`.
# - Only typical operations for "mathematical" vectors are supported.

# + slideshow={"slide_type": "subslide"}
V = space.make_array(np.arange(20).reshape(10, 2).T)
print(V)

W = space.make_array(np.arange(0, 40, 2).reshape(10, 2).T)
print(W)

# + slideshow={"slide_type": "fragment"}
print(V + W)

# + slideshow={"slide_type": "fragment"}
print(V - W)

# + slideshow={"slide_type": "fragment"}
print(V * 2)

# + slideshow={"slide_type": "fragment"}
print(V.norm())

# + slideshow={"slide_type": "fragment"}
VW = V.copy()
VW.append(W)
print(VW)

# + [markdown] slideshow={"slide_type": "subslide"}
# ### `VectorArray` vs NumPy array
#
# Both types of arrays are used extensively in pyMOR.
#
# `VectorArrays` are used whenever vectors that live in a (potentially high-dimensional) state-space are represented, e.g.:
#
# - internal state of a `Model`,
# - basis used for projection,
# - vectors that "live" in an external PDE solver.
#
# NumPy arrays are used for (low-dimensional) data which has no direct connection to an external solver, e.g.:
#
# - inputs and outputs of models,
# - coefficients for linear combinations,
# - computed inner products.
# -

print(V)
print(W)

# + slideshow={"slide_type": "subslide"}
print(V.inner(W))

# + slideshow={"slide_type": "fragment"}
V.lincomb(np.array([1, 2]))

# + [markdown] slideshow={"slide_type": "subslide"}
# `VectorArrays` are ***mutable***.
#
# Also, depending on the implementation, they may change the underlying data.
# This is the case for `NumpyVectorArrays`.

# + slideshow={"slide_type": "fragment"}
a = np.array([[1, 2, 3],
              [4, 5, 6]])
V = NumpyVectorSpace.make_array(a)
V.scal(2)
a

# + [markdown] slideshow={"slide_type": "subslide"}
# Depending on the `VectorArray` implementation, `to_numpy` and `from_numpy` methods may be available.
#
# For `NumpyVectorArrays` this clearly works and `from_numpy` and `make_array` essentially do the same thing.

# + slideshow={"slide_type": "fragment"}
print(a)
V = NumpyVectorSpace.from_numpy(a)
V.to_numpy()

# + [markdown] slideshow={"slide_type": "fragment"}
# If you are using only NumPy-based data in your own application, using these methods is typically fine.
#
# However, pyMOR code will generally avoid these methods since they may not be available for all `VectorArray` implementations.

# + [markdown] slideshow={"slide_type": "subslide"}
# Indexing and advanced indexing similar as for NumPy arrays is available and will always create a ***view on the data***.

# + slideshow={"slide_type": "fragment"}
print(V)

# + slideshow={"slide_type": "fragment"}
w = V[-1]
print(w)

# + slideshow={"slide_type": "fragment"}
w.scal(2)

# + slideshow={"slide_type": "fragment"}
print(w)

# + slideshow={"slide_type": "-"}
print(V)

# + [markdown] slideshow={"slide_type": "subslide"}
# Accessing individual entries via indexing is possible via the `dofs` method.

# + slideshow={"slide_type": "fragment"}
print(a[1, 1])
print(V[1].dofs([1]))

# + [markdown] slideshow={"slide_type": "fragment"}
# Only a small number of `dofs` should be extracted at once. This is again motivated by the fact that the `VectorArray` may be using a backend where extracting all (a large number) is either not possible or expensive.
#
# `dofs` is mostly used for empirical interpolation.

# + [markdown] slideshow={"slide_type": "subslide"}
# ### `ListVectorArray`
#
# Linear algebra backends of many PDE solvers only have the notion of a single vector.
#
# pyMOR supports an interface for single vectors via the `Vector` class.
#
# A `VectorArray` can then simply be obtained via the `ListVectorArray` class which can be considerd a collection of corresponding `Vector` instances.
#
# This means for a corresponding PDE solver backend one only needs to specify how a vector behaves by implementing a `*PDESolver*Vector` class and then gets the `VectorArray` for free.

# + slideshow={"slide_type": "fragment"}
from pymor.vectorarrays.list import NumpyListVectorSpace

space = NumpyListVectorSpace(4)
W = space.random(3)
W.impl._list

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Exercise 1
#
# Compute the Frobenius norm of `V.to_numpy()` without using `to_numpy`.

# + slideshow={"slide_type": "-"}
V = NumpyVectorSpace.make_array(np.arange(20).reshape(10, 2).T)
np.linalg.norm(V.to_numpy())

# + slideshow={"slide_type": "-"}
# your code here

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Exercise 2
#
# Compute two linear combinations using a single call of `lincomb`.

# + slideshow={"slide_type": "-"}
print(V.lincomb(np.array([1, 2])))
print(V.lincomb(np.array([3, 4])))

# + slideshow={"slide_type": "-"}
print(...)  # your code here

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Exercise 3
#
# Generate a normally distributed vector with 5 components of mean $1$ and standard deviation $0.1$.

# + slideshow={"slide_type": "-"}
# your code here

# + [markdown] slideshow={"slide_type": "slide"}
# ## `Operators`
#
# - `Operators` in pyMOR define a (non-)parametric, (non-)linear mapping between `VectorSpaces`.
# - This means an `Operator` can be applied to any given `VectorArray` from the `Operator`'s `source` `VectorSpace`.
# - Similar to `VectorArrays`, any concrete operator class needs to inherit from `pymor.operators.interface.Operator`.

# + slideshow={"slide_type": "fragment"}
import scipy.sparse as sps

A = sps.diags([-2, 1, 1], [0, -1, 1], shape=(10, 10))

A.toarray()

# + slideshow={"slide_type": "subslide"}
from pymor.operators.numpy import NumpyMatrixOperator

Aop = NumpyMatrixOperator(A)

print(Aop.range)
print(Aop.source)
print(Aop)
print(repr(Aop))

# + [markdown] slideshow={"slide_type": "subslide"}
# We can now create `VectorArrays` directly from the `VectorSpace` given by `Operator.source` and apply the `Operator`.

# + slideshow={"slide_type": "fragment"}
V = Aop.source.ones(1)
W = Aop.source.zeros(1)
V.append(W)

print(V)
Aop.apply(V)

# + [markdown] slideshow={"slide_type": "subslide"}
# `Operators` also have an `apply_inverse` method.

# + slideshow={"slide_type": "fragment"}
Aop.apply_inverse(V)

# + [markdown] slideshow={"slide_type": "subslide"}
# Here, `Aop` is a `NumpyMatrixOperator` which can handle both dense NumPy arrays and sparse SciPy matrices.
#
# The `apply_inverse` method chooses the appropriate solution approach for the corresponding data. E.g.,
#
# - dense solver for dense NumPy arrays.
# - SciPy sparse solvers for sparse SciPy matrices.

# + [markdown] slideshow={"slide_type": "subslide"}
# The easiest way to specify different solvers is via pyMORs `set_defaults` method.

# + slideshow={"slide_type": "fragment"}
from pymor.core.defaults import set_defaults

set_defaults({'pymor.bindings.scipy.apply_inverse.default_solver': 'scipy_bicgstab_spilu'})

# + [markdown] slideshow={"slide_type": "subslide"}
# Now calling `NumpyMatrixOperator.apply_inverse` with a sparse SciPy matrix will use the specified solver.
#
# Under the hood this is realized via the `@defaults` decorator.

# + slideshow={"slide_type": "fragment"}
from pymor.core.defaults import defaults

@defaults('tolerance')
def some_algorithm(x, y, tolerance=1e-5):
    print(tolerance)

def test_some_algorithm(x, y, tolerance_for_some_algorithm=None):
    some_algorithm(x, y, tolerance=tolerance_for_some_algorithm)


# + slideshow={"slide_type": "fragment"}
test_some_algorithm(1, 2)

# + [markdown] slideshow={"slide_type": "fragment"}
# Another option is using the `solver_options` parameter when initializing the `Operator`, but that specifies the solver only for the single `Operator`.

# + [markdown] slideshow={"slide_type": "subslide"}
# Generally, `Operators` might not have a specific `apply_inverse` method provided by the PDE backend.
#
# Or the `Operator` is non-linear in which case one typically resorts to iterative methods.
#
# In this case, pyMOR has its own generic iterative solver which only requires the `apply` method to be implemented.

# + slideshow={"slide_type": "fragment"}
from pymor.core.defaults import *

set_defaults({
    'pymor.operators.numpy.NumpyMatrixOperator.apply_inverse.default_sparse_solver_backend': 'generic'
})

Aop.apply_inverse(V)

# + [markdown] slideshow={"slide_type": "subslide"}
# `Operators` also support various other methods.

# + slideshow={"slide_type": "fragment"}
# Evaluate W.T @ A @ V
print(Aop.apply2(W, V))

# + slideshow={"slide_type": "fragment"}
print(Aop.apply_adjoint(V))

# + slideshow={"slide_type": "fragment"}
print(Aop.jacobian(V))

# + [markdown] slideshow={"slide_type": "subslide"}
# ### `LincombOperator`
#
# In MOR algorithms we may be interested to perform arithmetic operations with `Operators` which themselves are ***linear combinations*** of other `Operators`.
#
# Unlike for `VectorArrays`, adding, subtracting or scaling is not performed immediately but rather a `LincombOperator` is created.

# + slideshow={"slide_type": "fragment"}
Bop = NumpyMatrixOperator(np.eye(10))
Lop = 3*Aop - 2*Bop
Lop

# + slideshow={"slide_type": "fragment"}
Lop.apply(V)

# + slideshow={"slide_type": "fragment"}
Lop.assemble()

# + [markdown] slideshow={"slide_type": "subslide"}
# Aside from the `LincombOperator`, there are many different types of operators in `pymor.operators.constructions`.

# + slideshow={"slide_type": "subslide"}
from pymor.operators.constructions import ZeroOperator, IdentityOperator, ConcatenationOperator

Zop = ZeroOperator(Aop.range, Aop.source)
Iop = IdentityOperator(Aop.range, Aop.source)
Cop = ConcatenationOperator((Aop, Bop))

LLop = Zop + Iop + Lop
LLop

# + slideshow={"slide_type": "fragment"}
LLop.apply(V)

# + slideshow={"slide_type": "fragment"}
LLop.assemble()

# + [markdown] slideshow={"slide_type": "subslide"}
# How does the `LincombOperator` know how to assemble the linear combination?

# + slideshow={"slide_type": "fragment"}
from pymor.algorithms.lincomb import assemble_lincomb

# assemble_lincomb?

# + slideshow={"slide_type": "fragment"}
from pymor.algorithms.lincomb import AssembleLincombRules

AssembleLincombRules

# + [markdown] slideshow={"slide_type": "subslide"}
# #### `projection` of `Operators`
#
# The most important method of most model reduction algorithms is projection.

# + slideshow={"slide_type": "fragment"}
from pymor.algorithms.projection import project

V = LLop.source.random(3)
W = LLop.range.random(3)

projected_LLop = project(LLop, W, V)
projected_LLop

# + [markdown] slideshow={"slide_type": "fragment"}
# Similar as for the `assemble` method of the `LincombOperator` we have a `RuleTable` that handles projection in a smart way.

# + [markdown] slideshow={"slide_type": "subslide"}
# pyMORs `Operators` can be implemented to wrap matrices/operator in a desired PDE backend,
#
# but also enable efficient and convenient handling of certain structured operators `BlockOperator`, `CanonicalSymplecticFormOperator`, `NumpyHankelOperator`, ...

# + [markdown] slideshow={"slide_type": "subslide"}
# Note that `Operators` in pyMOR are immutable.
#
# Unlike for `VectorArrays`, we cannot make internal changes to an `Operator`.

# + slideshow={"slide_type": "fragment"}
# The code below will result in an error
# Aop.matrix = np.zeros((3, 3))

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Exercise
#
# Convert an `Operator` to a matrix using `pymor.algorithms.to_matrix.to_matrix`.

# + slideshow={"slide_type": "-"}
from pymor.algorithms.to_matrix import to_matrix

# your code here

# + [markdown] slideshow={"slide_type": "slide"}
# ## `Models`
#
# - `Models` are collections of `Operators` and `VectorArrays` that represent a particular type of model (equation system).
# - `pymor.models.interface.Model` defines the `solve` and `output` methods for the parameter-to-solution and parameter-to-output mapping, respectively.
# - Model order reduction algorithms typically take `Models` as an input and compute a ROM which will again be a `Model`.

# + slideshow={"slide_type": "subslide"}
from pymor.basic import *

p = StationaryProblem(
    domain=RectDomain([[0., 0.], [1., 1.]], left='robin', right='robin', top='robin', bottom='robin'),
    diffusion=ConstantFunction(1., 2),
    robin_data=(ConstantFunction(1., 2), ExpressionFunction('(x[0] < 1e-10) * 1.', 2)),
    outputs=[('l2_boundary', ExpressionFunction('(x[0] > (1 - 1e-10)) * 1.', 2))]
)

fom, _ = discretize_stationary_cg(p, diameter=0.1)

# + slideshow={"slide_type": "subslide"}
fom

# + slideshow={"slide_type": "fragment"}
fom.rhs

# + slideshow={"slide_type": "subslide"}
V = fom.solve()

# + slideshow={"slide_type": "fragment"}
fom.visualize(V)

# + slideshow={"slide_type": "subslide"}
from pymor.models.transfer_function import TransferFunction

tf = TransferFunction(1, 1, lambda s: 1 / (s + 1))

tf.eval_tf(1j)

# + [markdown] slideshow={"slide_type": "subslide"}
# In pyMOR `Models` are immutable objects.
#
# The `with_` method creates a shallow copy of the given immutable object (`Operators`,`Models`) and allows replacing attributes (`__init__` arguments) along the way.

# + slideshow={"slide_type": "fragment"}
# The code below will result in error
# fom.name = 'SimpleProblem'

fom.with_(name='SimpleProblem')

# + [markdown] slideshow={"slide_type": "subslide"}
# Typically, evaluating large-scale models is very expensive.
#
# In order to prevent expensive computations to be performed multiple times, pyMOR has its own caching mechanism.
#
# The easiest way to enable caching in pyMOR is to use the `@cached` decorator on a method of a model which inherits from `CacheableObject`.

# + slideshow={"slide_type": "fragment"}
from pymor.core.cache import CacheableObject, cached


# + slideshow={"slide_type": "fragment"}
class CustomModel(CacheableObject):

    cache_region = 'memory'

    @cached
    def expensive_evaluation(self, x):
        ...


# + [markdown] slideshow={"slide_type": "fragment"}
# Aside from `'memory'` one can also use `'disk'` or `'persistent'` as cache regions or configure a custom cache region.

# + [markdown] slideshow={"slide_type": "slide"}
# ## Parameters and Parametric Objects
#
# `Operators` and `Models` can generally be parametrized by one or more parameters in pyMOR.

# + [markdown] slideshow={"slide_type": "subslide"}
# ### `Parameters`
#
# - Dictionary of parameter names with corresponding dimensions.
# - Defines what parameters a parametric object depends on.
# - Names `'t'`, `'s'`, and `'input'` should only be used for time, frequency, and input, respectively.

# + slideshow={"slide_type": "fragment"}
from pymor.parameters.base import Parameters

parameters = Parameters({'a': 1, 'b': 2})
parameters

# + [markdown] slideshow={"slide_type": "fragment"}
# `Parameters` are immutable.

# + slideshow={"slide_type": "-"}
parameters['c'] = 3

# + [markdown] slideshow={"slide_type": "subslide"}
# ### `Mu`
#
# - Dictionary of parameters names with corresponding values as 1D NumPy arrays.

# + slideshow={"slide_type": "fragment"}
from pymor.parameters.base import Mu

mu = Mu({'a': np.array([1]), 'b': np.array([2, 3]), 'c': np.array([4, 5, 6])})
mu

# + [markdown] slideshow={"slide_type": "fragment"}
# It is possible to check if parameter values belong to a `Parameter`.

# + slideshow={"slide_type": "-"}
parameters.is_compatible(mu)

# + [markdown] slideshow={"slide_type": "subslide"}
# It is possible to avoid constructing `Mu` by hand if `Parameters` is available via its `parse` method.

# + slideshow={"slide_type": "-"}
parameters.parse([1, 2, 3])

# + [markdown] slideshow={"slide_type": "subslide"}
# ### `ParameterSpace`
#
# - Represents the set of possible parameter values.
# - Restricted to be of cubic type (each parameter component has a minimum and maximum value).

# + slideshow={"slide_type": "fragment"}
from pymor.parameters.base import ParameterSpace

print(ParameterSpace(parameters, (0, 1)))
print(ParameterSpace(parameters, {'a': (0, 1), 'b': (1, 2)}))

# + [markdown] slideshow={"slide_type": "fragment"}
# A `ParameterSpace` can also be created using the `space` method of `Parameters`.

# + slideshow={"slide_type": "-"}
print(parameters.space(0, 1))
print(parameters.space({'a': (0, 1), 'b': (1, 2)}))

# + [markdown] slideshow={"slide_type": "subslide"}
# `ParameterSpaces` are useful for sampling.

# + slideshow={"slide_type": "-"}
parameter_space = parameters.space(0, 1)
parameter_space.sample_uniformly(3)

# + slideshow={"slide_type": "fragment"}
parameter_space.sample_randomly(10)

# + [markdown] slideshow={"slide_type": "subslide"}
# ### `ParameterFunctional`
#
# - Represent mappings from parameter values to $\mathbb{R}$ or $\mathbb{C}$.
# -

parameters

# + slideshow={"slide_type": "fragment"}
from pymor.parameters.functionals import ProjectionParameterFunctional
 
f1 = ProjectionParameterFunctional('b', size=2, index=0)
f1.evaluate(mu=parameters.parse([1, 2, 3]))

# + slideshow={"slide_type": "fragment"}
from pymor.parameters.functionals import GenericParameterFunctional

f2 = GenericParameterFunctional(lambda mu: mu['b'][0]**2, parameters)
f2.evaluate(mu=parameters.parse([1, 2, 3]))

# + slideshow={"slide_type": "fragment"}
from pymor.parameters.functionals import ExpressionParameterFunctional

f3 = ExpressionParameterFunctional('b[0]**2', parameters)
f3.evaluate(mu=parameters.parse([1, 2, 3]))

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Time-Dependent Parameter Values

# + slideshow={"slide_type": "-"}
mu = Parameters(f=1, t=1).parse(['sin(t)', 0])
mu

# + slideshow={"slide_type": "fragment"}
mu['f']

# + slideshow={"slide_type": "fragment"}
mu2 = mu.with_(t=np.pi/4)
mu2

# + slideshow={"slide_type": "fragment"}
mu2['f']

# + [markdown] slideshow={"slide_type": "slide"}
# ### Parametric `Operators`

# + [markdown] slideshow={"slide_type": "fragment"}
# The simplest case is a parameter-separable `Operator`.

# + slideshow={"slide_type": "-"}
Pop = Aop + ProjectionParameterFunctional('p') * Bop
Pop

# + [markdown] slideshow={"slide_type": "subslide"}
# It is possible to check if an `Operator` (or any object inheriting from `ParametricObject`) is parametric via its `parametric` attribute.

# + slideshow={"slide_type": "-"}
Pop.parametric

# + [markdown] slideshow={"slide_type": "fragment"}
# The `Parameters` of the `Operator` are automatically deduced.

# + slideshow={"slide_type": "-"}
Pop.parameters

# + [markdown] slideshow={"slide_type": "subslide"}
# We can `apply` the `Operator` on a `VectorArray` for different parameter values.

# + slideshow={"slide_type": "-"}
V = Pop.source.ones(1)
Pop.apply(V, mu=Pop.parameters.parse(0))

# + slideshow={"slide_type": "-"}
Pop.apply(V, mu=Pop.parameters.parse(1))

# + [markdown] slideshow={"slide_type": "fragment"}
# Note that it is necessary to pass a `Mu` object.

# + slideshow={"slide_type": "-"}
Pop.apply(V, mu=1)

# + [markdown] slideshow={"slide_type": "subslide"}
# `VectorArrays` cannot be parametric.
# A way to represent parameter-dependent vectors is via a parametric `Operator`,
# e.g., using `VectorArrayOperators`.

# + slideshow={"slide_type": "fragment"}
from pymor.operators.constructions import VectorArrayOperator

space = NumpyVectorSpace(10)

U = space.ones()
print(U)
V = space.full(2.)
print(V)

# + slideshow={"slide_type": "fragment"}
Uop = VectorArrayOperator(U)
print(Uop)
Vop = VectorArrayOperator(V)
print(Vop)

# + slideshow={"slide_type": "fragment"}
UVop = Uop + ProjectionParameterFunctional('p') * Vop
print(UVop)

# + slideshow={"slide_type": "fragment"}
UVop.as_vector(mu=UVop.parameters.parse(5))

# + [markdown] slideshow={"slide_type": "slide"}
# ## Parametric Models
#
# If a `Model` is based on any parametric `Operators`, the `Model` will be parametric as well.

# + slideshow={"slide_type": "fragment"}
A = np.random.rand(10, 10)
B = np.ones((10, 1))
C = np.ones((1, 10))
fom = LTIModel.from_matrices(A, B, C)

fom

# + slideshow={"slide_type": "fragment"}
fom.parameters

# + slideshow={"slide_type": "subslide"}
Pop = fom.A + ProjectionParameterFunctional('p') * fom.A
fom = fom.with_(A=Pop)

# + slideshow={"slide_type": "fragment"}
fom

# + slideshow={"slide_type": "fragment"}
fom.parameters

# + [markdown] slideshow={"slide_type": "slide"}
# ## Conclusion
#
# - Fundamental interfaces:
#   - `VectorArray`
#   - `Operator`
#   - `Model`
# - Extensively used features:
#   - defaults
#   - caching
#   - parameters
#   - rule tables
