# orthopy

Python tools for orthogonal polynomials and Gaussian quadrature.

![](https://nschloe.github.io/orthopy/orthopy.png)

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/orthopy/master.svg)](https://circleci.com/gh/nschloe/orthopy/tree/master)
[![codecov](https://codecov.io/gh/nschloe/orthopy/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/orthopy)
[![PyPi Version](https://img.shields.io/pypi/v/orthopy.svg)](https://pypi.python.org/pypi/orthopy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/orthopy.svg?style=social&label=Stars&maxAge=2592000)](https://github.com/nschloe/orthopy)

Gaussian quadrature schemes and orthogonal polynomials of the form
```
pi_{k+1}(x) = (x - alpha[k]) * pi_k(x) - beta[k] * pi_{k-1}(x)
```
(defined by their _recurrence coefficients_ `alpha` and `beta`) are closely
related. This module provides tools for working with them.

_Note that most function have a `sympy` and `mpmath` mode for symbolic and
arbitrary precision computation, respectively._

Some examples:

#### Transform between a Gaussian schemes and recurrence coefficients
```python
import orthopy

# alpha = ...
# beta = ...
points, weights = orthopy.gauss_from_coefficients(alpha, beta)
alpha, beta = orthopy.coefficients_from_gauss(points, weights)
```

#### Recurrence coefficients of classical weight functions

The recurrence coefficients of Gauss rules for the weight function
`w(x) = (1-x)^a * (1+x)^b` with any `a` or `b` are explicitly known. Retrieve
them with
```python
import orthopy

alpha, beta = orthopy.jacobi_recurrence_coefficients(n, a, b)
```
Of course, it's easy to generate the corresponding Gaussian rule; for example
for Gauss-Legendre of order 5:
```python
import orthopy

alpha, beta = orthopy.jacobi_recurrence_coefficients(5, 0.0, 0.0)
points, weights = orthopy.gauss_from_coefficients(alpha, beta)
```

#### Recurrence coefficients for your own weight function

A couple of algorithms are implemented for that, particularly

  * Stieltjes, and
  * Golub-Welsch,
  * (modified) Chebyshev.

The method `stieltjes` does symbolic computation, its input arguments are the
weight function, the integration limits, and the desired number of
coefficients. For example,
```python
alpha, beta = orthopy.stieltjes(lambda t: 1, -1, +1, 5)
```
will recover the Legendre coefficients.

The input `golub_welsch` and `chebyshev{_modified}` is an array of _moments_,
i.e., the integrals
```
integral(w(x) p_k(x) dx)
```
with `p_k(x)` either the monomials `x^k` (for Golub-Welsch and Chebyshev), or
a known set of orthogonal polynomials (for modified Chebyshev). Depending on
your weight function, the moments may have an analytic representation.

```python
import orthopy

# Modified moments of `int x^2 p_k(x) dx` with Legendre polynomials.
# Almost all moments are 0.
n = 5
moments = numpy.zeros(2*n)
moments[0] = 2.0/3.0
moments[2] = 8.0/45.0

a, b = orthopy.jacobi_recurrence_coefficients(2*n, 0.0, 0.0)
alpha, beta = orthopy.chebyshev_modified(moments, a, b)
```

Be aware of the fact that Golub-Welsch and unmodified Chebyshev are _very_
ill-conditioned, so don't push `n` too far! In any case, [as recommended by
Gautschi](https://doi.org/10.1007/BF02218441), you can test your
moment-based scheme with
```python
import orthopy
orthopy.check_coefficients(moments, alpha, beta)
```

#### Other tools

* [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for
  computing the weighted sum of orthogonal polynomials:
  ```python
  import orthopy
  vals = orthopy.clenshaw(a, alpha, beta, t)
  ```
* Evaluate orthogonal polynomials (at many points at once):
  ```python
  import orthopy
  vals = orthopy.evaluate_orthogonal_polynomial(alpha, beta, t)
  ```

### Relevant publications

 * [Gene H. Golub and John H. Welsch, Calculation of Gauss Quadrature Rules, Mathematics of Computation, Vol. 23, No. 106 (Apr., 1969), pp. 221-230+s1-s10](https://dx.doi.org/10.2307/2004418)
 * [W. Gautschi, How and how not to check Gaussian quadrature formulae, BIT Numerical Mathematics, June 1983, Volume 23, Issue 2, pp 209–216](https://doi.org/10.1007/BF02218441)
 * [D. Boley and G.H. Golub, A survey of matrix inverse eigenvalue problems, Inverse Problems, 1987, Volume 3, Number 4](https://doi.org/10.1088/0266-5611/3/4/010)
 * [W. Gautschi, Algorithm 726: ORTHPOL–a package of routines for generating orthogonal polynomials and Gauss-type quadrature rules, ACM Transactions on Mathematical Software (TOMS), Volume 20, Issue 1, March 1994, Pages 21-62](http://doi.org/10.1145/174603.174605)


### Installation

orthopy is [available from the Python Package Index](https://pypi.python.org/pypi/orthopy/), so with
```
pip install -U orthopy
```
you can install/upgrade.

### Testing

To run the tests, simply check out this repository and run
```
pytest
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. publish to PyPi and GitHub:
    ```
    $ make publish
    ```

### License
orthopy is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
