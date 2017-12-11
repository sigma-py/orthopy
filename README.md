# orthopy

Python tools for orthogonal polynomials and Gaussian quadrature for
[lines](#line-segment), [triangles](#triangle), and [spheres](#sphere).

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/orthopy/master.svg)](https://circleci.com/gh/nschloe/orthopy/tree/master)
[![codecov](https://codecov.io/gh/nschloe/orthopy/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/orthopy)
[![awesome](https://img.shields.io/badge/awesome-yes-brightgreen.svg)](https://img.shields.io/badge/awesome-yes-brightgreen.svg)
[![PyPi Version](https://img.shields.io/pypi/v/orthopy.svg)](https://pypi.python.org/pypi/orthopy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/orthopy.svg?style=social&label=Stars&maxAge=2592000)](https://github.com/nschloe/orthopy)

_All functions in this module are fully vectorized and, where possible and
practical, return results in exact arithmetic._

### Line segment

![](https://nschloe.github.io/orthopy/line-segment.png)

Gaussian quadrature schemes and orthogonal polynomials of the form
```
pi_{k+1}(x) = (a[k] x - b[k]) * pi_k(x) - c[k] * pi_{k-1}(x)
```
(defined by their _recurrence coefficients_ `a`, `b`, `c`) are closely
related. orthopy provides tools for working with them.

#### Classical schemes

With orthopy, it's easy to regenerate classical Gauss quadrature schemes are
listed in, e.g., [Stroud & Secrest](https://books.google.de/books/about/Gaussian_quadrature_formulas.html?id=X7M-AAAAIAAJ).

Some examples:
```python
points, weights = orthopy.line.schemes.legendre(96, decimal_places=30)
points, weights = orthopy.line.schemes.hermite(14, decimal_places=20)
points, weights = orthopy.line.schemes.laguerre(13, decimal_places=50)
```

#### Generating your own Gauss quadrature in three simple steps

You have a measure (or, more colloquially speaking, a domain and a nonnegative
weight function) and would like to generate the matching Gauss quadrature?
Great, here's how to do it.

As an example, let's try and generate the Gauss quadrature with 10 points for
the weight function `x^2` on the interval `[-1, +1]`.

TLDR:
```python
import orthopy
moments = orthopy.line.compute_moments(lambda x: x**2, -1, +1, 20)
alpha, beta = orthopy.line.chebyshev(moments)
points, weights = orthopy.line.schemes.custom(alpha, beta, decimal_places=30)
```

Some explanations:

  1. You need to compute the first `2*n` _moments_ of your measure
     ```
     integral(w(x) p_k(x) dx)
     ```
     with a particular set of polynomials `p_k`. A common choice are the
     monomials `x^k`. You can do that by hand or use
     ```python
     moments = orthopy.line.compute_moments(lambda x: x**2, -1, +1, 20)
     ```
     ```
     [2/3, 0, 2/5, 0, 2/7, 0, 2/9, 0, 2/11, 0, 2/13, 0, 2/15, 0, 2/17, 0, 2/19, 0, 2/21, 0]
     ```
     Note that the moments have all been computed symbolically here.

     If you have the moments in floating point (for example because you need to
     compute the scheme fast), it makes sense to think about the numerical
     implications here. That's because the map to the recurrence coefficients
     (step 2) can be _very_ ill-conditioned, meaning that small round-off
     errors can lead to an unusable scheme.
     For further computation, it's numerically beneficial if the moments are
     either 0 or in the same order of magnitude. The above numbers are alright,
     but if you want to max it out, you could try Legendre polynomials for
     `p_k`:
     ```python
     moments = orthopy.line.compute_moments(
         lambda x: x**2, -1, +1, 20,
         polynomial_class=orthopy.line.legendre
         )
     ```
     ```
     [2/3, 0, 8/45, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
     ```
     Better!

  2. From the moments, we generate the recurrence coefficients of our custom
     orthogonal polynomials. There are a few choices to accomplish this:

       * `golub_welsch`: uses Cholesky at its core; can be numerically unstable
       * `stieltjes`: moments not even needed here, but can also be numerically
         unstable
       * `chebyshev`: can be used if you chose monomials in the first step;
         again, potentially numerically unstable
       * `chebyshev_modified`: to be used if you chose something other than
         monomials in the first step; stable if the `polynomial_class` was
         chosen wisely

       Since we have computed modified moments in step one, let's use the
       latter method:
       ```python
       _, _, a, b = orthopy.line.recurrence_coefficients.legendre(20, 'monic')
       alpha, beta = orthopy.line.chebyshev_modified(moments, a, b)
       ```
       ```
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
       [2/3, 3/5, 4/35, 25/63, 16/99, 49/143, 12/65, 27/85, 64/323, 121/399]
       ```
       (Note that, since everything is done symbolically in this example,
       we could have used Stieltjes's or Chebyshev's unmodified method; the
       results are the same.)

  3. Lastly, we generate the Gauss points and weights from `alpha` and `beta`.
     Since symbolic computation can take _very_ long even for small sizes, we
     choose the `mpmath` mode (default) with 30 decimal digits
     ```python
     points, weights = \
         orthopy.line.schemes.custom(alpha, beta, mode='mpmath', decimal_places=30)
     ```
     ```
     [-0.978228658146056992803938001123,
      -0.887062599768095299075157769304,
      -0.730152005574049324093416252031,
      -0.519096129206811815925725669458,
      -0.269543155952344972331531985401,
      0.2695431559523449723315319854,
      0.519096129206811815925725669458,
      0.730152005574049324093416252031,
      0.887062599768095299075157769304,
      0.978228658146056992803938001123]
     ```
     ```
       [0.0532709947237135572432759986252,
        0.0988166881454075626728761840589,
        0.0993154007474139787312043384226,
        0.0628365763465911675266984722740,
        0.0190936733702070671592783399524,
        0.0190936733702070671592783399524,
        0.0628365763465911675266984722744,
        0.0993154007474139787312043384225,
        0.0988166881454075626728761840592,
        0.0532709947237135572432759986251]
     ```
     Congratulations! Your Gaussian quadrature rule.


#### Other tools

 * Transforming Gaussian points and weights back to recurrence coefficients:
   ```python
   alpha, beta = orthopy.line.coefficients_from_gauss(points, weights)
   ```

 * Recurrence coefficients of Jacobi polynomials
   `w(x) = (1-x)^alpha * (1+x)^beta` with any `alpha` or `beta` are explicitly
   given:
   ```python
   p0, a, b, c = orthopy.line.recurrence_coefficients.jacobi(n, a, b, 'monic')
   ```
   Possible choices for the standardization are `'monic'`,
   `'p(1)=(n+alpha over n)'`, and `'||p**2||=1`.

 * The Gautschi test: [As recommended by
   Gautschi](https://doi.org/10.1007/BF02218441), you can test your
   moment-based scheme with
   ```python
   err = orthopy.line.check_coefficients(moments, alpha, beta)
   ```
 * [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for
   computing the weighted sum of orthogonal polynomials:
   ```python
   vals = orthopy.line.clenshaw(a, alpha, beta, t)
   ```

 * Evaluate orthogonal polynomials (at many points at once):
   ```python
   vals = orthopy.line.evaluate_orthogonal_polynomial(alpha, beta, t)
   ```

 * Evaluate the entire _associated Legendre_ tree up to a given level at once:
   ```python
   vals = orthopy.line.alp_tree(
       n, x,
       normalization='full',
       with_condon_shortley_phase=True
       )
   ```
   The implementation is numerically stable.

### Triangle

<img src="https://nschloe.github.io/orthopy/triangle.png" width="25%">

Just like in one dimension, orthogonal polynomials can be defined for any
domain and weight function. orthopy provides the means for
computing orthogonal/-normal polynomials for the triangle. The implementation
is recurrent and numerically stable.
```python
vals = orthopy.triangle.orth_tree(4, x, 'normal', symbolic=False)
```
Available standardizations are `'normal'` (normalized polynomials, i.e.,
`||p||=1`) and `'1'` where the polynomial is `1` in at least one corner of
the triangle.

If `symbolic=True` is specified, all computations are performed
symbolically. This can be used, for example, to get the classical
representations of the polynomials:
```python
import numpy
import orthopy
import sympy

b0, b1, b2 = sympy.Symbol('b0'), sympy.Symbol('b1'), sympy.Symbol('b2')

tree = orthopy.triangle.orth_tree(
        3, numpy.array([b0, b1, b2]), 'normal', symbolic=True
        )

print(sympy.expand(tree[3][1]))
```
```
42*sqrt(6)*b0*b2**2 - 24*sqrt(6)*b0*b2 + 2*sqrt(6)*b0 - 42*sqrt(6)*b1*b2**2 + 24*sqrt(6)*b1*b2 - 2*sqrt(6)*b1
```


### Sphere

Evaluate the entire _spherical harmonics_ tree up to a given level at once.
Again, the implementation is numerically stable.
```python
vals = orthopy.sphere.sph_tree(n, x, symbolic=False)
```


### Relevant publications

 * [A.H. Stroud and D. Secrest, Gaussian Quadrature Formulas, 1966, Prentice Hall, Series in Automatic Computation](https://books.google.de/books/about/Gaussian_quadrature_formulas.html?id=X7M-AAAAIAAJ)
 * [Gene H. Golub and John H. Welsch, Calculation of Gauss Quadrature Rules, Mathematics of Computation, Vol. 23, No. 106 (Apr., 1969), pp. 221-230+s1-s10](https://dx.doi.org/10.2307/2004418)
 * [W. Gautschi, On Generating Orthogonal Polynomials, SIAM J. Sci. and Stat. Comput., 3(3), 289–317](https://doi.org/10.1137/0903018)
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
