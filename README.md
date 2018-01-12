# orthopy

Your one-stop shop for orthogonal polynomials in Python.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/orthopy/master.svg)](https://circleci.com/gh/nschloe/orthopy/tree/master)
[![codecov](https://codecov.io/gh/nschloe/orthopy/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/orthopy)
[![awesome](https://img.shields.io/badge/awesome-yes-brightgreen.svg)](https://img.shields.io/badge/awesome-yes-brightgreen.svg)
[![PyPi Version](https://img.shields.io/pypi/v/orthopy.svg)](https://pypi.python.org/pypi/orthopy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/orthopy.svg?style=social&label=Stars)](https://github.com/nschloe/orthopy)

Various orthogonal polynomial classes for [lines](#line-segment),
[triangles](#triangle), [disks](#disk), and [spheres](#sphere). All
computations are done using recurrence schemes and are numerically stable.
Furthermore, all functions are fully vectorized and, where possible and
practical, can return results in _exact_ arithmetic.

### Line segment [-1, +1] with weight function (1-x)<sup>α</sup> (1-x)<sup>β</sup>

<img src="https://nschloe.github.io/orthopy/line-segment.png" width="25%">

Jacobi, Gegenbauer (α=β), Chebyshev 1 (α=β=-1/2), Chebyshev 2 (α=β=1/2),
Legendre (α=β=0) polynomials.

```python
vals = orthopy.line_segment.tree(4, x, alpha=0, standardization='normal', symbolic=False)
```

#### Associated Legendre polynomials

```python
vals = orthopy.line_segment.alp_tree(4, x, phi=None, standardization=None, with_condon_shortley_phase=True, symbolic=False)
```

### 1D half-space with weight function x<sup>α</sup> exp(-r)
<img src="https://nschloe.github.io/orthopy/e1r.png" width="25%">

(Generalized) Laguerre polynomials.
```python
vals = orthopy.e1r.tree(4, x, alpha=0, standardization='normal', symbolic=False)
```


### 1D space with weight function exp(-r<sup>2</sup>)
<img src="https://nschloe.github.io/orthopy/e1r2.png" width="25%">

Hermite polynomials.
```python
vals = orthopy.e1r2.tree(4, x, symbolic=False)
```
All polynomials are normalized over the measure.


### Triangle

<img src="https://nschloe.github.io/orthopy/triangle.png" width="25%">

Just like in one dimension, orthogonal polynomials can be defined for any
domain and weight function. orthopy provides the means for
computing orthogonal/-normal polynomials for the triangle. The implementation
is recurrent and numerically stable.
```python
vals = orthopy.triangle.tree(4, x, 'normal', symbolic=False)
```
Available standardizations are `'normal'` (normalized polynomials, i.e., the
integral of the squared function equals 1) and `'1'` where the polynomial is
`1` in at least one corner of the triangle.

If `symbolic=True` is specified, all computations are performed
symbolically. This can be used, for example, to get the classical
representations of the polynomials:
```python
import numpy
import orthopy
import sympy

b0, b1, b2 = sympy.Symbol('b0'), sympy.Symbol('b1'), sympy.Symbol('b2')

tree = orthopy.triangle.tree(
        3, numpy.array([b0, b1, b2]), 'normal', symbolic=True
        )

print(sympy.expand(tree[3][1]))
```
```
42*sqrt(6)*b0*b2**2 - 24*sqrt(6)*b0*b2 + 2*sqrt(6)*b0 - 42*sqrt(6)*b1*b2**2 + 24*sqrt(6)*b1*b2 - 2*sqrt(6)*b1
```

### Quadrilateral

<img src="https://nschloe.github.io/orthopy/quad.png" width="25%">

```python
vals = orthopy.quadrilateral.tree(4, x, symbolic=False)
```
All polynomials are normalized on the quadrilateral.


### Disk

<img src="https://nschloe.github.io/orthopy/disk.png" width="25%">

```python
vals = orthopy.disk.tree(4, x, symbolic=False)
```
All polynomials are normalized on the unit disk.


### 2D space with weight function exp(-r<sup>2</sup>)
<img src="https://nschloe.github.io/orthopy/e2r2.png" width="25%">

```python
vals = orthopy.e2r2.tree(4, x, symbolic=False)
```
All polynomials are normalized over the measure.


### Sphere

<img src="https://nschloe.github.io/orthopy/sphere.png" width="25%">

Evaluate the entire _spherical harmonics_ tree up to a given level at once.
Again, the implementation is numerically stable.
```python
vals = orthopy.sphere.sph_tree(n, x, symbolic=False)
```
Note that spherical harmonics are complex-valued in general. The above plot
only shows the absolute value of SPH(5, 3).


### Hexahedron

<img src="https://nschloe.github.io/orthopy/hexa.png" width="25%">

```python
vals = orthopy.hexahedron.tree(3, x, symbolic=False)
```
All polynomials are normalized on the hexahedron.


### n-Cube

```python
vals = orthopy.ncube.tree(6, x, symbolic=False)
```
All polynomials are normalized on the n-dimensional cube. The dimensionality is
determined by `X.shape[0]`.

### nD space with weight function exp(-r<sup>2</sup>)

```python
vals = orthopy.enr2.tree(4, x, symbolic=False)
```
All polynomials are normalized over the measure. The dimensionality is
determined by `X.shape[0]`.


### Relevant publications

 * [A.H. Stroud and D. Secrest, Gaussian Quadrature Formulas, 1966, Prentice Hall, Series in Automatic Computation](https://books.google.de/books/about/Gaussian_quadrature_formulas.html?id=X7M-AAAAIAAJ)
 * [Gene H. Golub and John H. Welsch, Calculation of Gauss Quadrature Rules, Mathematics of Computation, Vol. 23, No. 106 (Apr., 1969), pp. 221-230+s1-s10](https://dx.doi.org/10.2307/2004418)
 * [W. Gautschi, On Generating Orthogonal Polynomials, SIAM J. Sci. and Stat. Comput., 3(3), 289–317](https://doi.org/10.1137/0903018)
 * [W. Gautschi, How and how not to check Gaussian quadrature formulae, BIT Numerical Mathematics, June 1983, Volume 23, Issue 2, pp 209–216](https://doi.org/10.1007/BF02218441)
 * [D. Boley and G.H. Golub, A survey of matrix inverse eigenvalue problems, Inverse Problems, 1987, Volume 3, Number 4](https://doi.org/10.1088/0266-5611/3/4/010)
 * [W. Gautschi, Algorithm 726: ORTHPOL–a package of routines for generating orthogonal polynomials and Gauss-type quadrature rules, ACM Transactions on Mathematical Software (TOMS), Volume 20, Issue 1, March 1994, Pages 21-62](http://doi.org/10.1145/174603.174605)

### Other tools

 * Recurrence coefficients of Jacobi polynomials
   `w(x) = (1-x)^alpha * (1+x)^beta` with any `alpha` or `beta` are explicitly
   given:
   ```python
   p0, a, b, c = orthopy.line.recurrence_coefficients.jacobi(n, a, b, 'monic')
   ```
   Possible choices for the standardization are `'monic'`,
   `'p(1)=(n+alpha over n)'`, and `'normal`.

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
