<p align="center">
  <a href="https://github.com/nschloe/orthopy"><img alt="orthopy" src="https://nschloe.github.io/orthopy/orthopy-logo-with-text.png" width="30%"></a>
  <p align="center">All about orthogonal polynomials.</p>
</p>

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/orthopy/ci?style=flat-square)](https://github.com/nschloe/orthopy/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/orthopy.svg?style=flat-square)](https://codecov.io/gh/nschloe/orthopy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)
[![orthogonal](https://img.shields.io/badge/orthogonal-definitely-ff69b4.svg?style=flat-square)](https://github.com/nschloe/orthopy)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/orthopy.svg?style=flat-square)](https://pypi.org/pypi/orthopy/)
[![PyPi Version](https://img.shields.io/pypi/v/orthopy.svg?style=flat-square)](https://pypi.org/project/orthopy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1173151.svg?style=flat-square)](https://doi.org/10.5281/zenodo.1173151)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/orthopy.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/orthopy)
[![PyPi downloads](https://img.shields.io/pypi/dm/orthopy.svg?style=flat-square)](https://pypistats.org/packages/orthopy)

orthopy provides various orthogonal polynomial classes for
[lines](#line-segment--1-1-with-weight-function-1-x%CE%B1-1-x%CE%B2),
[triangles](#triangle-42),
[disks](#disk-s2),
[spheres](#sphere-u2),
[n-cubes](#n-cube-cn),
[nD space with weight function exp(-r<sup>2</sup>)](#nd-space-with-weight-function-exp-r2-enr2)
All computations are done using numerically stable recurrence schemes.
Furthermore, all functions are fully vectorized and can return results in [_exact
arithmetic_](#symbolic-and-numerical-computation).

### Basic usage

The main function of all submodules is the iterator `Eval` which evaluates the series of
orthogonal polynomials with increasing degree, e.g.,
```python
import orthopy

x = 0.5

for k, val in enumerate(orthopy.c1.legendre.Eval(x, "classical")):
     print(val)
     if k == 4:
         break
```
```python
1.0          # P_0(0.5)
0.5          # P_1(0.5)
-0.125       # P_2(0.5)
-0.4375      # P_3(0.5)
-0.2890625   # P_4(0.5)
```
Another way of getting the first `n` items is to use
[itertools](https://docs.python.org/3/library/itertools.html):
```python
import itertools

vals = list(itertools.islice(Eval(x, "normal"), n + 1))
```
Instead of evaluating at only one point, you can provide an array of arbitrary shape for
`x`. The polynomials will then be evaluated for all points at once. The computation are
fully be vectorized and usually very fast.  You can also use sympy for symbolic
computation; make sure to pass a variable and set
`symbolic=True`:
```python
import itertools
import orthopy
import sympy

x = sympy.Symbol("x")

evaluator = orthopy.c1.legendre.Eval(x, "classical", symbolic=True)
for level in itertools.islice(evaluator, 5):
     print(sympy.expand(level))
```
```
1
x
3*x**2/2 - 1/2
5*x**3/2 - 3*x/2
35*x**4/8 - 15*x**2/4 + 3/8
```
(Set `x` to `0.5` to get the values above.)

All `Eval` methods have a `scaling` argument which can be set to three values:

  * `"monic"`: The leading coefficient is 1.
  * `"classical"`: The maximum value is 1 (or  (n+alpha over n)).
  * `"normal"`: The integral of the squared function over the domain is 1.

For univariate ("one-dimensional") integrals, every level contains one functions. For
bivariate ("two-dimensional") functions, every level will contain one functions more
than the previous.


See the trees for triangles and disks below.


### Line segment [-1, +1] with weight function (1-x)<sup>α</sup> (1-x)<sup>β</sup>

<img src="https://nschloe.github.io/orthopy/legendre.svg" width="100%"> | <img src="https://nschloe.github.io/orthopy/chebyshev1.svg" width="100%"> | <img src="https://nschloe.github.io/orthopy/chebyshev2.svg" width="100%">
:-------------------:|:------------------:|:-------------:|
Legendre             |  Chebyshev 1       |  Chebyshev 2  |

Jacobi, Gegenbauer (α=β), Chebyshev 1 (α=β=-1/2), Chebyshev 2 (α=β=1/2), Legendre
(α=β=0) polynomials.

```python
orthopy.c1.legendre.Eval(x, "normal", symbolic=False)
orthopy.c1.chebyshev1.Eval(x, "normal", symbolic=False)
orthopy.c1.chebyshev2.tree(x, "normal", symbolic=False)
orthopy.c1.gegenbauer.tree(x, lmbda, "normal", symbolic=False)
orthopy.c1.jacobi.tree(x, alpha, beta, "normal", symbolic=False)
```

Recurrence coefficients can be explicitly retrieved by
```python
p0, a, b, c = orthopy.c1.jacobi.recurrence_coefficients(n, a, b, "monic")
```


#### Associated Legendre "polynomials"

<img src="https://nschloe.github.io/orthopy/associated-legendre.svg" width="45%">

```python
vals = orthopy.c1.associated_legendre.tree(
    x, 4, phi=None, standardization="natural", with_condon_shortley_phase=True,
    symbolic=False
    )
```

### 1D half-space with weight function x<sup>α</sup> exp(-r)
<img src="https://nschloe.github.io/orthopy/e1r.svg" width="45%">

(Generalized) Laguerre polynomials.
```python
vals = orthopy.e1r.tree(x, 4, alpha=0, scaling="normal", symbolic=False)
```


### 1D space with weight function exp(-r<sup>2</sup>)
<img src="https://nschloe.github.io/orthopy/e1r2.svg" width="45%">

Hermite polynomials.
```python
vals = orthopy.e1r2.tree(x, 4, "normal", symbolic=False)
```
All polynomials are normalized over the measure.


### Triangle (_T<sub>2</sub>_)

<img src="https://nschloe.github.io/orthopy/triangle-tree.png" width="40%">

```python
for level in orthopy.t2.Eval(x, "normal", symbolic=False):
    # `level` contains all evalutations of the orthogonal polynomials with the next
    # degree at the points x
    pass

# or for the entire tree up to degree 4
vals = orthopy.t2.tree(x, 4, "normal", symbolic=False)
```
Available scalings are
  * `"normal"` (normalized polynomials, i.e., the integral of the squared function equals 1) and
  * `"1"` where the polynomial is `1` in at least one corner of the triangle.


### Disk (_S<sub>2</sub>_)

<img src="https://nschloe.github.io/orthopy/disk-yu-tree.png" width="70%"> | <img src="https://nschloe.github.io/orthopy/disk-zernike-tree.png" width="70%"> | <img src="https://nschloe.github.io/orthopy/disk-zernike2-tree.png" width="70%">
:------------:|:-----------------:|:-----------:|
Yu            |  [Zernike](https://en.wikipedia.org/wiki/Zernike_polynomials)          |  Zernike 2  |

```python
for level in orthopy.s2.Eval(x, symbolic=False):
    # `level` contains all evalutations of the orthogonal polynomials with the next
    # degree at the points x
    pass

# or for the entire tree up to degree 4
vals = orthopy.s2.tree(4, x, symbolic=False)
```
All polynomials are normalized on the unit disk.


### Sphere (_U<sub>3</sub>_)


<img src="https://nschloe.github.io/orthopy/sph-tree.png" width="50%">

Complex-valued _spherical harmonics,_ plotted with
[cplot](https://github.com/nschloe/cplot/) coloring.

```python
for level in orthopy.u3.Eval(x, scaling="quantum mechanic", symbolic=False):
    # `level` contains all evalutations of the spherical harmonics with the next
    # degree at the points x
    pass

# or for the entire tree up to degree n
vals = orthopy.u3.tree(
    polar, azimuthal, n, standardization="quantum mechanic", symbolic=False
)
```

### n-Cube (_C<sub>n</sub>_)

```python
vals = orthopy.cn.tree(6, x, symbolic=False)
```
All polynomials are normalized on the n-dimensional cube. The dimensionality is
determined by `X.shape[0]`.

### nD space with weight function exp(-r<sup>2</sup>) (_E<sub>n</sub><sup>r<sup>2</sup></sup>_)

```python
vals = orthopy.enr2.tree(4, x, symbolic=False)
```
All polynomials are normalized over the measure. The dimensionality is
determined by `X.shape[0]`.


### Other tools

 * [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for
   computing the weighted sum of orthogonal polynomials:
   ```python
   vals = orthopy.c1.clenshaw(a, alpha, beta, t)
   ```


### Installation

orthopy is [available from the Python Package
Index](https://pypi.python.org/pypi/orthopy/), so use
```
pip install orthopy
```
to install.

### Testing

To run the tests, simply check out this repository and run
```
pytest
```

### Relevant publications

* [Robert C. Kirby, Singularity-free evaluation of collapsed-coordinate orthogonal polynomials, ACM Transactions on Mathematical Software (TOMS), Volume 37, Issue 1, January 2010](https://doi.org/10.1145/1644001.1644006)
* [Abedallah Rababah, Recurrence Relations for Orthogonal Polynomials on Triangular Domains, MDPI Mathematics 2016, 4(2)](https://doi.org/10.3390/math4020025)
* [Yuan Xu, Orthogonal polynomials of several variables, archiv.org, January 2017](https://arxiv.org/abs/1701.02709)

### License
This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
