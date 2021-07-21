<p align="center">
  <a href="https://github.com/nschloe/orthopy"><img alt="orthopy" src="https://nschloe.github.io/orthopy/orthopy-logo-with-text.png" width="30%"></a>
  <p align="center">All about orthogonal polynomials.</p>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/orthopy.svg?style=flat-square)](https://pypi.org/project/orthopy)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/orthopy.svg?style=flat-square)](https://pypi.org/pypi/orthopy/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1173151.svg?style=flat-square)](https://doi.org/10.5281/zenodo.1173151)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/orthopy.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/orthopy)
[![PyPi downloads](https://img.shields.io/pypi/dm/orthopy.svg?style=flat-square)](https://pypistats.org/packages/orthopy)

[![Discord](https://img.shields.io/static/v1?logo=discord&label=chat&message=on%20discord&color=7289da&style=flat-square)](https://discord.gg/hnTJ5MRX2Y)
[![orthogonal](https://img.shields.io/badge/orthogonal-yes-ff69b4.svg?style=flat-square)](https://github.com/nschloe/orthopy)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/orthopy/ci?style=flat-square)](https://github.com/nschloe/orthopy/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/orthopy.svg?style=flat-square)](https://codecov.io/gh/nschloe/orthopy)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/orthopy.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/orthopy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

orthopy provides various orthogonal polynomial classes for
[lines](#line-segment--1-1-with-weight-function-1-x%CE%B1-1-x%CE%B2),
[triangles](#triangle-42),
[disks](#disk-s2),
[spheres](#sphere-u2),
[n-cubes](#n-cube-cn),
[the nD space with weight function exp(-r<sup>2</sup>)](#nd-space-with-weight-function-exp-r2-enr2)
and more.
All computations are done using numerically stable recurrence schemes.  Furthermore, all
functions are fully vectorized and can return results in _exact arithmetic_.

### Basic usage

Install orthopy from [PyPi](https://pypi.org/project/orthopy) via
```
pip install orthopy
```
The main function of all submodules is the iterator `Eval` which evaluates the series of
orthogonal polynomials with increasing degree at given points using a recurrence
relation, e.g.,
```python
import orthopy

x = 0.5

evaluator = orthopy.c1.legendre.Eval(x, "classical")
for _ in range(5):
     print(next(evaluator))
```
```python
1.0          # P_0(0.5)
0.5          # P_1(0.5)
-0.125       # P_2(0.5)
-0.4375      # P_3(0.5)
-0.2890625   # P_4(0.5)
```
Other ways of getting the first `n` items are
<!--pytest-codeblocks:skip-->
```python
evaluator = Eval(x, "normal")
vals = [next(evaluator) for _ in range(n)]

import itertools
vals = list(itertools.islice(Eval(x, "normal"), n))
```
Instead of evaluating at only one point, you can provide any array for `x`; the
polynomials will then be evaluated for all points at once. You can also use sympy for
symbolic computation:
```python
import itertools
import orthopy
import sympy

x = sympy.Symbol("x")

evaluator = orthopy.c1.legendre.Eval(x, "classical")
for val in itertools.islice(evaluator, 5):
     print(sympy.expand(val))
```
```
1
x
3*x**2/2 - 1/2
5*x**3/2 - 3*x/2
35*x**4/8 - 15*x**2/4 + 3/8
```

All `Eval` methods have a `scaling` argument which can have three values:

  * `"monic"`: The leading coefficient is 1.
  * `"classical"`: The maximum value is 1 (or  (n+alpha over n)).
  * `"normal"`: The integral of the squared function over the domain is 1.

For univariate ("one-dimensional") integrals, every new iteration contains one function.
For bivariate ("two-dimensional") domains, every level will contain one function more
than the previous, and similarly for multivariate families. See the tree plots below.


### Line segment (-1, +1) with weight function (1-x)<sup>α</sup> (1+x)<sup>β</sup>

<img src="https://nschloe.github.io/orthopy/legendre.svg" width="100%"> | <img src="https://nschloe.github.io/orthopy/chebyshev1.svg" width="100%"> | <img src="https://nschloe.github.io/orthopy/chebyshev2.svg" width="100%">
:-------------------:|:------------------:|:-------------:|
Legendre             |  Chebyshev 1       |  Chebyshev 2  |

Jacobi, Gegenbauer (α=β), Chebyshev 1 (α=β=-1/2), Chebyshev 2 (α=β=1/2), Legendre
(α=β=0) polynomials.
<!--pytest-codeblocks:skip-->
```python
import orthopy

orthopy.c1.legendre.Eval(x, "normal")
orthopy.c1.chebyshev1.Eval(x, "normal")
orthopy.c1.chebyshev2.Eval(x, "normal")
orthopy.c1.gegenbauer.Eval(x, "normal", lmbda)
orthopy.c1.jacobi.Eval(x, "normal", alpha, beta)
```

The plots above are generated with
```python
import orthopy

orthopy.c1.jacobi.show(5, "normal", 0.0, 0.0)
# plot, savefig also exist
```

Recurrence coefficients can be explicitly retrieved by
```python
import orthopy

rc = orthopy.c1.jacobi.RecurrenceCoefficients(
    "monic",  # or "classical", "normal"
    alpha=0, beta=0, symbolic=True
)
print(rc.p0)
for k in range(5):
    print(rc[k])
```
```
1
(1, 0, None)
(1, 0, 1/3)
(1, 0, 4/15)
(1, 0, 9/35)
(1, 0, 16/63)
```


### 1D half-space with weight function x<sup>α</sup> exp(-r)
<img src="https://nschloe.github.io/orthopy/e1r.svg" width="45%">

(Generalized) Laguerre polynomials.
<!--pytest-codeblocks:skip-->
```python
evaluator = orthopy.e1r.Eval(x, alpha=0, scaling="normal")
```

### 1D space with weight function exp(-r<sup>2</sup>)
<img src="https://nschloe.github.io/orthopy/e1r2.svg" width="45%">

Hermite polynomials come in two standardizations:

  * `"physicists"` (against the weight function `exp(-x ** 2)`
  * `"probabilists"` (against the weight function `1 / sqrt(2 * pi) * exp(-x ** 2 / 2)`

<!--pytest-codeblocks:skip-->
```python
evaluator = orthopy.e1r2.Eval(
    x,
    "probabilists",  # or "physicists"
    "normal"
)
```

#### Associated Legendre "polynomials"
<img src="https://nschloe.github.io/orthopy/associated-legendre.svg" width="45%">

Not all of those are polynomials, so they should really be called associated Legendre
_functions_. The <i>k</i>th iteration contains _2k+1_ functions, indexed from
_-k_ to _k_. (See the color grouping in the above plot.)
<!--pytest-codeblocks:skip-->
```python
evaluator = orthopy.c1.associated_legendre.Eval(
    x, phi=None, standardization="natural", with_condon_shortley_phase=True
)
```

### Triangle (_T<sub>2</sub>_)
<img src="https://nschloe.github.io/orthopy/triangle-tree.png" width="40%">

orthopy's triangle orthogonal polynomials are evaluated in terms of [barycentric
coordinates](https://en.wikipedia.org/wiki/Barycentric_coordinate_system), so the
`X.shape[0]` has to be 3.

```python
import orthopy

bary = [0.1, 0.7, 0.2]
evaluator = orthopy.t2.Eval(bary, "normal")
```


### Disk (_S<sub>2</sub>_)

<img src="https://nschloe.github.io/orthopy/disk-xu-tree.png" width="70%"> | <img src="https://nschloe.github.io/orthopy/disk-zernike-tree.png" width="70%"> | <img src="https://nschloe.github.io/orthopy/disk-zernike2-tree.png" width="70%">
:------------:|:-----------------:|:-----------:|
Xu            |  [Zernike](https://en.wikipedia.org/wiki/Zernike_polynomials)          |  Zernike 2  |

orthopy contains several families of orthogonal polynomials on the unit disk: After
[Xu](https://arxiv.org/abs/1701.02709),
[Zernike](https://en.wikipedia.org/wiki/Zernike_polynomials), and a simplified version
of Zernike polynomials.

```python
import orthopy

x = [0.1, -0.3]

evaluator = orthopy.s2.xu.Eval(x, "normal")
# evaluator = orthopy.s2.zernike.Eval(x, "normal")
# evaluator = orthopy.s2.zernike2.Eval(x, "normal")
```


### Sphere (_U<sub>3</sub>_)

<img src="https://nschloe.github.io/orthopy/sph-tree.png" width="50%">

Complex-valued _spherical harmonics,_ plotted with
[cplot](https://github.com/nschloe/cplot/) coloring (black=zero, green=real positive,
pink=real negative, blue=imaginary positive, yellow=imaginary negative). The functions
in the middle are real-valued. The complex angle takes _n_ turns on the <i>n</i>th
level.
<!--pytest-codeblocks:skip-->
```python
evaluator = orthopy.u3.EvalCartesian(
    x,
    scaling="quantum mechanic"  # or "acoustic", "geodetic", "schmidt"
)

evaluator = orthopy.u3.EvalSpherical(
    theta_phi,  # polar, azimuthal angles
    scaling="quantum mechanic"  # or "acoustic", "geodetic", "schmidt"
)
```
To generate the above plot, write the tree mesh to a file
```python
import orthopy

orthopy.u3.write_tree("u3.vtk", 5, "quantum mechanic")
```
and open it with [ParaView](https://www.paraview.org/). Select the _srgb1_ data set and
turn off _Map Scalars_.

### _n_-Cube (_C<sub>n</sub>_)

<img src="https://nschloe.github.io/orthopy/c1.svg" width="100%"> | <img src="https://nschloe.github.io/orthopy/c2.png" width="100%"> | <img src="https://nschloe.github.io/orthopy/c3.png" width="100%">
:-------------------------:|:------------------:|:---------------:|
C<sub>1</sub> (Legendre)   |  C<sub>2</sub>     |  C<sub>3</sub>  |

Jacobi product polynomials.
All polynomials are normalized on the n-dimensional cube. The dimensionality is
determined by `X.shape[0]`.

<!--pytest-codeblocks:skip-->
```python
evaluator = orthopy.cn.Eval(X, alpha=0, beta=0)
values, degrees = next(evaluator)
```

### <i>n</i>D space with weight function exp(-r<sup>2</sup>) (_E<sub>n</sub><sup>r<sup>2</sup></sup>_)

<img src="https://nschloe.github.io/orthopy/e1r2.svg" width="100%"> | <img src="https://nschloe.github.io/orthopy/e2r2.png" width="100%"> | <img src="https://nschloe.github.io/orthopy/e3r2.png" width="100%">
:-------------------------:|:------------------:|:---------------:|
_E<sub>1</sub><sup>r<sup>2</sup></sup>_   |  _E<sub>2</sub><sup>r<sup>2</sup></sup>_     | _E<sub>3</sub><sup>r<sup>2</sup></sup>_  |

Hermite product polynomials.
All polynomials are normalized over the measure. The dimensionality is determined by
`X.shape[0]`.

<!--pytest-codeblocks:skip-->
```python
evaluator = orthopy.enr2.Eval(
    x,
    standardization="probabilists"  # or "physicists"
)
values, degrees = next(evaluator)
```


### Other tools

 * Generating recurrence coefficients for 1D domains with
   [Stieltjes](https://github.com/nschloe/orthopy/wiki/Generating-1D-recurrence-coefficients-for-a-given-weight#stieltjes),
   [Golub-Welsch](https://github.com/nschloe/orthopy/wiki/Generating-1D-recurrence-coefficients-for-a-given-weight#golub-welsch),
   [Chebyshev](https://github.com/nschloe/orthopy/wiki/Generating-1D-recurrence-coefficients-for-a-given-weight#chebyshev), and
   [modified
   Chebyshev](https://github.com/nschloe/orthopy/wiki/Generating-1D-recurrence-coefficients-for-a-given-weight#modified-chebyshev).

 * The the sanity of recurrence coefficients with test 3 from [Gautschi's article](https://doi.org/10.1007/BF02218441):
   computing the weighted sum of orthogonal polynomials:
   <!--pytest-codeblocks:skip-->
   ```python
   orthopy.tools.gautschi_test_3(moments, alpha, beta)
   ```

 * [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for
   computing the weighted sum of orthogonal polynomials:
   <!--pytest-codeblocks:skip-->
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
* [Yuan Xu, Orthogonal polynomials of several variables, arxiv.org, January 2017](https://arxiv.org/abs/1701.02709)

### License
This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
