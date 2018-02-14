# orthopy

Your one-stop shop for orthogonal polynomials in Python.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/orthopy/master.svg)](https://circleci.com/gh/nschloe/orthopy/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/orthopy.svg)](https://codecov.io/gh/nschloe/orthopy)
[![Codacy grade](https://img.shields.io/codacy/grade/895c05bd82364370841cea4ab2121c99.svg)](https://app.codacy.com/app/nschloe/orthopy/dashboard)
[![awesome](https://img.shields.io/badge/awesome-yes-brightgreen.svg)](https://img.shields.io/badge/awesome-yes-brightgreen.svg)
[![PyPi Version](https://img.shields.io/pypi/v/orthopy.svg)](https://pypi.python.org/pypi/orthopy)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1173151.svg)](https://doi.org/10.5281/zenodo.1173151)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/orthopy.svg?style=social&label=Stars)](https://github.com/nschloe/orthopy)

Various orthogonal polynomial classes for
[lines](#line-segment--1-1-with-weight-function-1-x%CE%B1-1-x%CE%B2),
[triangles](#triangle),
[quadrilaterals](#quadrilateral),
[disks](#disk),
[spheres](#sphere),
[hexahedra](#hexahedron), and
[n-cubes](#n-cube).
All computations are done using numerically stable recurrence schemes.
Furthermore, all functions are fully vectorized and can return results in
[_exact arithmetic_](#symbolic-and-numerical-computation).

_Note:_ In previous versions, orthopy contained tools for working with Gauss
quadrature rules as well. Those have moved over to
[quadpy](https://github.com/nschloe/quadpy).

### Line segment [-1, +1] with weight function (1-x)<sup>α</sup> (1-x)<sup>β</sup>

<img src="https://nschloe.github.io/orthopy/line-segment-legendre.png" width="25%">

Jacobi, Gegenbauer (α=β), Chebyshev 1 (α=β=-1/2), Chebyshev 2 (α=β=1/2),
Legendre (α=β=0) polynomials.

```python
vals = orthopy.line_segment.tree_jacobi(x, 4, alpha, beta, 'normal', symbolic=False)
```

Recurrence coefficients can be explicitly retrieved by
```python
p0, a, b, c = orthopy.line_segment.recurrence_coefficients.jacobi(n, a, b, 'monic')
```
Possible choices for the standardization are `'monic'`, `'p(1)=(n+alpha over
n)'`, and `'normal`.


#### Associated Legendre polynomials

<img src="https://nschloe.github.io/orthopy/line-segment-alp.png" width="25%">

```python
vals = orthopy.line_segment.tree_alp(
    x, 4, phi=None, standardization='natural', with_condon_shortley_phase=True,
    symbolic=False
    )
```

### 1D half-space with weight function x<sup>α</sup> exp(-r)
<img src="https://nschloe.github.io/orthopy/e1r.png" width="25%">

(Generalized) Laguerre polynomials.
```python
vals = orthopy.e1r.tree(x, 4, alpha=0, standardization='normal', symbolic=False)
```


### 1D space with weight function exp(-r<sup>2</sup>)
<img src="https://nschloe.github.io/orthopy/e1r2.png" width="25%">

Hermite polynomials.
```python
vals = orthopy.e1r2.tree(x, 4, 'normal', symbolic=False)
```
All polynomials are normalized over the measure.


### Triangle

<img src="https://nschloe.github.io/orthopy/triangle-1-0.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/triangle-2-1.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/triangle-3-1.png" width="70%">
:-------------------:|:------------------:|:----------:|
n=1, k=0             |  n=2, k=1          |  n=3, k=1  |

```python
vals = orthopy.triangle.tree(x, 4, 'normal', symbolic=False)
```
Available standardizations are
  * `'normal'` (normalized polynomials, i.e., the integral of the squared function equals 1) and
  * `'1'` where the polynomial is `1` in at least one corner of the triangle.


### Quadrilateral

<img src="https://nschloe.github.io/orthopy/quad-1-0.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/quad-2-1.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/quad-3-1.png" width="70%">
:-------------------:|:------------------:|:----------:|
n=1, k=0             |  n=2, k=1          |  n=3, k=1  |

```python
vals = orthopy.quadrilateral.tree(x, 4, symbolic=False)
```
All polynomials are normalized on the quadrilateral.


### Disk

<img src="https://nschloe.github.io/orthopy/disk-1-0.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/disk-2-1.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/disk-4-3.png" width="70%">
:-------------------:|:------------------:|:----------:|
n=1, k=0             |  n=2, k=1          |  n=4, k=3  |

```python
vals = orthopy.disk.tree(x, 4, symbolic=False)
```
All polynomials are normalized on the unit disk.


### 2D space with weight function exp(-r<sup>2</sup>)

<img src="https://nschloe.github.io/orthopy/e2r2-1-0.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/e2r2-2-1.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/e2r2-3-1.png" width="70%">
:-------------------:|:------------------:|:----------:|
n=1, k=0             |  n=2, k=1          |  n=3, k=1  |

```python
vals = orthopy.e2r2.tree(x, 4, symbolic=False)
```
All polynomials are normalized over the measure.


### Sphere

<img src="https://nschloe.github.io/orthopy/sphere-1-0.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/sphere-2-1.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/sphere-5-3.png" width="70%">
:-------------------:|:------------------:|:----------:|
n=1, k=0             |  n=2, k=1          |  n=5, k=3  |

_Spherical harmonics._

(Note that spherical harmonics are complex-valued in general; the above plots
only show the absolute values.)

```python
vals = orthopy.sphere.tree_sph(
    polar, azimuthal, n, standardization='quantum mechanic', symbolic=False
    )
```


### Hexahedron

<img src="https://nschloe.github.io/orthopy/hexa-1-0.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/hexa-2-1.png" width="70%"> |
<img src="https://nschloe.github.io/orthopy/hexa-5-5.png" width="70%">
:-------------------:|:------------------:|:----------:|
n=1, k=0             |  n=2, k=1          |  n=5, k=5  |

```python
vals = orthopy.hexahedron.tree(x, 3, symbolic=False)
```
All polynomials are normalized on the hexahedron.


### n-Cube

```python
vals = orthopy.ncube.tree(x, 6, symbolic=False)
```
All polynomials are normalized on the n-dimensional cube. The dimensionality is
determined by `X.shape[0]`.

### nD space with weight function exp(-r<sup>2</sup>)

```python
vals = orthopy.enr2.tree(x, 4, symbolic=False)
```
All polynomials are normalized over the measure. The dimensionality is
determined by `X.shape[0]`.


### Other tools

 * [Clenshaw algorithm](https://en.wikipedia.org/wiki/Clenshaw_algorithm) for
   computing the weighted sum of orthogonal polynomials:
   ```python
   vals = orthopy.line_segment.clenshaw(a, alpha, beta, t)
   ```


#### Symbolic and numerical computation

By default, all operations are performed numerically. However, if
`symbolic=True` is specified, all computations are performed symbolically. This
can be used, for example, to get explicit representations of the polynomials:
```python
import numpy
import orthopy
import sympy

b0, b1, b2 = sympy.Symbol('b0'), sympy.Symbol('b1'), sympy.Symbol('b2')

tree = orthopy.triangle.tree(numpy.array([b0, b1, b2]), 3, 'normal', symbolic=True)

print(sympy.expand(tree[3][1]))
```
```
42*sqrt(6)*b0*b2**2 - 24*sqrt(6)*b0*b2 + 2*sqrt(6)*b0 - 42*sqrt(6)*b1*b2**2
+ 24*sqrt(6)*b1*b2 - 2*sqrt(6)*b1
```


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
