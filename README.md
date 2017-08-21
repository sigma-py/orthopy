# orthopy

Python tools for orthogonal polynomials and Gaussian quadature.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/orthopy/master.svg)](https://circleci.com/gh/nschloe/orthopy/tree/master)
[![codecov](https://codecov.io/gh/nschloe/orthopy/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/orthopy)
[![PyPi Version](https://img.shields.io/pypi/v/orthopy.svg)](https://pypi.python.org/pypi/orthopy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/orthopy.svg?style=social&label=Stars&maxAge=2592000)](https://github.com/nschloe/orthopy)

Relevant publications:

 * [Gene H. Golub and John H. Welsch, Calculation of Gauss Quadrature Rules, Mathematics of Computation, Vol. 23, No. 106 (Apr., 1969), pp. 221-230+s1-s10](https://dx.doi.org/10.2307/2004418)
 * [W. Gautschi, How and how not to check Gaussian quadrature formulae, BIT Numerical Mathematics, June 1983, Volume 23, Issue 2, pp 209–216](https://doi.org/10.1007/BF02218441)
 * [W. Gautschi, Algorithm 726: ORTHPOL–a package of routines for generating orthogonal polynomials and Gauss-type quadrature rules, ACM Transactions on Mathematical Software (TOMS), Volume 20, Issue 1, March 1994, Pages 21-62](http://doi.org/10.1145/174603.174605)


### Bring your own weight function
Do you need Gauss points and weights with respect to your own special weight
function `w`? No problem: You simply need to provide the desired number of Gauss
points `n` and the moments `integral(w(x) * x**k)` with `k` up to `2n+1`:
```python
my_gauss = quadpy.line_segment.Gauss(n, moments)
```
quadpy takes care of the rest (using the Golub-Welsch algorithm). The resulting
scheme has degree `2n-1` and can be used like any other scheme in quadpy.
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
