# orthopy

Python tools for orthogonal polynomials and Gaussian quadature.

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/orthopy/master.svg)](https://circleci.com/gh/nschloe/orthopy/tree/master)
[![codecov](https://codecov.io/gh/nschloe/orthopy/branch/master/graph/badge.svg)](https://codecov.io/gh/nschloe/orthopy)
[![PyPi Version](https://img.shields.io/pypi/v/orthopy.svg)](https://pypi.python.org/pypi/orthopy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/orthopy.svg?style=social&label=Stars&maxAge=2592000)](https://github.com/nschloe/orthopy)

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
