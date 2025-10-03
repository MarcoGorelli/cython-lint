[![Build Status](https://github.com/MarcoGorelli/cython-lint/workflows/tox/badge.svg)](https://github.com/MarcoGorelli/cython-lint/actions?workflow=tox)
[![Coverage](https://codecov.io/gh/MarcoGorelli/cython-lint/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoGorelli/cython-lint)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/MarcoGorelli/cython-lint/main.svg)](https://results.pre-commit.ci/latest/github/MarcoGorelli/cython-lint/main)

cython-lint
===========

Everything ``flake8`` used to do (by accident), plus much more.

A tool and pre-commit hook to lint Cython files.

## Used by

Here's some major projects using ``cython-lint`` - is yours missing? Feel free to open a pull request!

- [arrow-adbc](https://github.com/apache/arrow-adbc)
- [ansible-pylibssh](https://github.com/ansible/pylibssh)
- [arrow](https://github.com/apache/arrow)
- [CuPy](https://github.com/cupy/cupy)
- [dpctl](https://github.com/IntelPython/dpctl)
- [frozenlist](https://github.com/aio-libs/frozenlist)
- [mkl_fft](https://github.com/IntelPython/mkl_fft)
- [mkl-service](https://github.com/IntelPython/mkl-service)
- [open library](https://github.com/internetarchive/openlibrary)
- [pandas](https://github.com/pandas-dev/pandas)
- [pymatgen](https://github.com/materialsproject/pymatgen)
- [PyRFC](https://github.com/SAP/PyRFC)
- [RAPIDS cuspatial](https://github.com/rapidsai/cuspatial)
- [RAPIDS cudf](https://github.com/rapidsai/cudf)
- [RAPIDS Memory Manager](https://github.com/rapidsai/rmm)
- [sagemath/sage](https://github.com/sagemath/sage/issues)
- [scikit-image](https://github.com/scikit-image/scikit-image)
- [scikit-learn](https://github.com/scikit-learn/scikit-learn)
- [scipy](https://github.com/scipy/scipy)
- [spaCy](https://github.com/explosion/spaCy)
- [yarl](https://github.com/aio-libs/yarl)
- [yt](https://github.com/yt-project/yt)

## Installation

```console
$ pip install cython-lint
```

## Usage as a pre-commit hook

See [pre-commit](https://github.com/pre-commit/pre-commit) for instructions

Sample `.pre-commit-config.yaml`:

```yaml
-   repo: https://github.com/MarcoGorelli/cython-lint
    rev: v0.17.0
    hooks:
    -   id: cython-lint
    -   id: double-quote-cython-strings
```

## Command-line example

```console
$ cython-lint my_file_1.pyx my_file_2.pyx
my_file_1.pyx:54:5: 'get_conversion_factor' imported but unused
my_file_2.pyx:1112:38: 'mod' defined but unused (try prefixing with underscore?)
my_file_3.pyx:4:9: dangerous default value!
my_file_3.pyx:5:9: comma after base type in definition
```

## Configuration

The following configuration options are available:
- exclude lines by including a ``# no-cython-lint`` comment (analogous to ``# noqa`` in ``flake8``);

as well as the following command-line arguments:
- ``--max-line-length`` to control the maximum line length used by pycodestyle;
- ``--no-pycodestyle`` if you don't want the pycodestyle checks;
- ``--ban-relative-imports`` if you want to ban relative imports;
- ``--ignore`` if you want to ignore some specific pycodestyle checks;
- ``--files`` to pass a Regex pattern with which to match files to include;
- ``--exclude`` to pass a Regex pattern with which to match files to exclude.

Configuration can be set project-wise in a `pyproject.toml` file at the root of the project.
Here's an example `pyproject.toml`:
```
[tool.cython-lint]
max-line-length = 88
ignore = ['E503', 'E504']
exclude = 'my_project/excluded_cython_file.pyx'
```

## Which checks are implemented?

- assert statement with tuple condition (always true...)
- comma after base type definition (e.g. ``cdef ndarray, arr``)
- comparison between constants
- dangerous default value
- dict key repeated
- dict key variable repeated
- f-string without placeholders
- if-statement with tuple condition (always true...)
- late-binding closures https://docs.python-guide.org/writing/gotchas/#late-binding-closures
- pointless string statement
- ``pycodestyle`` nitpicks (which you can turn off with ``--no-pycodestyle``)
- repeated element in set
- ``.strip``, ``.rstrip``, or ``.lstrip`` used with repeated characters
- unnecessary list index lookup
- unnecessary import alias
- variable defined but unused
- variable imported but unused

In addition, the following automated fixers are implemented:

- double-quote-cython-strings (replace single quotes with double quotes, like the ``black`` formatter does)

More to come! Requests welcome!
