[![Build Status](https://github.com/MarcoGorelli/cython-lint/workflows/tox/badge.svg)](https://github.com/MarcoGorelli/cython-lint/actions?workflow=tox)
[![Coverage](https://codecov.io/gh/MarcoGorelli/cython-lint/branch/main/graph/badge.svg)](https://codecov.io/gh/MarcoGorelli/cython-lint)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/MarcoGorelli/cython-lint/main.svg)](https://results.pre-commit.ci/latest/github/MarcoGorelli/cython-lint/main)

cython-lint
===========

A tool and pre-commit hook to lint Cython files.

## Installation

```console
$ pip install cython-lint
```

## Usage as a pre-commit hook

See [pre-commit](https://github.com/pre-commit/pre-commit) for instructions

Sample `.pre-commit-config.yaml`:

```yaml
-   repo: https://github.com/MarcoGorelli/cython-lint
    rev: v0.3.1
    hooks:
    -   id: cython-lint
```

## Command-line example

```console
$ cython-lint my_file_1.pyx my_file_2.pyx
my_file_1.pyx:54:5: 'get_conversion_factor' imported but unused
my_file_2.pyx:1112:38: 'mod' defined but unused
```


## Configuration

The following configuration options are available:
- exclude lines by including a ``# no-cython-lint`` comment (analogous to ``# noqa`` in ``flake8``);
- use the command-line argument ``--max-line-length`` to control the maximum line length used by pycodestyle;
- use the command-line argument ``--no-pycodestyle`` if you don't want the pycodestyle checks.

Currently, the following checks are implemented:

- variable defined but unused
- variable imported but unused
- comma after base type definition (e.g. ``cdef ndarray, arr``)
- pycodestyle checks, except these that aren't in general applicable to Cython code:
    - E121 continuation line under-indented for hanging indent
    - E123 closing bracket does not match indentation of opening bracket’s line
    - E126 continuation line over-indented for hanging indent
    - E133 closing bracket is missing indentation
    - E203 whitespace before ‘,’, ‘;’, or ‘:’
    - E211 whitespace before '('
    - E225 missing whitespace around operator
    - E226 missing whitespace around arithmetic operator
    - E227 missing whitespace around bitwise or shift operator
    - E241 multiple spaces after ‘,’
    - E242 tab after ‘,’
    - E271 multiple spaces after keyword
    - E272 multiple spaces before keyword
    - E275 missing whitespace after keyword
    - E4 imports (``isort`` supports Cython code, best to just use that)
    - E704 multiple statements on one line (def)
    - E9 runtime
    - W5 line break warning

More to come! Requests welcome!
