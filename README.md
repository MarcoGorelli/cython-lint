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
    rev: v0.1.2
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

Exclude lines by include a ``# no-cython-lint`` comment (analogous to ``# noqa`` in ``flake8``).

Currently, the following checks are implemented:

- variable defined but unused
- variable imported but unused

More to come! Requests welcome!
