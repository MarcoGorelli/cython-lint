from __future__ import annotations

import textwrap
from typing import Any

import pytest

from cython_lint.string_fixer import main

TESTS = (
    # Base cases
    ("''", "''", 0),
    ('""', "''", 1),
    (r'"\'"', r'"\'"', 0),
    (r'"\""', r'"\""', 0),
    (r"'\"\"'", r"'\"\"'", 0),
    # String somewhere in the line
    ('x = "foo"', "x = 'foo'", 1),
    # Test escaped characters
    (r'"\'"', r'"\'"', 0),
    # Docstring
    ('""" Foo """', '""" Foo """', 0),
    (
        textwrap.dedent(
            """
        x = " \\
        foo \\
        "\n
        """,
        ),
        textwrap.dedent(
            """
        x = ' \\
        foo \\
        '\n
        """,
        ),
        1,
    ),
    ('"foo""bar"', "'foo''bar'", 1),
)


@pytest.mark.parametrize(("input_s", "output", "expected_retval"), TESTS)
def test_rewrite(
    input_s: str,
    output: str,
    expected_retval: int,
    tmpdir: Any,
) -> None:
    path = tmpdir.join("file.py")
    path.write(input_s)
    retval = main([str(path), "--never"])
    assert path.read() == output
    assert retval == expected_retval


DOUBLE_QUOTE_TESTS = (
    # Base cases
    ("''", '""', 1),
    ('""', '""', 0),
    (r'"\'"', r'"\'"', 0),
    (r'"\""', r'"\""', 0),
    (r"'\"\"'", r"'\"\"'", 0),
    # String somewhere in the line
    ('x = "foo"', 'x = "foo"', 0),
    ("x = 'foo'", 'x = "foo"', 1),
    # Test escaped characters
    (r'"\'"', r'"\'"', 0),
    # Docstring
    ('""" Foo """', '""" Foo """', 0),
    (
        textwrap.dedent(
            """
        x = " \\
        foo \\
        "\n
        """,
        ),
        textwrap.dedent(
            """
        x = " \\
        foo \\
        "\n
        """,
        ),
        0,
    ),
    (
        textwrap.dedent(
            """
        x = ' \\
        foo \\
        '\n
        """,
        ),
        textwrap.dedent(
            """
        x = " \\
        foo \\
        "\n
        """,
        ),
        1,
    ),
    ('"foo""bar"', '"foo""bar"', 0),
    ("'foo''bar'", '"foo""bar"', 1),
    # f-strings
    ('''f"'foo'"''', '''f"'foo'"''', 0),
    ('''f"{'foo'}"''', '''f"{'foo'}"''', 0),
)


@pytest.mark.parametrize(
    ("input_s", "output", "expected_retval"),
    DOUBLE_QUOTE_TESTS,
)
def test_rewrite_double_quotes(
    input_s: str,
    output: str,
    expected_retval: int,
    tmpdir: Any,
) -> None:
    path = tmpdir.join("file.py")
    path.write(input_s)
    retval = main([str(path)])
    assert path.read() == output
    assert retval == expected_retval


def test_rewrite_crlf(tmpdir: Any) -> None:
    f = tmpdir.join("f.py")
    f.write_binary(b'"foo"\r\n"bar"\r\n')
    assert main((str(f), "--never"))
    assert f.read_binary() == b"'foo'\r\n'bar'\r\n"
