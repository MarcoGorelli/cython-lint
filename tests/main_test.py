from __future__ import annotations

import os
from typing import Any

import Cython
import pytest

from cython_lint.cython_lint import _main
from cython_lint.cython_lint import main

INCLUDE_FILE_0 = os.path.join("tests", "data", "foo.pxi")
INCLUDE_FILE_1 = os.path.join("tests", "data", "bar.pxi")


@pytest.mark.parametrize(
    "src",
    [
        (
            "cdef bint foo():\n"
            "    cdef int _\n"
            "    cdef int _a\n"
            "    cdef int unused\n"
            "    cdef int unused_a\n"
        ),
        (
            "cdef bint foo():\n"
            "    cdef int _ = 1\n"
            "    cdef int _a = 1\n"
            "    cdef int unused = 1\n"
            "    cdef int unused_a = 1\n"
        ),
        (
            "cdef bint foo():\n"
            "    cdef int _\n"
            "    cdef int _a\n"
            "    cdef int unused\n"
            "    cdef int unused_a\n"
            "    _ = 1\n"
            "    _a = 1\n"
            "    unused = 1\n"
            "    unused_a = 1\n"
        ),
    ],
)
def test_named_unused(capsys: Any, src: str) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == ""
    assert ret == 0


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "cdef bint foo():\n    cdef int a\n",
            "t.py:2:14: 'a' defined but unused (try prefixing with underscore?)\n",
        ),
        (
            "cdef bint foo():\n    cdef int a\n    a = 3\n",
            "t.py:3:5: 'a' defined but unused (try prefixing with underscore?)\n",
        ),
        (
            "cdef bint foo():\n    cdef int a\n    a, b, myclass.c = 3\n    return b\n",
            "t.py:3:5: 'a' defined but unused (try prefixing with underscore?)\n",
        ),
    ],
)
def test_assigned_unused(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "cimport foo\n",
            "t.py:1:9: 'foo' imported but unused\n",
        ),
        (
            "from foo cimport bar\n",
            "t.py:1:18: 'bar' imported but unused\n",
        ),
        (
            "from foo import bar, bar2\n",
            "t.py:1:17: 'bar' imported but unused\n"
            "t.py:1:22: 'bar2' imported but unused\n",
        ),
        (
            'cimport quox\ninclude "foo.pxi"\n',
            "t.py:1:9: 'quox' imported but unused\n",
        ),
        (
            "import numpy as np\n",
            "t.py:1:8: 'np' imported but unused\n",
        ),
    ],
)
def test_imported_unused(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, "t.py", ext=".pyx")
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "cimport foo as foo\n\nfoo\n",
            "t.py:1:9: Found useless import alias\n",
        ),
        (
            "from foo cimport (\n    a as a\n,    b,\n)\na = b\n",
            "t.py:2:2: Found useless import alias\n",
        ),
    ],
)
def test_useless_alias(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "from ._common cimport foo\n_ = foo\n",
            "t.py:1:0: Found relative import\n",
        ),
        (
            "from ._common cimport foo as foot\n_ = foot\n",
            "t.py:1:0: Found relative import\n",
        ),
        (
            "from ._common import foo\n_ = foo\n",
            "t.py:1:0: Found relative import\n",
        ),
        (
            "from ._common import foo as foot\n_ = foot\n",
            "t.py:1:0: Found relative import\n",
        ),
    ],
)
def test_relative_import(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True, ban_relative_imports=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1

    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == ""
    assert ret == 0


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            'def foo():\n    print()\n    "foobar"\n',
            "t.py:3:5: pointless string statement\n",
        ),
    ],
)
def test_pointless_string_statement(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "for i, v in enumerate(values):\n    a == values[i]\n",
            "t.py:2:10: unnecessary list index lookup: use `v` instead of `values[i]`\n",
        ),
        (
            "for i, v in enumerate(values):\n    a = values[i]\n",
            "t.py:2:9: unnecessary list index lookup: use `v` instead of `values[i]`\n",
        ),
        (
            "for i, v in enumerate(values):\n    a.append(values[i])\n",
            "t.py:2:14: unnecessary list index lookup: use `v` instead of `values[i]`\n",
        ),
        (
            "for i, v in enumerate(values):\n    values[i] == a\n",
            "t.py:2:5: unnecessary list index lookup: use `v` instead of `values[i]`\n",
        ),
    ],
)
def test_unnecessary_index(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.skipif(
    tuple(Cython.__version__.split(".")) > ("3",),
    reason="invalid syntax in new Cython",
)
@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "cdef a[0, 1], b\n",
            "t.py:1:5: comma after base type in definition\n",
        ),
        (
            "cdef a(0, 1), b\n",
            "t.py:1:5: comma after base type in definition\n",
        ),
    ],
)
def test_misplaced_comma_old_cython(
    capsys: Any,
    src: str,
    expected: str,
) -> None:  # pragma: no cover
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            'cdef void(): f"abc"\n',
            "t.py:1:13: f-string without any placeholders\n",
        ),
    ],
)
def test_f_string_not_formatted(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "import foo\nfoo()\ndef bar(foo): pass\n",
            "t.py:3:9: 'foo' shadows global import on line 1 col 8\n",
        ),
        (
            "import foo\nfoo()\ndef bar(baz):\n    foo = 3\n    return foo\n",
            "t.py:4:5: 'foo' shadows global import on line 1 col 8\n",
        ),
    ],
)
def test_shadows_import(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    if tuple(Cython.__version__.split(".")) < ("3",):  # pragma: no cover
        # old Cython records the location slightly differently
        # no big deal
        expected = expected.replace("t.py:3:9", "t.py:3:12")
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "{0, 0, 1}\n",
            "t.py:1:2: Repeated element in set\n",
        ),
    ],
)
def test_repeated_set_element(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "{0: 1, 0: 2}\n",
            "t.py:1:1: dict key 0 repeated 2 times\n",
        ),
        (
            "{a: 1, a: 2}\n",
            "t.py:1:1: dict key variable a repeated 2 times\n",
        ),
    ],
)
def test_repeated_dict_keys(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "def foo(a = []): pass\n",
            "t.py:1:9: dangerous default value!\n",
        ),
        (
            "cdef void foo(int a = [])\n",
            "t.py:1:15: dangerous default value!\n",
        ),
    ],
)
def test_dangerous_default(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "if (False,): pass\n",
            "t.py:1:3: if-statement with tuple as condition is always true - "
            "perhaps remove comma?\n",
        ),
        (
            "if False:\n    pass\nelif (False,):\n    pass\n",
            "t.py:3:5: if-statement with tuple as condition is always true - "
            "perhaps remove comma?\n",
        ),
        (
            "assert (False,)\n",
            "t.py:1:0: assert statement with tuple as condition is always "
            "true - perhaps remove comma?\n",
        ),
    ],
)
def test_always_true(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "if 0 == 0: pass\n",
            "t.py:1:6: Comparison between constants\n",
        ),
        (
            'if "0" == "0": pass\n',
            "t.py:1:8: Comparison between constants\n",
        ),
    ],
)
def test_constants_comparison(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            'mystring.strip("aab")\n',
            "t.py:1:15: Using 'strip' with repeated elements\n",
        ),
        (
            'mystring.lstrip("aab")\n',
            "t.py:1:16: Using 'lstrip' with repeated elements\n",
        ),
        (
            'mystring.rstrip("aab")\n',
            "t.py:1:16: Using 'rstrip' with repeated elements\n",
        ),
    ],
)
def test_strip_repeated_elements(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        (
            "def create_multipliers():\n"
            "    return [lambda x : i * x for i in range(5)]\n",
            "t.py:2:12: Late binding closure! Careful "
            "https://docs.python-guide.org/writing/gotchas/"
            "#late-binding-closures\n",
        ),
        (
            "def create_multipliers():\n"
            "    return {i: lambda x : i * x for i in range(5)}\n",
            "t.py:2:12: Late binding closure! Careful "
            "https://docs.python-guide.org/writing/gotchas/"
            "#late-binding-closures\n",
        ),
        (
            "lst = []\n"
            "for i in range(5):\n"
            "    def foo(a):\n"
            "        return i * a\n"
            "    lst.append(foo(a))\n",
            "t.py:2:1: Late binding closure! Careful "
            "https://docs.python-guide.org/writing/gotchas/"
            "#late-binding-closures\n",
        ),
    ],
)
def test_late_binding_closure(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    ("ignore", "expected", "exp_ret"),
    [
        (
            set(),
            "{0}:1:11: E701 multiple statements on one line (colon)\n"
            "{0}:2:6: W291 trailing whitespace\n",
            1,
        ),
        (
            {"W291"},
            "{0}:1:11: E701 multiple statements on one line (colon)\n",
            1,
        ),
        ({"W291", "E701"}, "", 0),
    ],
)
def test_pycodestyle(
    tmpdir: Any,
    capsys: Any,
    ignore: set[str],
    expected: str,
    exp_ret: int,
) -> None:
    file = os.path.join(tmpdir, "t.py")
    with open(file, "w", encoding="utf-8") as fd:
        fd.write("while True: pass\n")  # E701
        fd.write("x = 1 \n")  # W291
    with open(os.path.join(tmpdir, "tox.ini"), "w") as fd:
        fd.write("[pycodestyle]\nstatistics=True\n")
    src = ""
    ret = _main(src, file, ext=".pxd", ignore=ignore)
    out, _ = capsys.readouterr()

    assert out == expected.format(file)
    assert ret == exp_ret


@pytest.mark.skipif(
    tuple(Cython.__version__.split(".")) < ("3",),
    reason="invalid syntax only in new Cython",
)
def test_pycodestyle_when_ast_parsing_fails(
    tmpdir: Any,
    capsys: Any,
) -> None:  # pragma: no cover
    file = os.path.join(tmpdir, "t.py")
    src = (
        "extending.pyx\n"
        "-------------\n"
        "\n"
        ".. include:: ../../../../../../numpy/random/examples/extending.pyx\n"
    )
    with open(file, "w", encoding="utf-8") as fd:
        fd.write(src)
    with open(os.path.join(tmpdir, "tox.ini"), "w") as fd:
        fd.write("[pycodestyle]\nstatistics=True\n")
    ret = _main(src, file, ext=".pyx")
    out, _ = capsys.readouterr()
    expected = (
        f"Skipping file {file}, as it cannot be parsed. Error: "
        "AttributeError(\"'_thread._local' object has no attribute "
        "'cython_errors_stack'\")\n"
        f"{file}:4:11: E231 missing whitespace after ':'\n"
    )
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    "src",
    [
        "cdef bint foo():\n    raise NotImplemen",
        "cdef bint foo():\n    cdef int i\n    for i in bar: pass\n",
        "cdef bint foo(a):\n   pass\n",
        "cdef bint foo(int a):\n   pass\n",
        "cdef bint foo(int *a):\n   pass\n",
        "cdef bint* foo(int a):\n   pass\n",
        "cdef bint foo():\n    cdef int i\n    for i, j in bar: pass\n",
        "cdef bint foo(object (*operation)(int64_t value, object right)):\n   pass\n",
        "class Foo: pass\n",
        "cdef class Foo: pass\n",
        "cdef bint foo(a):\n    bar(<int>a)\n",
        "cdef bint foo(a):\n    bar(i for i in a)\n",
        "cdef bint foo(a):\n    bar(lambda x: x)\n",
        "import int64_t\nctypedef fused foo:\n    int64_t\n    int32_t\n",
        f'import quox\ninclude "{INCLUDE_FILE_0}"\n',
        f'import quox\ninclude "{INCLUDE_FILE_1}"\n',
        "from quox cimport *\n",
        "cdef inline bool _compare_records(\n"
        "    const FrontierRecord& left,\n"
        "    const FrontierRecord& right,\n"
        "):\n"
        "    return left.improvement < right.improvement\n",
        "from foo import bar\nctypedef fused quox:\n    bar\n",
        "cimport cython\n"
        "ctypedef fused int_or_float:\n"
        "    cython.integral\n"
        "    cython.floating\n"
        "    signed long long\n",
        "cdef inline bool _compare_records(\n    double x[],\n):\n    return 1\n",
        "cimport foo.bar\ncdef bool quox():\n    foo.bar.ni()\n",
        "from foo cimport CTaskArgByReference\n"
        "cdef baz():\n"
        "    bar(new CTaskArgByReference())\n",
        "cdef PyObject** make_kind_names(list strings):\n"
        "    cdef PyObject** array = <PyObject**>mem.alloc(len(strings), sizeof(PyObject*))\n"
        "    cdef object name\n"
        "    for i, string in enumerate(strings):\n"
        "        name = intern(string)\n"
        "        Py_XINCREF(<PyObject*>name)\n"
        "        array[i] = <PyObject*>name\n"
        "    return <PyObject**>array\n",
        "def foo():\n    cdef int i\n    i = 3\n    print(i)\n",
        "def foo(int a[1][1]):\n    pass\n",
        "cdef:\n    thread()\n",
        "cdef foo[1, 1] bar\n",
        "cdef int foo(int bar(const char*)):pass\n",
        "cdef void f(char *argv[]): pass\n",
        'cdef void foo(): f"{a}"\n',
        'cdef void foo(): f"{a:02d}"\n',
        '{"a": 0, a: 1}\n',
        "{a.b: 0, a: 1}\n",
        "import foo\n\ndef bar():\n    import foo.bat\n",
        "def foo():\n    _ = bar()\n",
        "def create_multipliers():\n    return 3\n",
        'def create_multipliers():\n    return [f"a{3}" for i in range(10)]\n',
        "def create_multipliers():\n    return [lambda x: y for i in range(10)]\n",
        "lst = []\n"
        "for i in range(5):\n"
        "    def foo(a):\n"
        "        return b * a\n"
        "    lst.append(foo(a))\n",
        "{0: 1, 1: 2}\n",
        "{0, 1, 2}\n",
        "{0, f.b}\n",
        'mystring.rstrip("abc")\n',
        "mystring.rstrip(suffix)\n",
        "[(j for j in i) for i in items]\n",
        "for i, v in enumerate(values):\n    a == values[[i]]\n",
        "for i, v in enumerate(values):\n    pass\n    arr.extend(values[i])\n    pass\n",
        "for i, v in enumerate(values):\n    b = t[i]\n",
        "import numpy as np\n\n\ndef foo() -> np.ndarray:\n    pass\n",
        "current_notification = 3\n"
        "\n"
        "\n"
        "def set_fallback_notification(level):\n"
        "    global current_notification\n"
        "    current_notification = level\n",
        "def _read_string(GenericStream st, size_t n):\n"
        "    cdef object obj = st.read_string(n, &d_ptr, True)  "
        "# no-cython-lint\n",
        "def foo():\n    cdef int size\n    asarray(<char[:size]> foo)\n",
        'include "heap_watershed.pxi"\n',
        "import foo\n\n\ndef bar():\n    a: foo\n",
    ],
)
def test_noop(capsys: Any, src: str) -> None:
    ret = _main(src, "t.py", ext=".pyx")
    out, _ = capsys.readouterr()
    assert out == ""
    assert ret == 0


@pytest.mark.parametrize(
    "src",
    ["cdef int&& bar(): pass\n"],
)
@pytest.mark.skipif(
    tuple(Cython.__version__.split(".")) > ("3",),
    reason="invalid syntax in new Cython",
)
def test_noop_old_cython(capsys: Any, src: str) -> None:  # pragma: no cover
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == ""
    assert ret == 0


def test_config_file(tmpdir: Any, capsys: Any) -> None:
    # tmpdir will be the root of the project
    # tmpdir
    # ├── submodule1/
    # |   ├── submodule2/
    # |   |   └── a.pyx
    # |   └── b.pyx
    # ├── submodule3/
    # |   └── c.pyx
    # └── pyproject.toml
    config_file = os.path.join(tmpdir, "pyproject.toml")
    with open(config_file, "w") as fd:
        fd.write("[tool.cython-lint]\n")
        fd.write('ignore = ["E701"]\n')

    submodules = (
        os.path.join(tmpdir, "submodule1"),
        os.path.join(tmpdir, "submodule1", "submodule2"),
        os.path.join(tmpdir, "submodule3"),
    )
    for submodule in submodules:
        os.makedirs(submodule, exist_ok=True)

    cython_files = (
        os.path.join(submodules[0], "a.pyx"),
        os.path.join(submodules[1], "b.pyx"),
        os.path.join(submodules[2], "c.pyx"),
    )

    for file in cython_files:
        with open(file, "w", encoding="utf-8") as fd:
            fd.write("while True: pass\n")  # E701

    # config file is respected
    main(cython_files)
    out, _ = capsys.readouterr()
    assert out == ""

    # Command line arguments take precedence over config file
    main(['--ignore=""', *cython_files])
    out, _ = capsys.readouterr()

    for file in cython_files:
        assert f"{file}:1:11: E701 multiple statements on one line" in out

    # .cython-lint.toml takes precedence, even if empty
    config_file = os.path.join(tmpdir, ".cython-lint.toml")
    with open(config_file, "w") as fd:
        fd.write("[tool.cython-lint]\n")

    main(cython_files)
    out, _ = capsys.readouterr()

    for file in cython_files:
        assert f"{file}:1:11: E701 multiple statements on one line" in out


@pytest.mark.parametrize("config_file", ["pyproject.toml", "setup.cfg"])
def test_config_file_no_cython_lint(
    tmpdir: Any,
    capsys: Any,
    config_file: str,
) -> None:
    config_file = os.path.join(tmpdir, config_file)
    with open(config_file, "w") as fd:
        # config file with no cython-lint section
        fd.write("\n")

    file = os.path.join(tmpdir, "t.pyx")
    with open(file, "w", encoding="utf-8") as fd:
        fd.write("while True: pass\n")  # E701

    main([file])
    out, _ = capsys.readouterr()
    assert "t.pyx:1:11: E701 multiple statements on one line" in out


def test_no_config_file(tmpdir: Any, capsys: Any) -> None:
    file = os.path.join(tmpdir, "t.pyx")
    with open(file, "w", encoding="utf-8") as fd:
        fd.write("while True: pass\n")  # E701

    main(["--ignore=E701", file])
    out, _ = capsys.readouterr()
    assert out == ""

    main([file])
    out, _ = capsys.readouterr()
    assert "t.pyx:1:11: E701 multiple statements on one line" in out


def test_exported_imports(
    capsys: Any,
) -> None:
    src = 'import numpy\nimport polars\n__all__ = ["numpy", "os", 3]\n'
    ret = _main(src, "t.py", ext=".pyx", no_pycodestyle=True)
    out, _ = capsys.readouterr()
    expected = "t.py:2:8: 'polars' imported but unused\n"
    assert out == expected
    assert ret == 1
