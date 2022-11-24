import os
from typing import Any

import Cython
import pytest

from cython_lint.cython_lint import _main

INCLUDE_FILE_0 = os.path.join('tests', 'data', 'foo.pxi')
INCLUDE_FILE_1 = os.path.join('tests', 'data', 'bar.pxi')


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'cdef bint foo():\n'
            '    cdef int a\n',
            't.py:2:14: \'a\' defined but unused\n',
        ),
        (
            'cdef bint foo():\n'
            '    cdef int a\n'
            '    a = 3\n',
            't.py:3:5: \'a\' defined but unused\n',
        ),
    ],
)
def test_assigned_unused(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'cimport foo\n',
            't.py:1:9: \'foo\' imported but unused\n',
        ),
        (
            'from foo cimport bar\n',
            't.py:1:18: \'bar\' imported but unused\n',
        ),
        (
            'from foo import bar, bar2\n',
            't.py:1:17: \'bar\' imported but unused\n'
            't.py:1:22: \'bar2\' imported but unused\n',
        ),
        (
            'cimport quox\n'
            'include "foo.pxi"\n',
            't.py:1:9: \'quox\' imported but unused\n',
        ),
        (
            'import numpy as np\n',
            't.py:1:8: \'np\' imported but unused\n',
        ),
    ],
)
def test_imported_unused(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, 't.py')
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'cdef a, b\n',
            't.py:1:5: comma after base type in definition\n',
        ),
        (
            'cdef:\n'
            '    a, b\n',
            't.py:2:4: comma after base type in definition\n',
        ),
    ],
)
def test_misplaced_comma(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.skipif(
    tuple(Cython.__version__.split('.')) > ('3',),
    reason='invalid syntax in new Cython',
)
@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'cdef a[0, 1], b\n',
            't.py:1:5: comma after base type in definition\n',
        ),
        (
            'cdef a(0, 1), b\n',
            't.py:1:5: comma after base type in definition\n',
        ),
    ],
)
def test_misplaced_comma_old_cython(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'cdef void(): f"abc"\n',
            't.py:1:13: f-string without any placeholders\n',
        ),
    ],
)
def test_f_string_not_formatted(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'import foo\n'
            'foo()\n'
            'def bar(foo): pass\n',
            "t.py:3:9: 'foo' shadows global import on line 1 col 8\n",
        ),
        (
            'import foo\n'
            'foo()\n'
            'def bar(baz):\n'
            '    foo = 3\n'
            '    return foo\n',
            "t.py:4:5: 'foo' shadows global import on line 1 col 8\n",
        ),
    ],
)
def test_shadows_import(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    if tuple(Cython.__version__.split('.')) < ('3',):  # pragma: no cover
        # old Cython records the location slightly differently
        # no big deal
        expected = expected.replace('t.py:3:9', 't.py:3:12')
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            '{0, 0, 1}\n',
            't.py:1:2: Repeated element in set\n',
        ),
    ],
)
def test_repeated_set_element(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            '{0: 1, 0: 2}\n',
            't.py:1:1: dict key 0 repeated 2 times\n',
        ),
        (
            '{a: 1, a: 2}\n',
            't.py:1:1: dict key variable a repeated 2 times\n',
        ),
    ],
)
def test_repeated_dict_keys(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'def foo(a = []): pass\n',
            't.py:1:9: dangerous default value!\n',
        ),
        (
            'cdef void foo(int a = [])\n',
            't.py:1:15: dangerous default value!\n',
        ),
    ],
)
def test_dangerous_default(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'if (False,): pass\n',
            't.py:1:3: if-statement with tuple as condition is always true - '
            'perhaps remove comma?\n',
        ),
        (
            'if False:\n'
            '    pass\n'
            'elif (False,):\n'
            '    pass\n',
            't.py:3:5: if-statement with tuple as condition is always true - '
            'perhaps remove comma?\n',
        ),
        (
            'assert (False,)\n',
            't.py:1:0: assert statement with tuple as condition is always '
            'true - perhaps remove comma?\n',
        ),
    ],
)
def test_always_true(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'if 0 == 0: pass\n',
            't.py:1:6: Comparison between constants\n',
        ),
        (
            'if "0" == "0": pass\n',
            't.py:1:8: Comparison between constants\n',
        ),
    ],
)
def test_constants_comparison(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
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
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'def create_multipliers():\n'
            '    return [lambda x : i * x for i in range(5)]\n',
            't.py:2:12: Late binding closure! Careful '
            'https://docs.python-guide.org/writing/gotchas/'
            '#late-binding-closures\n',
        ),
        (
            'def create_multipliers():\n'
            '    return {i: lambda x : i * x for i in range(5)}\n',
            't.py:2:12: Late binding closure! Careful '
            'https://docs.python-guide.org/writing/gotchas/'
            '#late-binding-closures\n',
        ),
        (
            'lst = []\n'
            'for i in range(5):\n'
            '    def foo(a):\n'
            '        return i * a\n'
            '    lst.append(foo(a))\n',
            't.py:2:1: Late binding closure! Careful '
            'https://docs.python-guide.org/writing/gotchas/'
            '#late-binding-closures\n',
        ),
    ],
)
def test_late_binding_closure(
    capsys: Any,
    src: str,
    expected: str,
) -> None:
    ret = _main(src, 't.py', no_pycodestyle=True)
    out, _ = capsys.readouterr()
    assert out == expected
    assert ret == 1


def test_pycodestyle(tmpdir: Any, capsys: Any) -> None:
    file = os.path.join(tmpdir, 't.py')
    with open(file, 'w', encoding='utf-8') as fd:
        fd.write('while True: pass\n')
    with open(os.path.join(tmpdir, 'tox.ini'), 'w') as fd:
        fd.write('[pycodestyle]\nstatistics=True\n')
    src = ''
    ret = _main(src, file)
    out, _ = capsys.readouterr()
    expected = (
        f'{file}:1:11:  E701 multiple statements on one line (colon)\n'
    )
    assert out == expected
    assert ret == 1


@pytest.mark.parametrize(
    'src',
    [
        'cdef bint foo():\n'
        '    raise NotImplemen',
        'cdef bint foo():\n'
        '    cdef int i\n'
        '    for i in bar: pass\n',
        'cdef bint foo(a):\n'
        '   pass\n',
        'cdef bint foo(int a):\n'
        '   pass\n',
        'cdef bint foo(int *a):\n'
        '   pass\n',
        'cdef bint* foo(int a):\n'
        '   pass\n',
        'cdef bint foo():\n'
        '    cdef int i\n'
        '    for i, j in bar: pass\n',
        'cdef bint foo(object (*operation)(int64_t value, object right)):\n'
        '   pass\n',
        'class Foo: pass\n',
        'cdef class Foo: pass\n',
        'cdef bint foo(a):\n'
        '    bar(<int>a)\n',
        'cdef bint foo(a):\n'
        '    bar(i for i in a)\n',
        'cdef bint foo(a):\n'
        '    bar(lambda x: x)\n',
        'import int64_t\n'
        'ctypedef fused foo:\n'
        '    int64_t\n'
        '    int32_t\n',
        'import quox\n'
        f'include "{INCLUDE_FILE_0}"\n',
        'import quox\n'
        f'include "{INCLUDE_FILE_1}"\n',
        'from quox cimport *\n',
        'cdef inline bool _compare_records(\n'
        '    const FrontierRecord& left,\n'
        '    const FrontierRecord& right,\n'
        '):\n'
        '    return left.improvement < right.improvement\n',
        'from foo import bar\n'
        'ctypedef fused quox:\n'
        '    bar\n',
        'cimport cython\n'
        'ctypedef fused int_or_float:\n'
        '    cython.integral\n'
        '    cython.floating\n'
        '    signed long long\n',
        'cdef inline bool _compare_records(\n'
        '    double x[],\n'
        '):\n'
        '    return 1\n',
        'cimport foo.bar\n'
        'cdef bool quox():\n'
        '    foo.bar.ni()\n',
        'from foo cimport CTaskArgByReference\n'
        'cdef baz():\n'
        '    bar(new CTaskArgByReference())\n',
        'cdef PyObject** make_kind_names(list strings):\n'
        '    cdef PyObject** array = <PyObject**>mem.alloc(len(strings), sizeof(PyObject*))\n'  # noqa: E501
        '    cdef object name\n'
        '    for i, string in enumerate(strings):\n'
        '        name = intern(string)\n'
        '        Py_XINCREF(<PyObject*>name)\n'
        '        array[i] = <PyObject*>name\n'
        '    return <PyObject**>array\n',
        'def foo():\n'
        '    cdef int i\n'
        '    i = 3\n'
        '    print(i)\n',
        'def foo(int a[1][1]):\n'
        '    pass\n',
        'cdef:\n'
        '    thread()\n',
        'cdef foo[1, 1] bar\n',
        'cdef int foo(int bar(const char*)):pass\n',
        'cdef void f(char *argv[]): pass\n',
        'cdef void foo(): f"{a}"\n',
        'cdef void foo(): f"{a:02d}"\n',
        '{"a": 0, a: 1}\n',
        '{a.b: 0, a: 1}\n',
        'import foo\n'
        '\n'
        'def bar():\n'
        '    import foo.bat\n',
        'def foo():\n'
        '    _ = bar()\n',
        'def create_multipliers():\n'
        '    return 3\n',
        'def create_multipliers():\n'
        '    return [f"a{3}" for i in range(10)]\n',
        'def create_multipliers():\n'
        '    return [lambda x: y for i in range(10)]\n',
        'lst = []\n'
        'for i in range(5):\n'
        '    def foo(a):\n'
        '        return b * a\n'
        '    lst.append(foo(a))\n',
        '{0: 1, 1: 2}\n',
        '{0, 1, 2}\n',
        '{0, f.b}\n',
        'mystring.rstrip("abc")\n',
        'mystring.rstrip(suffix)\n',
        '[(j for j in i) for i in items]\n',
    ],
)
def test_noop(capsys: Any, src: str) -> None:
    ret = _main(src, 't.py')
    out, _ = capsys.readouterr()
    assert out == ''
    assert ret == 0


@pytest.mark.parametrize(
    'src', ['cdef int&& bar(): pass\n'],
)
@pytest.mark.skipif(
    tuple(Cython.__version__.split('.')) > ('3',),
    reason='invalid syntax in new Cython',
)
def test_noop_old_cython(capsys: Any, src: str) -> None:
    ret = _main(src, 't.py')
    out, _ = capsys.readouterr()
    assert out == ''
    assert ret == 0
