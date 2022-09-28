import os
from typing import Any

import pytest

from cython_lint import _main

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
    ],
)
def test_assigned_unused(capsys: Any, src: str, expected: str) -> None:
    ret = _main(src, 't.py')
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
    ],
)
def test_noop(capsys: Any, src: str) -> None:
    ret = _main(src, 't.py')
    out, _ = capsys.readouterr()
    assert out == ''
    assert ret == 0
