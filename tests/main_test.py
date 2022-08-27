import pytest

from cython_lint import main

@pytest.mark.parametrize(
    'src, expected',
    [
        (
            'cdef bint foo():\n'
            '    cdef int a\n',
            't.py:2:13: \'a\' defined but unused\n'
        ),
    ]
)
def test_assigned_unused(capsys, src, expected):
    ret = main(src, 't.py')
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
    ]
)
def test_noop(capsys, src):
    ret = main(src, 't.py')
    out, _ = capsys.readouterr()
    assert out == ''
    assert ret == 0