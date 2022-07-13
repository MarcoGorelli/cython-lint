from t import transform
import pytest

@pytest.mark.parametrize(
    "src, expected",
    [
        pytest.param(
            'ctypedef enum Foo:\n'
            '    A\n'
            '    B\n',
            'class Foo:\n'
            '    A\n'
            '    B\n',
            id='ctypedef',
        ),
        pytest.param(
            'cdef enum Foo:\n'
            '    A\n'
            '    B\n',
            'class Foo:\n'
            '    A\n'
            '    B\n',
            id='cdef',
        ),
        pytest.param(
            'cdef extern from "foo.h":\n'
            '    enum: A\n',
            'if True:\n'
            '    lambda: A\n',
            id='just enum',
        ),
    ]
)
def test_main(src, expected):
    result = transform(src, '')
    assert result == expected

