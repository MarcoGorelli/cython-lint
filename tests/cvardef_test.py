from t import transform
import pytest

@pytest.mark.parametrize(
    "src, expected",
    [
        pytest.param(
            'cdef mytype myvar\n' ,
            'myvar = 0\n',
            id='simple',
        ),
        pytest.param(
            'cdef mytype a, b\n' ,
            'a, b = 0, 0\n',
            id='multiple',
        ),
        pytest.param(
            'cdef mytype a=0, b\n' ,
            'a, b = 0, 0\n',
            id='multiple with initial',
        ),
        pytest.param(
            'cdef public mytype myvar\n' ,
            'myvar = 0\n',
            id='public',
        ),
        pytest.param(
            'cdef mytype[::1] myvar\n' ,
            'myvar = 0\n',
            id='memview',
        ),
        pytest.param(
            'cdef mytype *myvar\n' ,
            'myvar = 0\n',
            id='pointer',
        ),
        pytest.param(
            'cdef class Foo:\n'
            '    cdef public mytype a\n',
            'class Foo:\n'
            '    a = 0\n',
            id='public in class',
        ),
    ]
)
def test_cvardef_inline(src, expected):
    result = transform(src, '')
    assert result == expected

@pytest.mark.parametrize(
    "src, expected",
    [
        (
            'cdef:\n    mytype a\n    mytype b\n',
            'if True:\n    a = 0\n    b = 0\n',
        ),
    ]
)
def test_cvardef_block(src, expected):
    result = transform(src, '')
    assert result == expected

