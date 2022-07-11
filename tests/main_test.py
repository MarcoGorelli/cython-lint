from t import transform
import pytest

@pytest.mark.parametrize(
    "src, expected",
    [
        (
            'cdef mytype myvar\n' ,
            'myvar = 0\n'
        ),
        (
            'cdef mytype a, b\n' ,
            'a, b = 0, 0\n'
        ),
        (
            'cdef mytype a=0, b\n' ,
            'a, b = 0, 0\n'
        ),
        (
            'cdef public mytype myvar\n' ,
            'myvar = 0\n'
        ),
    ]
)
def test_cvardef(src, expected):
    result = transform(src, '')
    assert result == expected

