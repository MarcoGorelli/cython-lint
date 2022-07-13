from t import transform
import pytest

@pytest.mark.parametrize(
    "src, expected",
    [
        pytest.param(
            'ctypedef fused bar:\n'
            '    qux\n'
            '    quox\n',
            'class bar:\n'
            '    qux = 0\n'
            '    quox = 0\n',
            id='simple',
        ),
        pytest.param(
            'ctypedef fused bar:\n'
            '    ndarray[object, ndim=1]\n',
            'class bar:\n'
            '    ndarray = 0\n',
            id='memview old',
        ),
    ]
)
def test_cvardef_inline(src, expected):
    result = transform(src, '')
    assert result == expected

