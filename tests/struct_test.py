from t import transform
import pytest

@pytest.mark.parametrize(
    "src, expected",
    [
        # todo add that crazy one that broke it
        pytest.param(
            'ctypedef struct foo:\n'
            '    void *source\n',
            'class foo:\n'
            '    source = 0\n',
            id='simple',
        ),
    ]
)
def test_main(src, expected):
    result = transform(src, '')
    assert result == expected

