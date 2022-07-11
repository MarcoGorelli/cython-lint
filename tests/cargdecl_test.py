from t import transform
import pytest

@pytest.mark.parametrize(
    "src, expected",
    [
        pytest.param(
            'def foo(mytype a): pass\n',
            'def foo(a): pass\n',
            id='simple',
        ),
        pytest.param(
            'def foo(a: mytype): pass\n',
            'def foo(a: mytype): pass\n',
            id='Python type annotation',
        ),
    ]
)
def test_cvardef_inline(src, expected):
    result = transform(src, '')
    assert result == expected

