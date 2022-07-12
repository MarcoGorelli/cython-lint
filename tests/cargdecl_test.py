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
        pytest.param(
            'def foo(mytype[type_, ndim=2] bar): pass\n',
            'def foo(bar): pass\n',
            id='Memory view old',
        ),
        pytest.param(
            'cpdef bar foo(self, ndarray values): pass\n',
            'def foo(self, values): pass\n',
            id='self and other arg',
        ),
    ]
)
def test_cvardef_inline(src, expected):
    result = transform(src, '')
    assert result == expected

