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
        pytest.param(
            'cpdef bar foo(self, ndarray values) except -1: pass\n',
            'def foo(self, values): pass\n',
            id='except -1',
        ),
        pytest.param(
            'cpdef (bar, quox) foo(self, ndarray values): pass\n',
            'def foo(self, values): pass\n',
            id='tuple return',
        ),
        pytest.param(
            'cpdef foo(\n'
            '    a,\n'
                'foo b=0,\n'
            '): pass\n',
            'def foo(\n'
            '    a,\n'
                'b=0,\n'
            '): pass\n',
            id='multiline',
        ),
        pytest.param(
            'def foo(mytype bar not None): pass\n',
            'def foo(bar): pass\n',
            id='not None',
        ),
        pytest.param(
            'cdef inline int _try_double_nogil(parser_t *parser,\n'
            '                     float64_t (*double_converter)(\n'
            '                         const char *, char **, char,\n'
            '                         char, char, int, int *, int *) nogil,\n'
            '                     int *na_count) nogil: pass\n',
            'def _try_double_nogil(parser,\n'
            '                     double_converter\n'
            '                         \n'
            '                         ,\n'
            '                     na_count): pass\n',
        ),
    ]
)
def test_cvardef_inline(src, expected):
    result = transform(src, '')
    assert result == expected

