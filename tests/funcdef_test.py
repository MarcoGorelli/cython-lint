from t import transform
import pytest

@pytest.mark.parametrize(
    "src, expected",
    [
        pytest.param(
            'cdef extern from "foo.h":\n'
            '    ctypedef void* (*io_callback)(void *src, size_t nbytes, size_t *bytes_read,\n'
            '                    int *status, const char *encoding_errors)\n',
            'if True:\n'
            '    lambda: io_callback(src, nbytes, bytes_read,\n'
            '                    status, encoding_errors)\n',
            id='pointer to function',
        ),
    ]
)
def test_main(src, expected):
    result = transform(src, '')
    assert result == expected

