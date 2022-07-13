from t import transform
import pytest

@pytest.mark.parametrize(
    "src, expected",
    [
        pytest.param(
            'cdef extern from "foo.h":\n'
            '    int a\n',
            'if True:\n'
            '    a = 0\n',
            id='extern cvardef',
        ),
        pytest.param(
            'cdef extern from "foo.h":\n'
            '    int a()\n',
            'if True:\n'
            '    a = 0\n',
            id='extern cvardef',
        ),
        pytest.param(
            'cdef extern from "foo.h":\n'
            '    # foo\n'
            '    int a()\n',
            'if True:\n'
            '    # foo\n'
            '    a = 0\n',
            id='with comment',
        ),
        pytest.param(
            'cdef extern from "numpy/arrayobject.h":\n'
            '    ctypedef class numpy.dtype [object PyArray_Descr]:\n'
            '        cdef:\n'
            '            int type_num\n',
            'if True:\n'
            '    class dtype:\n'
            '        if True:\n'
            '            type_num = 0\n',
            id='class []',
        ),
        # todo: add a case where module name is more nested
        pytest.param(
            'cdef extern from "numpy/arrayobject.h":\n'
            '    ctypedef class dtype [object PyArray_Descr]:\n'
            '        cdef:\n'
            '            int type_num\n',
            'if True:\n'
            '    class dtype:\n'
            '        if True:\n'
            '            type_num = 0\n',
            id='class []',
        ),
    ]
)
def test_main(src, expected):
    result = transform(src, '')
    assert result == expected

