"""
maybe, we should just transform

cdef:
    type a = b

to

a = b

naah, don't mess with indents

will need to distinguish between inline cdefs
and cdef blocks then...
"""

from Cython.Compiler.TreeFragment import parse_from_strings
from Cython.Compiler.Nodes import StatListNode, CVarDefNode
import ast

with open('algos.pyx') as fd:
    code = fd.read()

exclude = code.find('# generated from template')
if exclude != -1:
    code = code[:exclude]

tree = parse_from_strings('algos', code)

def visit_cvardefnode(node, prev):
    base_type = node.base_type
    cvardefs = [{
        'line': base_type.pos[1]-1,
        'endline': base_type.pos[1],
        'start': base_type.pos[2],
        'end': node.declarators[0].pos[2],
    }]
    if prev is not None:
        cdefs = [{
            'line': prev.pos[1],
            'endline': node.pos[1],
        }]
    else:
        cdefs = []
    return cvardefs, cdefs


def visit_statlistnode(node, prev):
    cvardefs = []
    cdefs = []
    for _node in node.stats:
        if isinstance(_node, CVarDefNode):
            _cvardef, _cdef = visit_cvardefnode(_node, prev=None)
            cvardefs.extend(_cvardef)
            cdefs.extend(_cdef)
    if cvardefs:
        # we must be in a cdef block
        cdefs.append({
            'line': prev.pos[1]+1,
            'endline': node.pos[1],
        })
    return cvardefs, cdefs



body = tree.body
prev = None
cvardefs = []
cdefs = []
for node in body.stats:
    if isinstance(node, StatListNode):
        _cvardefs, _cdef = visit_statlistnode(node, prev)
        cvardefs.extend(_cvardefs)
        cdefs.extend(_cdef)
    elif isinstance(node, CVarDefNode):
        _cvardefs, _cdef = visit_cvardefnode(node, prev)
        cvardefs.extend(_cvardefs)
        cdefs.extend(_cdef)
    prev = node

pycode = code.replace('cimport', 'import')
lines = pycode.splitlines(keepends=True)

for cdef in cdefs:
    before = ''.join(lines[:cdef['line']])
    during = ''.join(lines[cdef['line']:cdef['endline']])
    after = ''.join(lines[cdef['endline']:])
    pycode = before + during.replace('cdef:', 'if True:') + after
    lines = pycode.splitlines(keepends=True)

for cvardef in cvardefs:
    before = ''.join(lines[:cvardef['line']])
    during = ''.join(lines[cvardef['line']:cvardef['endline']])
    after = ''.join(lines[cvardef['endline']:])
    pycode = before + during[:cvardef['start']] + during[cvardef['end']:] + after
    lines = pycode.splitlines(keepends=True)

breakpoint()

# want to get to: ast.parse(code)
