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
            'start': node.pos[2],
        }]
    else:
        cdefs = []
    return {
            'cvardefs': cvardefs,
            'cdefs': cdefs,
    }


def visit_statlistnode(node, prev):
    collects = collections.defaultdict(list)
    for _node in node.stats:
        if isinstance(_node, CVarDefNode):
            collect = visit_cvardefnode(_node, prev=None)
            for key, var in collect.items():
                collects[key].extend(var)
    if collects['cvardefs']:
        # we must be in a cdef block
        collects['cdefblocks'] = [{
            'line': prev.pos[1]+1,
            'endline': node.pos[1],
        }]
    return collects


import collections

def main():
    body = tree.body
    prev = None
    collects = collections.defaultdict(list)

    for node in body.stats:
        if isinstance(node, StatListNode):
            collect = visit_statlistnode(node, prev)
            for key, var in collect.items():
                collects[key].extend(var)
        elif isinstance(node, CVarDefNode):
            collect = visit_cvardefnode(node, prev)
            for key, var in collect.items():
                collects[key].extend(var)
        prev = node

    pycode = code.replace('cimport', 'import')
    lines = pycode.splitlines(keepends=True)

    for cdef in collects['cdefblocks']:
        before = ''.join(lines[:cdef['line']])
        during = ''.join(lines[cdef['line']:cdef['endline']])
        after = ''.join(lines[cdef['endline']:])
        pycode = before + during.replace('cdef:', 'if True:') + after
        lines = pycode.splitlines(keepends=True)

    for cvardef in collects['cvardefs']:
        before = ''.join(lines[:cvardef['line']])
        during = ''.join(lines[cvardef['line']:cvardef['endline']])
        after = ''.join(lines[cvardef['endline']:])
        pycode = before + during[:cvardef['start']] + during[cvardef['end']:] + after
        lines = pycode.splitlines(keepends=True)

    for cdef in collects['cdefs']:
        before = ''.join(lines[:cdef['line']])
        during = ''.join(lines[cdef['line']:cdef['endline']])
        after = ''.join(lines[cdef['endline']:])
        pycode = before + during[cdef['start']:] + after
        lines = pycode.splitlines(keepends=True)

    breakpoint()

    # want to get to: ast.parse(code)

main()
