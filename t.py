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
from Cython.Compiler.Nodes import (
    StatListNode,
    CVarDefNode,
    CFuncDefNode,
)
from Cython.Compiler.ExprNodes import TypecastNode
import ast
import re

with open('algos.pyx') as fd:
    code = fd.read()

exclude = code.find('# generated from template')
if exclude != -1:
    code = code[:exclude]

tree = parse_from_strings('algos', code)

def replace_cvardef(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while tokens[j].name == 'UNIMPORTANT_WS':
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1


def visit_cvardefnode(node, *, block: bool = False):
    base_type = node.base_type
    yield (
        'cvardef',
        base_type.pos[1],
        base_type.pos[2],
    )
    coercions = []
    for declaration in node.declarators:
        if isinstance(declaration.default, TypecastNode):
            yield (
                    'coercion',
                    declaration.pos[1],
                    declaration.pos[1],
            )
    if not block:
        yield (
            'cdef',
            node.pos[1],
            node.pos[2],
        )


def visit_statlistnode(node, prev):
    collects = collections.defaultdict(list)
    cvarsdefs = False
    for _node in node.stats:
        if isinstance(_node, CVarDefNode):
            yield from visit_cvardefnode(_node, block=True)
            cvarsdefs = True
    if cvarsdefs:
        # we must be in a cdef block
        yield {
            'type': 'cdef',
            'line': prev.pos[1]+1,
            'endline': node.pos[1],
        }


def visit_cfuncdefnode(node):
    collects = {}
    # here, we need to put a cdef inside
    collects['cdefs'] = [{
        'line': node.pos[1],
        'endline': node.pos[1],
        'start': node.pos[2],
    }]
    return collects


import collections
from tokenize_rt import src_to_tokens, tokens_to_src, reversed_enumerate, Token

# ok, no, let's do something totally different
# let's have...let's do...
# some list of replacements

def main():
    tokens = src_to_tokens(code)
    body = tree.body
    prev = None
    collects = collections.defaultdict(list)
    replacements = {}

    for node in body.stats:
        if isinstance(node, StatListNode):
            for name, line, col in visit_statlistnode(node, prev):
                replacements[line, col] = name
        elif isinstance(node, CVarDefNode):
            for name, line, col in visit_cvardefnode(node, prev):
                replacements[line, col] = name
        elif isinstance(node, CFuncDefNode):
            continue
            collect = visit_cfuncdefnode(node)
            for key, var in collect.items():
                collects[key].extend(var)
        prev = node

    for n, token in reversed_enumerate(tokens):
        key = (token.line, token.utf8_byte_offset)
        if key in replacements:
            if replacements.pop(key) == 'cvardef':
                breakpoint()
                replace_cvardef(tokens, n)

    newsrc = tokens_to_src(tokens)
    breakpoint()

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

    for coercion in collects['coercions']:
        before = ''.join(lines[:coercion['line']])
        during = ''.join(lines[coercion['line']:coercion['endline']])
        after = ''.join(lines[coercion['endline']:])
        name = coercion['name']
        pycode = before + re.sub(fr'<\s*{name}\s*>', '', during) + after
        lines = pycode.splitlines(keepends=True)

    breakpoint()

    # want to get to: ast.parse(code)

main()
