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

def replace_cdef(tokens, i):
    breakpoint()

def replace_coercion(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i-1
    while not (tokens[j].name == 'OP' and tokens[j].src == '<'):
        j -= 1
        tokens[j] = Token(name='PLACEHOLDER', src='')
    tokens[j] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not (tokens[j].name == 'OP' and tokens[j].src == '>'):
        j += 1
        tokens[j] = Token(name='PLACEHOLDER', src='')
    tokens[j] = Token(name='PLACEHOLDER', src='')


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
                    declaration.default.base_type.pos[1],
                    declaration.default.base_type.pos[2],
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
    replacements = collections.defaultdict(list)

    for node in body.stats:
        if isinstance(node, StatListNode):
            for name, line, col in visit_statlistnode(node, prev):
                replacements[line, col].append(name)
        elif isinstance(node, CVarDefNode):
            for name, line, col in visit_cvardefnode(node, prev):
                replacements[line, col].append(name)
        elif isinstance(node, CFuncDefNode):
            continue
            collect = visit_cfuncdefnode(node)
            for key, var in collect.items():
                collects[key].extend(var)
        prev = node

    for n, token in reversed_enumerate(tokens):
        key = (token.line, token.utf8_byte_offset)
        if key in replacements:
            for name in replacements.pop(key):
                if name == 'cvardef':
                    replace_cvardef(tokens, n)
                elif name == 'cdef':
                    replace_cdef(tokens, n)
                elif name == 'coercion':
                    replace_coercion(tokens, n)

    newsrc = tokens_to_src(tokens)

    breakpoint()

    # want to get to: ast.parse(code)

main()
