"""
maybe, we should just transform

cdef:
    type a = b

to

a = b

naah, don't mess with indents

will need to distinguish between inline cdefs
and cdef blocks then...

still need statslist. check if they are all declarations, and if
so, look for a cdef before, and remove it
"""

from Cython.Compiler.TreeFragment import parse_from_strings
from Cython.Compiler.Nodes import (
    StatListNode,
    CVarDefNode,
    CFuncDefNode,
    CImportStatNode,
    CArgDeclNode,
    FromCImportStatNode,
)
from Cython.Compiler.ExprNodes import TypecastNode
import ast
import re

with open('algos.pyx') as fd:
    code = fd.read()

exclude = code.find('# generated from template')
if exclude != -1:
    code = code[:exclude]


def replace_cvardef(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while tokens[j].name == 'UNIMPORTANT_WS':
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1

def replace_cfuncdef(tokens, i):
    j = i
    while not (tokens[j].name == 'NAME' and tokens[j].src=='cdef'):
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j -= 1
    tokens[j] = Token(name='NAME', src='def')

def replace_cfuncarg(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not tokens[j].src.strip():  # TODO: tokenize whitespace?
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1

def replace_cdef(tokens, i):
    breakpoint()

def replace_cdefblock(tokens, i):
    j = i-1
    while not (tokens[j].name=='NAME' and tokens[j].src=='cdef'):
        j -= 1
    tokens[j] = Token(name='NAME', src='if True')

def replace_typecast(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not (tokens[j].name == 'OP' and tokens[j].src == '>'):
        j += 1
    for _i in range(i, j+1):
        tokens[_i] = Token(name='PLACEHOLDER', src='')


def replace_cargdecl(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not tokens[j].src.strip():  # TODO: tokenize whitespace?
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1


def replace_fromcimportstat(tokens, i):
    j = i+1
    while not (tokens[j].name=='NAME' and tokens[j].src=='cimport'):
        j += 1
    tokens[j] = Token(name='NAME', src='import')

def replace_cimportstat(tokens, i):
    j = i-1
    while not (tokens[j].name=='NAME' and tokens[j].src=='cimport'):
        j -= 1
    tokens[j] = Token(name='NAME', src='import')


def visit_cvardefnode(node):
    base_type = node.base_type
    yield (
        'cvardef',
        base_type.pos[1],
        base_type.pos[2],
    )

def visit_typecastnode(node):
    yield (
            'typecast',
            node.pos[1],
            node.pos[2],
    )


def visit_cfuncdefnode(node):
    yield (
        'cfuncdef',
        node.base_type.pos[1],
        node.base_type.pos[2],
    )

def visit_cargdeclnode(node):
    yield (
        'cargdecl',
        node.pos[1],
        node.pos[2],
    )

def visit_statlistnode(node):
    if all(isinstance(child, CVarDefNode) for child in node.stats):
        yield (
            'cdefblock',
            node.pos[1],
            node.pos[2],
        )

def visit_fromcimportstatnode(node):
    yield (
        'fromcimport',
        node.pos[1],
        node.pos[2],
    )

def visit_cimportstatnode(node):
    yield (
        'cimport',
        node.pos[1],
        node.pos[2],
    )


import collections
from tokenize_rt import src_to_tokens, tokens_to_src, reversed_enumerate, Token

# ok, no, let's do something totally different
# let's have...let's do...
# some list of replacements

def main():
    tree = parse_from_strings('algos', code)
    replacements = traverse(tree)
    tokens = src_to_tokens(code)

    for n, token in reversed_enumerate(tokens):
        key = (token.line, token.utf8_byte_offset)
        if key in replacements:
            for name in replacements.pop(key):
                if name == 'cvardef':
                    replace_cvardef(tokens, n)
                elif name == 'cdef':
                    replace_cdef(tokens, n)
                elif name == 'cdefblock':
                    replace_cdefblock(tokens, n)
                elif name == 'typecast':
                    replace_typecast(tokens, n)
                elif name == 'cfuncdef':
                    replace_cfuncdef(tokens, n)
                elif name == 'cfuncarg':
                    replace_cfuncarg(tokens, n)
                elif name == 'cargdecl':
                    replace_cargdecl(tokens, n)
                elif name == 'fromcimport':
                    replace_fromcimportstat(tokens, n)
                elif name == 'cimport':
                    replace_cimportstat(tokens, n)

    newsrc = tokens_to_src(tokens)
    breakpoint()


    # want to get to: ast.parse(code)

def traverse(tree):
    # check if child isn't []
    nodes = [tree]
    replacements = collections.defaultdict(list)

    funcs = {
        'CVarDefNode': visit_cvardefnode,
        'CFuncDefNode': visit_cfuncdefnode,
        'TypecastNode': visit_typecastnode,
        'CArgDeclNode': visit_cargdeclnode,
        'FromCImportStatNode': visit_fromcimportstatnode,
        'StatListNode': visit_statlistnode,
        'CImportStatNode': visit_cimportstatnode,
    }

    breakpoint()
    while nodes:
        node = nodes.pop()
        if node is None:
            continue

        func = funcs.get(type(node).__name__)
        if func is not None:
            iterator = func(node)
            if iterator is not None:
                for name, line, col in iterator:
                    replacements[line, col].append(name)

        for attr in node.child_attrs:
            child = getattr(node, attr)
            if isinstance(child, list):
                nodes.extend(child)
            else:
                nodes.append(child)
    return replacements

main()
