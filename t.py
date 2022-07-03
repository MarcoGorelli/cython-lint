"""
list_of_arrays: list
gets transformed wrongly to 
: list

also, missing nogil
"""

from Cython.Compiler.TreeFragment import parse_from_strings
from Cython.Compiler.Nodes import (
    StatListNode,
    CVarDefNode,
    CFuncDefNode,
    CImportStatNode,
    CArgDeclNode,
    FromCImportStatNode,
    CSimpleBaseTypeNode,
    MemoryViewSliceTypeNode,
    CPtrDeclaratorNode,
)
from Cython.Compiler.ExprNodes import TypecastNode, AmpersandNode
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
    j = i-1
    while not tokens[j].src.strip():
        j -= 1
    k = j-1
    while not tokens[k].src.strip():
        k -= 1
    if (
        tokens[j].name == 'OP'
        and tokens[j].src == ':'
        and tokens[k].name == 'NAME'
        and tokens[k].src == 'cdef'
    ):
        tokens[k] = Token(name='NAME', src='if True')
    elif tokens[j].name == 'NAME' and tokens[j].src == 'cdef':
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j = j+1
        while not tokens[j].src.strip():
            tokens[j] = Token(name='PLACEHOLDER', src='')
            j += 1

def replace_cfuncdef(tokens, i):
    if (tokens[i].name == 'NAME' and tokens[i].src == 'inline'):
        tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not tokens[j].src.strip():  # TODO: tokenize whitespace?
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1
    j = i
    while not (tokens[j].name == 'NAME' and tokens[j].src in ('cdef', 'cpdef')):
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

def replace_templatedtype(tokens, i):
    j = i
    while not (tokens[j].name=='OP' and tokens[j].src==']'):
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1
    tokens[j] = Token(name='PLACEHOLDER', src='')
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1

def replace_csimplebasetype(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1

def replace_cconsttypenode(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1

def replace_memoryviewslicetypenode(tokens, i):
    j = i
    while not (tokens[j].name=='OP' and tokens[j].src=='['):
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j -= 1
    tokens[j] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not (tokens[j].name=='OP' and tokens[j].src==']'):
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1
    tokens[j] = Token(name='PLACEHOLDER', src='')
    j = j+1
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1

def replace_ampersandnode(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')

def replace_cptrdeclaratornode(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')

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

def visit_templatedtypenode(node):
    # might need to unpack a whole load more things here
    yield (
        'templatedtype',
        node.base_type_node.pos[1],
        node.base_type_node.pos[2],
    )

def visit_csimplebasetypenode(node):
    yield (
        'csimplebasetype',
        node.pos[1],
        node.pos[2],
    )

def visit_cconsttypenode(node):
    yield (
        'cconsttype',
        node.pos[1],
        node.pos[2],
    )

def visit_memoryviewslicetypenode(node):
    yield (
        'memoryviewslicetype',
        node.pos[1],
        node.pos[2],
    )

def visit_ampersandnode(node):
    yield (
        'ampersand',
        node.pos[1],
        node.pos[2],
    )

def visit_cptrdeclaratornode(node):
    yield (
        'cptrdeclarator',
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
                elif name == 'typecast':
                    replace_typecast(tokens, n)
                elif name == 'cfuncdef':
                    replace_cfuncdef(tokens, n)
                elif name == 'cfuncarg':
                    replace_cfuncarg(tokens, n)
                elif name == 'fromcimport':
                    replace_fromcimportstat(tokens, n)
                elif name == 'cimport':
                    replace_cimportstat(tokens, n)
                elif name == 'templatedtype':
                    replace_templatedtype(tokens, n)
                elif name == 'csimplebasetype':
                    replace_csimplebasetype(tokens, n)
                elif name == 'cconsttypenode':
                    replace_cconsttypenode(tokens, n)
                elif name == 'memoryviewslicetype':
                    replace_memoryviewslicetypenode(tokens, n)
                elif name == 'ampersand':
                    replace_ampersandnode(tokens, n)
                elif name == 'cptrdeclarator':
                    replace_cptrdeclaratornode(tokens, n)

    newsrc = tokens_to_src(tokens)
    print(newsrc)
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
        'FromCImportStatNode': visit_fromcimportstatnode,
        'StatListNode': visit_statlistnode,
        'CImportStatNode': visit_cimportstatnode,
        'TemplatedTypeNode': visit_templatedtypenode,
        'CSimpleBaseTypeNode': visit_csimplebasetypenode,
        'CConstTypeNode': visit_cconsttypenode,
        'MemoryViewSliceTypeNode': visit_memoryviewslicetypenode,
        'AmpersandNode': visit_ampersandnode,
        'CPtrDeclaratorNode': visit_cptrdeclaratornode,
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
