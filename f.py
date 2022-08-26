"""
really, just need:
- walk function
- whatever, we'll figure this out
"""
from __future__ import annotations

from Cython.Compiler.Errors import CompileError
import sys
import os
import argparse
import subprocess
import tempfile
from Cython.Compiler.TreeFragment import parse_from_strings
from Cython.Compiler.Nodes import (
    StatListNode,
    PyClassDefNode,
    CVarDefNode,
    SingleAssignmentNode,
    CFuncDefNode,
    CImportStatNode,
    FromCImportStatNode,
    CSimpleBaseTypeNode,
    MemoryViewSliceTypeNode,
    CPtrDeclaratorNode,
    GILStatNode,
    FusedTypeNode,
    CClassDefNode,
    CArgDeclNode,
    CNameDeclaratorNode,
    CEnumDefNode,
    CTupleBaseTypeNode,
    CFuncDeclaratorNode,
    TemplatedTypeNode,
    CStructOrUnionDefNode,
    CTypeDefNode,
    CArrayDeclaratorNode,
)
from Cython.Compiler.ExprNodes import TypecastNode, AmpersandNode, NameNode, ImportNode
import ast
import re

class CythonLintError(Exception):
    pass

import collections
from tokenize_rt import src_to_tokens, tokens_to_src, reversed_enumerate, Token

# ok, no, let's do something totally different
# let's have...let's do...
# some list of replacements

BUILTIN_NAMES = frozenset((
    'hasattr',
    'set',
    'list',
    'abs',
    'str',
    'isinstance',
    'self',
    'dict',
    'len',
    'min',
    'max',
    'range',
    'TypeError',
    'AssertionError',
    'object',
 'SystemExit',
 'KeyboardInterrupt',
 'GeneratorExit',
 'Exception',
      'StopIteration',
      'StopAsyncIteration',
      'ArithmeticError',
      'FloatingPointError',
      'OverflowError',
      'ZeroDivisionError',
      'AssertionError',
      'AttributeError',
      'BufferError',
      'EOFError',
      'ImportError',
      'ModuleNotFoundError',
      'LookupError',
      'IndexError',
      'KeyError',
      'MemoryError',
      'NameError',
      'UnboundLocalError',
      'OSError',
      'BlockingIOError',
      'ChildProcessError',
      'ConnectionError',
      'BrokenPipeError',
      'ConnectionAbortedError',
      'ConnectionRefusedError',
      'ConnectionResetError',
      'FileExistsError',
      'FileNotFoundError',
      'InterruptedError',
      'IsADirectoryError',
      'NotADirectoryError',
      'PermissionError',
      'ProcessLookupError',
      'TimeoutError',
      'ReferenceError',
      'RuntimeError',
      'NotImplementedError',
      'RecursionError',
      'SyntaxError',
      'IndentationError',
      'TabError',
      'SystemError',
      'TypeError',
      'ValueError',
      'UnicodeError',
      'UnicodeDecodeError',
      'UnicodeEncodeError',
      'UnicodeTranslateError',
           'DeprecationWarning',
           'PendingDeprecationWarning',
           'RuntimeWarning',
           'SyntaxWarning',
           'UserWarning',
           'FutureWarning',
           'ImportWarning',
           'UnicodeWarning',
           'BytesWarning',
           'EncodingWarning',
           'ResourceWarning',
))

def tokenize_replacements(tokens):
    in_cdef = False
    cdef = []
    colon = []
    for i, token in enumerate(tokens):
        if in_cdef and not token.src.strip():
            continue
        elif in_cdef and (
                token.name == 'NAME'
                and token.src in ('public', 'readonly', 'from', 'extern')
            ):
            pass
        elif in_cdef and (token.name == 'STRING' and token.src.strip('\'"').strip().endswith('.h')):
            pass
        elif in_cdef and token.name == 'OP' and token.src == ':':
            colon.append(i)
            in_cdef = False
        elif token.name == 'NAME' and token.src == 'cdef':
            cdef.append(i)
            in_cdef = True
        else:
            # This wasn't a cdef we were looking for.
            if in_cdef:
                cdef.pop()
                in_cdef = False
    assert len(cdef) == len(colon)
    for start, end in zip(cdef, colon):
        tokens[start] = tokens[start]._replace(src='if True')
        for j in range(start+1, end):
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)


def visit_funcdef(node, imported_names, globals, filename):
    children = list(traverse(node))[1:]
    names = [(i.name, *i.pos[1:]) for i in children if isinstance(i, NameNode) if (i.name and i.name not in BUILTIN_NAMES)]
    defs = [(i.name, *i.pos[1:]) for i in children if isinstance(i, CNameDeclaratorNode) if i.name]
    args = []
    for i in children:
        if isinstance(i, CArgDeclNode):
            if isinstance(i.declarator, CNameDeclaratorNode):
                if i.declarator.name:
                    args.append((i.declarator.name, *i.declarator.pos[1:]))
                elif isinstance(i.base_type, CNameDeclaratorNode):
                    args.append((i.base_type.name, *i.base_type.pos[1:]))
            elif isinstance(i.declarator, CPtrDeclaratorNode):
                args.append((i.declarator.base.name, *i.declarator.base.pos[1:]))
    func_name = node.declarator.base.name

    for _def in defs:
        if _def[0] not in [i[0] for i in names] and _def[0] != func_name:
            print(f'{filename}:{_def[1]}:{_def[2]}: Name {_def[0]} defined but unused')
    
    defs = [*defs, *imported_names, *globals]
    names = [*names, *args]
    names = sorted(names, key=lambda x: (x[1], x[2]))
    defs = sorted(defs, key=lambda x: (x[1], x[2]))

    # so, then, we need a list of things we've imported
    for name in names:
        _name, _line, _col = name
        _def = [i for i in defs if i[0] == _name]
        if not _def or (_def[0][1:] > (_line, _col)):
            print(f'{filename}:{_line}:{_col}: Name {_name} undefined')

def transform(code, filename):
    tokens = src_to_tokens(code)
    exclude_lines = set()
    for token in tokens:
        if token.name == 'NAME' and token.src == 'include':
            exclude_lines.add(token.line)
    # compile time defs (not in ast?)
    compile_time_defs = []
    for i, token in enumerate(tokens):
        if token.name == 'NAME' and token.src == 'DEF':
            tokens[i] = Token(name='PLACEHOLDER', src='', line=tokens[i].line, utf8_byte_offset=tokens[i].utf8_byte_offset)
            j = i+1
            while not tokens[j].src.strip():
                tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
                j += 1
    code = tokens_to_src(tokens)
    code = ''.join([line for i, line in enumerate(code.splitlines(keepends=True), start=1) if i not in exclude_lines])
    try:
        tree = parse_from_strings(filename, code)
    except CompileError:
        raise CythonLintError(f'Could not parse file {filename}')
    nodes = list(traverse(tree))

    imported_names = []
    globals = []
    # find imported variables
    # add more here, we're on to something
    for node in nodes:
        if isinstance(node, FromCImportStatNode):
            for imp in node.imported_names:
                imported_names.append((imp[1], *node.pos[1:]))
        elif isinstance(node, CImportStatNode):
            imported_names.append((node.as_name or node.module_name, *node.pos[1:]))
        elif (
                isinstance(node, SingleAssignmentNode)
                and isinstance(node.rhs, ImportNode)):
            imported_names.append((node.lhs.name, *node.lhs.pos[1:]))
    imported_names = sorted(imported_names, key=lambda x: (x[1], x[2]))

    # find global variables
    if isinstance(tree.body, StatListNode):
        stats = tree.body.stats
    else:
        stats = [tree.body]
    for node in stats:
        if isinstance(node, StatListNode):
            if all(isinstance(i, CVarDefNode) for i in node.stats):
                for i in node.stats:
                    globals.extend([(decl.name, *decl.pos[1:]) for decl in i.declarators])
        elif isinstance(node, CVarDefNode):
            globals.extend([(decl.name, *decl.pos[1:]) for decl in node.declarators])
        elif isinstance(node, CFuncDefNode):
            globals.append((node.declarator.base.name, *node.declarator.base.pos[1:]))
        elif isinstance(node, PyClassDefNode):
            globals.append((node.name, *node.pos[1:]))

    for node in nodes:
        if isinstance(node, CFuncDefNode):
            visit_funcdef(node, imported_names, globals, filename)


def main(code, filename, append_config):
    newsrc = transform(code, filename)

def get_name(node: NameNode):
    if isinstance(node, NameNode):
        return node.name


def traverse(tree):
    nodes = [tree]

    while nodes:
        node = nodes.pop()
        if node is None:
            continue

        child_attrs = node.child_attrs
        if 'declarator' not in child_attrs and hasattr(node, 'declarator'):
            # noticed this in size(int*)
            # bug?
            child_attrs.append('declarator')
        if isinstance(node, FusedTypeNode):
            # bug?
            child_attrs.append('types')
        for attr in child_attrs:
            child = getattr(node, attr)
            if isinstance(child, list):
                nodes.extend(child)
            else:
                nodes.append(child)
        yield node

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--append-config', required=False)
    args = parser.parse_args()
    for path in args.paths:
        with open(path, encoding='utf-8') as fd:
            content = fd.read()
        try:
            main(content, path, args.append_config)
        except CythonLintError:
            continue