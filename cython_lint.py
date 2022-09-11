from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Iterator
from typing import NamedTuple
from typing import NoReturn
from typing import Sequence

from Cython import Tempita
from Cython.Compiler.Errors import CompileError
from Cython.Compiler.ExprNodes import GeneratorExpressionNode
from Cython.Compiler.ExprNodes import ImportNode
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.ExprNodes import TypecastNode
from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Nodes import CArgDeclNode
from Cython.Compiler.Nodes import CArrayDeclaratorNode
from Cython.Compiler.Nodes import CClassDefNode
from Cython.Compiler.Nodes import CFuncDeclaratorNode
from Cython.Compiler.Nodes import CFuncDefNode
from Cython.Compiler.Nodes import CImportStatNode
from Cython.Compiler.Nodes import CNameDeclaratorNode
from Cython.Compiler.Nodes import CPtrDeclaratorNode
from Cython.Compiler.Nodes import CReferenceDeclaratorNode
from Cython.Compiler.Nodes import CSimpleBaseTypeNode
from Cython.Compiler.Nodes import ForInStatNode
from Cython.Compiler.Nodes import FromCImportStatNode
from Cython.Compiler.Nodes import FromImportStatNode
from Cython.Compiler.Nodes import FusedTypeNode
from Cython.Compiler.Nodes import Node
from Cython.Compiler.Nodes import SingleAssignmentNode
from Cython.Compiler.TreeFragment import parse_from_strings
from tokenize_rt import src_to_tokens
from tokenize_rt import tokens_to_src


class Token(NamedTuple):
    name: str
    lineno: int
    colno: int


class CythonLintError(Exception):
    pass


def err_msg(node: Node, expected: str) -> NoReturn:
    raise CythonLintError(
        f'Unexpected error, please report bug. '
        f'Expected {expected}, got {node}\n'
        f'{node}\n',
    )


def visit_funcdef(
    node: CFuncDefNode,
    filename: str,
    lines: Sequence[str],
) -> int:
    ret = 0

    children = list(traverse(node))[1:]

    # e.g. cdef int a = 3
    defs = [
        Token(i.name, *i.pos[1:])
        for i in children
        if isinstance(i, CNameDeclaratorNode)
        if i.name
    ]
    # e.g. a = 3
    simple_assignments = [
        Token(_child.lhs.name, *_child.lhs.pos[1:])
        for _child in children
        if isinstance(_child, SingleAssignmentNode)
        and isinstance(_child.lhs, NameNode)
    ]
    defs = [*defs, *simple_assignments]

    names = [
        Token(_child.name, *_child.pos[1:])
        for _child in children
        if isinstance(_child, NameNode)
    ]

    args: list[Token] = []
    for _child in children:
        if isinstance(_child, CArgDeclNode):
            args.extend(_args_from_cargdecl(_child))

    if isinstance(node.declarator.base, CNameDeclaratorNode):
        # e.g. cdef int foo()
        func_name = node.declarator.base.name
    elif isinstance(node.declarator.base, CFuncDeclaratorNode):
        # e.g. cdef int* foo()
        if isinstance(node.declarator.base.base, CNameDeclaratorNode):
            func_name = node.declarator.base.base.name
        else:  # pragma: no cover
            err_msg(node.declarator.base.base, 'CNameDeclaratorNode')
    else:  # pragma: no cover
        err_msg(
            node.declarator.base,
            'CNameDeclaratorNode or CFuncDeclaratorNode',
        )

    for _def in defs:
        # we don't report on unused function args
        if (
            _def[0] not in [_name[0] for _name in names if _def != _name]
            and _def[0] != func_name
            and _def[0] not in [i[0] for i in args]
        ) and '# no-lint' not in lines[_def[1] - 1]:
            print(
                f'{filename}:{_def[1]}:{_def[2]+1}: '
                f"'{_def[0]}' defined but unused",
            )
            ret = 1
    return ret


def _name_from_cptrdeclarator(
    node: CPtrDeclaratorNode | CNameDeclaratorNode,
) -> CNameDeclaratorNode:
    while isinstance(node, CPtrDeclaratorNode):
        node = node.base
    if isinstance(node, CNameDeclaratorNode):
        return node
    err_msg(node, 'CNameDeclaratorNode')  # pragma: no cover


def _args_from_cargdecl(node: CArgDeclNode) -> Iterator[Token]:
    if isinstance(node.declarator, (CNameDeclaratorNode, CPtrDeclaratorNode)):
        # e.g. foo(int a), foo(int* a)
        _decl = _name_from_cptrdeclarator(node.declarator)
        if _decl.name:
            yield Token(_decl.name, *_decl.pos[1:])
        elif isinstance(
            node.base_type,
            (CNameDeclaratorNode, CSimpleBaseTypeNode),
        ):
            # e.g. foo(a)
            yield Token(node.base_type.name, *node.base_type.pos[1:])
        else:  # pragma: no cover
            err_msg(
                node.base_type,
                'CNameDeclaratorNode or CSimpleBaseTypeNode',
            )
    elif isinstance(node.declarator, CFuncDeclaratorNode):
        # e.g. cdef foo(object (*operation)(int64_t value))
        for _arg in node.declarator.args:
            yield from _args_from_cargdecl(_arg)
        _base = _name_from_cptrdeclarator(node.declarator.base)
        yield Token(_base.name, *_base.pos[1:])
    elif isinstance(
        node.declarator,
        (CReferenceDeclaratorNode, CArrayDeclaratorNode),
    ):
        # e.g. cdef foo(vector[FrontierRecord]& frontier)
        # e.g. cdef foo(double x[])
        if isinstance(node.declarator.base, CNameDeclaratorNode):
            yield Token(
                node.declarator.base.name,
                *node.declarator.base.pos[1:],
            )
        else:  # pragma: no cover
            err_msg(node.declarator.base, 'CNameDeclarator')
    else:  # pragma: no cover
        err_msg(
            node.declarator,
            'CNameDeclarator, '
            'CPtrDeclarator, '
            'CFuncDeclarator, or '
            'CReferenceDeclarator',
        )


def _traverse_file(
        code: str,
        filename: str,
        lines: Sequence[str],
        *,
        skip_check: bool = False,
) -> tuple[list[Token], list[Token], int]:
    """
    skip_check: only for when traversing an included file
    """
    ret = 0
    tree = parse_from_strings(filename, code)
    nodes = traverse(tree)
    imported_names: list[Token] = []
    names: list[Token] = []
    for node in nodes:
        if isinstance(node, FromCImportStatNode):
            imported_names.extend(
                Token(imp[2] or imp[1], *imp[0][1:])
                for imp in node.imported_names
            )

        elif isinstance(node, CImportStatNode):
            imported_names.append(
                Token(node.as_name or node.module_name, *node.pos[1:]),
            )
        elif isinstance(node, SingleAssignmentNode) and isinstance(
            node.rhs, ImportNode,
        ):
            # e.g. import numpy as np
            imported_names.append(Token(node.lhs.name, *node.lhs.pos[1:]))
        elif isinstance(node, FromImportStatNode):
            # from numpy import array
            imported_names.extend(
                Token(imp[1].name, *imp[1].pos[1:]) for imp in node.items
            )

        if isinstance(node, CFuncDefNode) and not skip_check:
            ret |= visit_funcdef(node, filename, lines)

        if isinstance(node, (NameNode, CSimpleBaseTypeNode)):
            # do we need node.module_path?
            names.append(Token(node.name, *node.pos[1:]))
            # need this for:
            # ctypedef fused foo:
            #     bar.quox
            names.extend([
                Token(_module, *node.pos[1:])
                for _module in getattr(node, 'module_path', [])
            ])

    return names, imported_names, ret


def _main(code: str, filename: str) -> int:
    tokens = src_to_tokens(code)
    exclude_lines = {
        token.line
        for token in tokens
        if token.name == 'NAME' and token.src == 'include'
    }

    code = tokens_to_src(tokens)
    lines = []
    _dir = os.path.dirname(filename)
    included_texts = []
    for i, line in enumerate(code.splitlines(keepends=True), start=1):
        if i in exclude_lines:
            _file = os.path.join(_dir, line.split()[-1].strip("'\""))
            if os.path.exists(f'{_file}.in'):
                with open(f'{_file}.in', encoding='utf-8') as fd:
                    content = fd.read()
                pyxcontent = Tempita.sub(content)
                included_texts.append(pyxcontent)
            elif os.path.exists(_file):
                with open(_file, encoding='utf-8') as fd:
                    content = fd.read()
                included_texts.append(content)
            lines.append('\n')
        else:
            lines.append(line)

    code = ''.join(lines)

    names, imported_names, ret = _traverse_file(code, filename, lines)

    included_names = []
    for _code in included_texts:
        _included_names, _, __ = _traverse_file(
            _code, filename, _code.splitlines(), skip_check=True,
        )
        included_names.extend(_included_names)

    for _import in imported_names:
        if _import[0] == '*':
            continue
        if (
            _import[0] not in [_name[0] for _name in names if _import != _name]
            and _import[0] not in [_name[0] for _name in included_names]
            and '# no-cython-lint' not in lines[_import[1] - 1]
        ):
            print(
                f'{filename}:{_import[1]}:{_import[2]+1}: '
                f'\'{_import[0]}\' imported but unused',
            )
            ret = 1

    return ret


def traverse(tree: ModuleNode) -> Node:
    nodes = [tree]

    while nodes:
        node = nodes.pop()
        if node is None:
            continue

        child_attrs = copy.deepcopy(node.child_attrs)

        if isinstance(node, CClassDefNode):
            child_attrs.extend(['bases', 'decorators'])
        elif isinstance(node, TypecastNode):
            child_attrs.append('base_type')
        elif isinstance(node, GeneratorExpressionNode):
            if hasattr(node, 'loop'):
                child_attrs.append('loop')
            else:  # pragma: no cover
                err_msg(node, 'GeneratorExpressionNode with loop attribute')
        elif isinstance(node, CFuncDefNode):
            child_attrs.append('decorators')
        elif isinstance(node, FusedTypeNode):
            child_attrs.append('types')
        elif isinstance(node, ForInStatNode):
            child_attrs.append('target')

        for attr in child_attrs:
            child = getattr(node, attr)
            if isinstance(child, list):
                nodes.extend(child)
            else:
                nodes.append(child)
        yield node


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args(argv)
    ret = 0
    for path in args.paths:
        _, ext = os.path.splitext(path)
        if ext != '.pyx':
            continue
        with open(path, encoding='utf-8') as fd:
            content = fd.read()
        try:
            ret |= _main(content, path)
        except CompileError:
            continue
    return ret


if __name__ == '__main__':
    sys.exit(main())
