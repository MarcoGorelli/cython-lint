from __future__ import annotations

import argparse
import sys
from typing import Iterator
from typing import NamedTuple
from typing import Sequence

from Cython.Compiler.Errors import CompileError
from Cython.Compiler.ExprNodes import GeneratorExpressionNode
from Cython.Compiler.ExprNodes import ImportNode
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.ExprNodes import TypecastNode
from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Nodes import CArgDeclNode
from Cython.Compiler.Nodes import CClassDefNode
from Cython.Compiler.Nodes import CFuncDeclaratorNode
from Cython.Compiler.Nodes import CFuncDefNode
from Cython.Compiler.Nodes import CImportStatNode
from Cython.Compiler.Nodes import CNameDeclaratorNode
from Cython.Compiler.Nodes import CPtrDeclaratorNode
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


def visit_funcdef(
    node: CFuncDefNode,
    filename: str,
    lines: Sequence[str],
) -> int:
    ret = 0

    children = list(traverse(node))[1:]
    names = [
        (i.name, *i.pos[1:])
        for i in children
        if isinstance(i, (NameNode, CSimpleBaseTypeNode))
    ]
    defs = [
        (i.name, *i.pos[1:])
        for i in children
        if isinstance(i, CNameDeclaratorNode)
        if i.name
    ]
    simple_assignments = [
        (i.lhs.name, *i.lhs.pos[1:])
        for i in children
        if isinstance(i, SingleAssignmentNode) and isinstance(i.lhs, NameNode)
    ]
    defs = [*defs, *simple_assignments]

    args = []
    for i in children:
        if isinstance(i, CArgDeclNode):
            for _arg in _args_from_cargdecl(i):
                args.append(_arg)
    if isinstance(node.declarator.base, CNameDeclaratorNode):
        func_name = node.declarator.base.name
    elif isinstance(node.declarator.base, CFuncDeclaratorNode):
        if isinstance(node.declarator.base.base, CNameDeclaratorNode):
            func_name = node.declarator.base.base.name
        else:  # pragma: no cover
            raise CythonLintError('Unexpected error')
    else:  # pragma: no cover
        raise CythonLintError()

    for _def in defs:
        if (
            _def[0] not in [i[0] for i in names]
            and _def[0] != func_name
            and _def[0] not in [i[0] for i in args]
        ) and '# no-lint' not in lines[_def[1] - 1]:
            print(
                f'{filename}:{_def[1]}:{_def[2]}: '
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
    raise CythonLintError('')  # pragma: no cover


def _args_from_cargdecl(node: CArgDeclNode) -> Iterator[Token]:
    if isinstance(node.declarator, CNameDeclaratorNode):
        if node.declarator.name:
            yield Token(node.declarator.name, *node.declarator.pos[1:])
        elif isinstance(
            node.base_type,
            (CNameDeclaratorNode, CSimpleBaseTypeNode),
        ):
            yield Token(node.base_type.name, *node.base_type.pos[1:])
        else:  # pragma: no cover
            raise CythonLintError(
                'Unexpected error, please report bug. '
                'Expected CNameDeclaratorNode or CSimpleBaseTypeNode, '
                f'got {node.base_type}',
            )
    elif isinstance(node.declarator, CPtrDeclaratorNode):
        yield (
            Token(
                node.declarator.base.name,
                *node.declarator.base.pos[1:],
            )
        )
    elif isinstance(node.declarator, CFuncDeclaratorNode):
        for _arg in node.declarator.args:
            yield from _args_from_cargdecl(_arg)
        _base = _name_from_cptrdeclarator(node.declarator.base)
        yield Token(_base.name, *_base.pos[1:])

    else:  # pragma: no cover
        raise CythonLintError(
            f'Unexpected error, please report bug. '
            f'Expected CPtrDeclaratorNode, got {node.declarator}\n'
            f'{node.declarator.pos}\n',
        )


def main(code: str, filename: str) -> int:
    ret = 0
    tokens = src_to_tokens(code)
    exclude_lines = {
        token.line
        for token in tokens
        if token.name == 'NAME' and token.src == 'include'
    }

    code = tokens_to_src(tokens)
    lines = [
        line
        for i, line in enumerate(code.splitlines(keepends=True), start=1)
        if i not in exclude_lines
    ]
    import os

    _dir = os.path.dirname(filename)
    included_files = [
        os.path.join(_dir, line.split()[-1].strip("'\"") + '.in')
        for i, line in enumerate(code.splitlines(keepends=True), start=1)
        if i in exclude_lines
    ]
    included_text = ''
    for _file in included_files:
        with open(_file, encoding='utf-8') as fd:
            content = fd.read()
        included_text += content
    code = ''.join(lines)

    tree = parse_from_strings(filename, code)
    nodes = list(traverse(tree))
    imported_names: list[Token] = []
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
            imported_names.append(Token(node.lhs.name, *node.lhs.pos[1:]))
        elif isinstance(node, FromImportStatNode):
            for imp in node.items:
                imported_names.append(Token(imp[1].name, *imp[1].pos[1:]))
    imported_names = sorted(imported_names, key=lambda x: (x[1], x[2]))

    for node in nodes:
        if isinstance(node, CFuncDefNode):
            ret |= visit_funcdef(node, filename, lines)

    names = [
        (i.name, *i.pos[1:])
        for i in nodes
        if isinstance(i, (NameNode, CSimpleBaseTypeNode))
    ]
    for _import in imported_names:
        if (
            _import[0] not in [i[0] for i in names]
            and _import[0] not in included_text
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

        import copy

        child_attrs = copy.deepcopy(node.child_attrs)

        if isinstance(node, CClassDefNode):
            child_attrs.extend(['bases', 'decorators'])
        elif isinstance(node, TypecastNode):
            child_attrs.append('base_type')
        elif isinstance(node, GeneratorExpressionNode):
            if hasattr(node, 'loop'):
                child_attrs.append('loop')
            else:  # pragma: no cover
                raise CythonLintError()
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    args = parser.parse_args()
    ret = 0
    for path in args.paths:
        with open(path, encoding='utf-8') as fd:
            content = fd.read()
        try:
            ret |= main(content, path)
        except CompileError:
            continue
    sys.exit(ret)