from __future__ import annotations

import argparse
import sys

from Cython.Compiler.Errors import CompileError
from Cython.Compiler.ExprNodes import (
    ImportNode,
    NameNode,
    TupleNode,
    TypecastNode,
    GeneratorExpressionNode,
    LambdaNode,
)
from Cython.Compiler.Nodes import (
    CArgDeclNode,
    CFuncDeclaratorNode,
    CClassDefNode,
    CFuncDefNode,
    CImportStatNode,
    CNameDeclaratorNode,
    CPtrDeclaratorNode,
    CSimpleBaseTypeNode,
    FusedTypeNode,
    ForInStatNode,
    FromCImportStatNode,
    FromImportStatNode,
    RaiseStatNode,
    SingleAssignmentNode,
)
from Cython.Compiler.TreeFragment import parse_from_strings


class CythonLintError(Exception):
    pass


from tokenize_rt import Token, src_to_tokens, tokens_to_src


def visit_funcdef(node, filename, lines):
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
    for_loop_vars = []
    # for i in children:
    #     if isinstance(i, ForInStatNode):
    #         for _node in _name_from_tuple(i.target):
    #             for_loop_vars.append((_node.name, *_node.pos[1:]))
    defs = [*defs, *simple_assignments, *for_loop_vars]

    args = []
    for i in children:
        if isinstance(i, CArgDeclNode):
            args += _args_from_cargdecl(i)
    if isinstance(node.declarator.base, CNameDeclaratorNode):
        func_name = node.declarator.base.name
    elif isinstance(node.declarator.base, CFuncDeclaratorNode):
        if isinstance(node.declarator.base.base, CNameDeclaratorNode):
            func_name = node.declarator.base.base.name
        else:  # pragma: no cover
            raise CythonLintError("Unexpected error")
    else:  # pragma: no cover
        raise CythonLintError()

    for _def in defs:
        if (
            _def[0] not in [i[0] for i in names]
            and _def[0] != func_name
            and _def[0] not in [i[0] for i in args]
        ) and "# no-lint" not in lines[_def[1] - 1]:
            print(f"{filename}:{_def[1]}:{_def[2]}: '{_def[0]}' defined but unused")
            ret = 1
    return ret


def _name_from_cptrdeclarator(node):
    while isinstance(node, CPtrDeclaratorNode):
        node = node.base
    if isinstance(node, CNameDeclaratorNode):
        return node
    raise CythonLintError("")  # pragma: no cover


def _args_from_cargdecl(i):
    args = []
    if isinstance(i.declarator, CNameDeclaratorNode):
        if i.declarator.name:
            args.append((i.declarator.name, *i.declarator.pos[1:]))
        elif isinstance(i.base_type, (CNameDeclaratorNode, CSimpleBaseTypeNode)):
            args.append((i.base_type.name, *i.base_type.pos[1:]))
        else:  # pragma: no cover
            raise CythonLintError(
                f"Unexpected error, please report bug. Expected CNameDeclaratorNode or CSimpleBaseTypeNode, got {i.base_type}"
            )
    elif isinstance(i.declarator, CPtrDeclaratorNode):
        args.append((i.declarator.base.name, *i.declarator.base.pos[1:]))
    elif isinstance(i.declarator, CFuncDeclaratorNode):
        _args = [_args_from_cargdecl(_arg) for _arg in i.declarator.args]
        args += _args
        _base = _name_from_cptrdeclarator(i.declarator.base)
        args.append((_base.name, *_base.pos[1:]))

    else:  # pragma: no cover
        raise CythonLintError(
            f"Unexpected error, please report bug. "
            f"Expected CPtrDeclaratorNode, got {i.declarator}\n"
            f"{i.declarator.pos}\n"
        )
    return args


def main(code, filename):
    ret = 0
    tokens = src_to_tokens(code)
    exclude_lines = {
        token.line
        for token in tokens
        if token.name == "NAME" and token.src == "include"
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
        os.path.join(_dir, line.split()[-1].strip("'\"") + ".in")
        for i, line in enumerate(code.splitlines(keepends=True), start=1)
        if i in exclude_lines
    ]
    included_text = ""
    for _file in included_files:
        with open(_file, encoding="utf-8") as fd:
            content = fd.read()
        included_text += content
    code = "".join(lines)

    tree = parse_from_strings(filename, code)
    nodes = list(traverse(tree))
    imported_names = []
    for node in nodes:
        if isinstance(node, FromCImportStatNode):
            imported_names.extend(
                (imp[2] or imp[1], *imp[0][1:]) for imp in node.imported_names
            )

        elif isinstance(node, CImportStatNode):
            imported_names.append((node.as_name or node.module_name, *node.pos[1:]))
        elif isinstance(node, SingleAssignmentNode) and isinstance(
            node.rhs, ImportNode
        ):
            imported_names.append((node.lhs.name, *node.lhs.pos[1:]))
        elif isinstance(node, FromImportStatNode):
            for imp in node.items:
                imported_names.append((imp[1].name, *imp[1].pos[1:]))
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
            and "# no-cython-lint" not in lines[_import[1] - 1]
        ):
            print(
                f"{filename}:{_import[1]}:{_import[2]+1}: Name {_import[0]} imported but unused"
            )
            ret = 1

    return ret


def traverse(tree):
    nodes = [tree]

    while nodes:
        node = nodes.pop()
        if node is None:
            continue

        import copy

        child_attrs = copy.deepcopy(node.child_attrs)

        if isinstance(node, CClassDefNode):
            child_attrs.extend(["bases", "decorators"])
        elif isinstance(node, TypecastNode):
            child_attrs.append("base_type")
        elif isinstance(node, GeneratorExpressionNode):
            if hasattr(node, "loop"):
                child_attrs.append("loop")
            else:  # pragma: no cover
                raise CythonLintError()
        elif isinstance(node, CFuncDefNode):
            child_attrs.append("decorators")
        elif isinstance(node, FusedTypeNode):
            child_attrs.append("types")
        elif isinstance(node, ForInStatNode):
            child_attrs.append("target")

        for attr in child_attrs:
            child = getattr(node, attr)
            if isinstance(child, list):
                nodes.extend(child)
            else:
                nodes.append(child)
        yield node


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args()
    ret = 0
    for path in args.paths:
        with open(path, encoding="utf-8") as fd:
            content = fd.read()
        try:
            ret |= main(content, path)
        except CompileError:
            continue
    sys.exit(ret)
