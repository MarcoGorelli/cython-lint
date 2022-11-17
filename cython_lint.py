from __future__ import annotations

import argparse
import collections
import copy
import os
import subprocess
import sys
import warnings
from typing import Hashable
from typing import Iterator
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Sequence

with warnings.catch_warnings():
    # DeprecationWarning: 'cgi' is deprecated and slated for
    # removal in Python 3.13
    # needs fixing in Cython
    warnings.simplefilter('ignore', DeprecationWarning)
    from Cython import Tempita
    from Cython.Compiler.TreeFragment import StringParseContext
import Cython
from Cython.Compiler.ExprNodes import GeneratorExpressionNode
from Cython.Compiler.ExprNodes import SimpleCallNode
from Cython.Compiler.ExprNodes import AttributeNode
from Cython.Compiler.ExprNodes import SetNode
from Cython.Compiler.ExprNodes import UnicodeNode, IntNode, FloatNode
from Cython.Compiler.ExprNodes import ComprehensionNode
from Cython.Compiler.ExprNodes import PrimaryCmpNode
from Cython.Compiler.ExprNodes import ComprehensionAppendNode
from Cython.Compiler.ExprNodes import DictComprehensionAppendNode
from Cython.Compiler.ExprNodes import TupleNode
from Cython.Compiler.ExprNodes import DictNode
from Cython.Compiler.ExprNodes import ListNode
from Cython.Compiler.ExprNodes import FormattedValueNode
from Cython.Compiler.ExprNodes import JoinedStrNode
from Cython.Compiler.ExprNodes import ImportNode
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.ExprNodes import NewExprNode
from Cython.Compiler.ExprNodes import LambdaNode
from Cython.Compiler.ExprNodes import TypecastNode
from Cython.Compiler.ModuleNode import ModuleNode
from Cython.Compiler.Nodes import CArgDeclNode
from Cython.Compiler.Nodes import AssertStatNode
from Cython.Compiler.Nodes import IfClauseNode
from Cython.Compiler.Nodes import StatListNode
from Cython.Compiler.Nodes import CClassDefNode
from Cython.Compiler.Nodes import CFuncDeclaratorNode
from Cython.Compiler.Nodes import CFuncDefNode
from Cython.Compiler.Nodes import CImportStatNode
from Cython.Compiler.Nodes import CNameDeclaratorNode
from Cython.Compiler.Nodes import CSimpleBaseTypeNode
from Cython.Compiler.Nodes import CVarDefNode
from Cython.Compiler.Nodes import DefNode

from Cython.Compiler.Nodes import ForInStatNode
from Cython.Compiler.Nodes import FromCImportStatNode
from Cython.Compiler.Nodes import FromImportStatNode
from Cython.Compiler.Nodes import FusedTypeNode
from Cython.Compiler.Nodes import Node
from Cython.Compiler.Nodes import SingleAssignmentNode
from Cython.Compiler.TreeFragment import parse_from_strings
from tokenize_rt import src_to_tokens
from tokenize_rt import tokens_to_src

# generate these with python generate_pycodestyle_codes.py
PYCODESTYLE_CODES = frozenset((
    'E121',
    'E123',
    'E126',
    'E133',
    'E203',
    'E211',
    'E225',
    'E226',
    'E227',
    'E241',
    'E242',
    'E251',
    'E271',
    'E272',
    'E275',
    'E4',
    'E704',
    'E9',
    'W5',
))

CONSTANT_NODE = (UnicodeNode, IntNode, FloatNode)


class NodeParent(NamedTuple):
    node: Node
    parent: Node | None


class Token(NamedTuple):
    name: str
    lineno: int
    colno: int


class CythonLintError(Exception):
    pass


class CythonParseError(Exception):
    pass


def err_msg(node: Node, expected: str) -> NoReturn:
    msg = (
        f'Unexpected error, please report bug. '
        f'Expected {expected}, got {node}\n'
        f'{node}\n'
    )
    if hasattr(node, 'pos'):
        msg += f'pos: {node.pos}\n'
    raise CythonLintError(
        msg,
    )


def visit_cvardef(
    node: CVarDefNode,
    lines: Sequence[str],
    violations: list[tuple[int, int, str]],
) -> int:
    _base = lines[node.pos[1]-1][node.pos[2]:]
    ret = 0
    round_parens = 0
    square_parens = 0
    _base_type = ''
    for _ch in _base:
        if _ch == '(':
            round_parens += 1
        elif _ch == ')':
            round_parens -= 1
        elif _ch == '[':
            square_parens += 1
        elif _ch == ']':
            square_parens -= 1
        if _ch == ' ' and not round_parens and not square_parens:
            break
        _base_type += _ch
    if (
            _base_type.endswith(',')
            and '# no-lint' not in lines[node.pos[1]-1]
    ):
        violations.append((
            node.pos[1], node.pos[2],
            'comma after base type in definition',
        ))
        ret = 1
    return ret


def visit_funcdef(
    node: CFuncDefNode | DefNode,
    filename: str,
    lines: Sequence[str],
    global_imports: list[Token],
    violations: list[tuple[int, int, str]],
) -> int:
    ret = 0

    children = [i.node for i in traverse(node)][1:]

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
        # e.g. import numpy as np
        and not isinstance(_child.rhs, ImportNode)
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

    if isinstance(node, CFuncDefNode):
        func = _func_from_base(node.declarator)
        func_name = _name_from_base(func.base).name
    else:
        func_name = node.name

    for _def in defs:
        # we don't report on unused function args
        if (
            _def[0] not in [_name[0] for _name in names if _def != _name]
            and _def[0] != func_name
            and _def[0] not in [i[0] for i in args]
            and not _def[0].startswith('_')
        ) and '# no-lint' not in lines[_def[1] - 1]:

            violations.append((
                _def[1], _def[2]+1,
                f'\'{_def[0]}\' defined but unused',
            ))
            ret = 1
        if _def[0] in [_import[0] for _import in global_imports]:
            _global_import = [
                _import for _import in global_imports if _import[0] == _def[0]
            ][0]
            violations.append((
                _def[1], _def[2]+1,
                f'\'{_def[0]}\' shadows global import on line '
                f'{_global_import[1]} col {_global_import[2]+1}',
            ))
            ret = 1
    return ret


def _name_from_base(node: Node) -> Node:
    while not hasattr(node, 'name'):
        if hasattr(node, 'base'):
            node = node.base
        else:
            err_msg(node, 'CNameDeclaratorNode')  # pragma: no cover
    return node


def _func_from_base(node: Node) -> Node:
    while not isinstance(node, (CFuncDeclaratorNode, CFuncDefNode)):
        if hasattr(node, 'base'):
            node = node.base
        else:
            err_msg(node, 'CFuncDeclaratorNode')  # pragma: no cover
    return node


def _args_from_cargdecl(node: CArgDeclNode) -> Iterator[Token]:
    if isinstance(node.declarator, CFuncDeclaratorNode):
        # e.g. cdef foo(object (*operation)(int64_t value))
        for _arg in node.declarator.args:
            yield from _args_from_cargdecl(_arg)
        _base = _name_from_base(node.declarator.base)
        yield Token(_base.name, *_base.pos[1:])
    elif hasattr(node.declarator, 'base'):
        # e.g. cdef foo(vector[FrontierRecord]& frontier)
        # e.g. cdef foo(double x[])
        _base = _name_from_base(node.declarator)
        yield Token(
            _base.name,
            *_base.pos[1:],
        )
    # e.g. foo(int a), foo(int* a)
    _decl = _name_from_base(node.declarator)
    yield Token(_decl.name, *_decl.pos[1:])


def _record_imports(node: Node) -> Iterator[Token]:
    if isinstance(node, FromCImportStatNode):
        yield from (
            Token(imp[2] or imp[1], *imp[0][1:])
            for imp in node.imported_names
        )
    elif isinstance(node, CImportStatNode):
        yield (
            Token(node.as_name or node.module_name, *node.pos[1:])
        )
    elif isinstance(node, SingleAssignmentNode) and isinstance(
        node.rhs, ImportNode,
    ):
        # e.g. import numpy as np
        yield (Token(node.lhs.name, *node.lhs.pos[1:]))
    elif isinstance(node, FromImportStatNode):
        # from numpy import array
        yield from (
            Token(imp[1].name, *imp[1].pos[1:]) for imp in node.items
        )


def _visit_dict_node(
    node: DictNode,
    violations: list[tuple[int, int, str]],
) -> int:
    ret = 0
    literal_counts: MutableMapping[
        Hashable,
        int,
    ] = collections.Counter()
    variable_counts: MutableMapping[
        Hashable,
        int,
    ] = collections.Counter()
    for key_value_pair in node.key_value_pairs:
        if getattr(key_value_pair.key, 'value', None) is not None:
            literal_counts[key_value_pair.key.value] += 1
        elif getattr(key_value_pair.key, 'name', None) is not None:
            variable_counts[key_value_pair.key.name] += 1
    for key, value in literal_counts.items():
        if value > 1:
            violations.append(
                (
                    node.pos[1], node.pos[2],
                    f'dict key {key} repeated {value} times',
                ),
            )
            ret = 1
    for key, value in variable_counts.items():
        if value > 1:
            violations.append(
                (
                    node.pos[1], node.pos[2],
                    f'dict key variable {key} repeated {value} times',
                ),
            )
            ret = 1
    return ret


def _traverse_file(
        code: str,
        filename: str,
        lines: Sequence[str],
        *,
        skip_check: bool = False,
        violations: list[tuple[int, int, str]] | None = None,
) -> tuple[list[Token], list[Token], int]:
    """
    skip_check: only for when traversing an included file
    """
    ret = 0
    try:
        context = StringParseContext(filename)
        context.set_language_level(3)
        tree = parse_from_strings(filename, code, context=context)
    except Exception as exp:  # pragma: no cover  # noqa: E722
        # If Cython can't parse this file, just skip it.
        print(f'cant parse {filename}: {repr(exp)}')
        raise CythonParseError
    nodes = list(traverse(tree))
    imported_names: list[Token] = []
    global_imports: list[Token] = []

    if isinstance(tree.body, StatListNode):
        for node in tree.body.stats:
            if isinstance(node, StatListNode):
                for _node in node.stats:
                    global_imports.extend(_record_imports(_node))
            global_imports.extend(_record_imports(node))

    names: list[Token] = []
    for node_parent in nodes:
        node = node_parent.node
        imported_names.extend(_record_imports(node))

    for node_parent in nodes:
        node = node_parent.node

        if isinstance(node, (CFuncDefNode, DefNode)) and not skip_check:
            assert violations is not None
            ret |= visit_funcdef(
                node, filename, lines,
                global_imports, violations=violations,
            )

        if isinstance(node, CVarDefNode) and not skip_check:
            assert violations is not None
            ret |= visit_cvardef(node, lines, violations)

        if isinstance(node, DictNode) and not skip_check:
            assert violations is not None
            ret |= _visit_dict_node(node, violations)

        if isinstance(node, (IfClauseNode, AssertStatNode)) and not skip_check:
            assert violations is not None
            version = tuple(Cython.__version__.split('.'))
            if version > ('3',):  # pragma: no cover
                test = isinstance(node.condition, TupleNode)
            elif isinstance(node, IfClauseNode):  # pragma: no cover
                test = isinstance(node.condition, TupleNode)
            else:  # pragma: no cover
                # Cython renamed this in version 3
                test = isinstance(node.cond, TupleNode)

            if test:
                if isinstance(node, IfClauseNode):
                    statement = 'if-statement'
                else:
                    statement = 'assert statement'
                violations.append(
                    (
                        node.pos[1], node.pos[2],
                        f'{statement} with tuple as condition is always '
                        'true - perhaps remove comma?',
                    ),
                )
                ret |= 1

        if (
            isinstance(node, JoinedStrNode)
            and not any(
                isinstance(_child, FormattedValueNode)
                for _child in node.values
            )
            and not isinstance(node_parent.parent, FormattedValueNode)
            and not skip_check
        ):
            assert violations is not None
            violations.append(
                (
                    node.pos[1], node.pos[2],
                    'f-string without any placeholders',
                ),
            )
            ret |= 1

        if isinstance(node, CArgDeclNode) and not skip_check:
            assert violations is not None
            if isinstance(node.default, (ListNode, DictNode)):
                violations.append(
                    (
                        node.pos[1], node.pos[2]+1,
                        'dangerous default value!',
                    ),
                )
                ret = 1

        if (
            isinstance(node, ComprehensionNode)
            and isinstance(node.loop.target, NameNode)
            and isinstance(node.loop.body, ComprehensionAppendNode)
            and not skip_check
        ):
            if isinstance(node.loop.body, DictComprehensionAppendNode):
                expr = node.loop.body.value_expr
            else:
                expr = node.loop.body.expr
            if isinstance(expr, LambdaNode) and not hasattr(expr, 'loop'):
                # GeneratorExpressionNode is a LambdaNode, and has a loop
                # attribute, so need to exclude it.
                assert violations is not None
                _children = [j.node for j in traverse(expr)]
                _names = [
                    _child.name for _child in _children if isinstance(
                        _child, NameNode,
                    )
                ]
                if node.loop.target.name in _names:
                    violations.append(
                        (
                            node.pos[1], node.pos[2]+1,
                            'Late binding closure! Careful '
                            'https://docs.python-guide.org/writing/gotchas/'
                            '#late-binding-closures',
                        ),
                    )
                    ret = 1

        if (
            isinstance(node, ForInStatNode)
            and isinstance(node.target, NameNode)
            and isinstance(node.body, StatListNode)
            and not skip_check
        ):
            assert violations is not None
            for _stat in node.body.stats:
                if isinstance(_stat, (DefNode, CFuncDefNode)):
                    expr = _stat.body
                    _children = [j.node for j in traverse(expr)]
                    _names = [
                        i.name for i in _children if isinstance(i, NameNode)
                    ]
                    if node.target.name in _names:
                        violations.append(
                            (
                                node.pos[1], node.pos[2]+1,
                                'Late binding closure! Careful '
                                'https://docs.python-guide.org/writing/gotchas'
                                '/#late-binding-closures',
                            ),
                        )
                        ret = 1

        if (
            isinstance(node, PrimaryCmpNode)
            and isinstance(node.operand1, CONSTANT_NODE)
            and isinstance(node.operand2, CONSTANT_NODE)
            and not skip_check
        ):
            assert violations is not None
            violations.append(
                (
                    node.pos[1], node.pos[2]+1,
                    'Comparison between constants',
                ),
            )
            ret = 1

        if (
            isinstance(node, SetNode)
            and not skip_check
        ):
            assert violations is not None
            counts: MutableMapping[object, int] = collections.Counter()
            for _arg in node.args:
                if hasattr(_arg, 'value'):
                    counts[_arg.value] += 1
            if counts and max(counts.values()) > 1:
                violations.append(
                    (
                        node.pos[1], node.pos[2]+1,
                        'Repeated element in set',
                    ),
                )
                ret = 1

        if (
            isinstance(node, SimpleCallNode)
            and isinstance(node.function, AttributeNode)
            and not skip_check
        ):
            assert violations is not None
            if node.function.attribute in {'strip', 'rstrip', 'lstrip'}:
                if node.args and isinstance(node.args[0], UnicodeNode):
                    if len(set(node.args[0].value)) != len(node.args[0].value):
                        violations.append(
                            (
                                node.pos[1], node.pos[2]+1,
                                f'Using \'{node.function.attribute}\' with '
                                'repeated elements',
                            ),
                        )
                        ret = 1

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


def _main(
    code: str,
    filename: str,
    *,
    line_length: int = 88,
    no_pycodestyle: bool = False,
) -> int:
    violations: list[tuple[int, int, str]] = []
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

    names, imported_names, ret = _traverse_file(
        code, filename, lines, violations=violations,
    )

    included_names = []
    for _code in included_texts:
        _included_names, _, __ = _traverse_file(
            _code, filename, _code.splitlines(), skip_check=True,
        )
        included_names.extend(_included_names)

    for _import in imported_names:
        if _import[0] == '*':
            continue
        elif '.' in _import[0]:
            # e.g. import foo.bar
            # foo.bar.bat()
            # skip for now so there's no false negative
            continue
        elif (
            _import[0] not in [_name[0] for _name in names if _import != _name]
            and _import[0] not in [_name[0] for _name in included_names]
            and '# no-cython-lint' not in lines[_import[1] - 1]
        ):
            violations.append((
                _import[1], _import[2]+1,
                f'\'{_import[0]}\' imported but unused',
            ))
            ret = 1

    if not no_pycodestyle:
        output = subprocess.run(
            [
                'pycodestyle',
                f'--ignore={",".join(PYCODESTYLE_CODES)}',
                f'--max-line-length={line_length}',
                '--format=%(row)d:%(col)d: %(code)s %(text)s',
                filename,
            ],
            text=True,
            capture_output=True,
        )
        ret = ret | bool(output.returncode)

        extra_lines = output.stdout.splitlines()
        for extra_line in extra_lines:
            import re
            if re.search(r'^\d+:\d+:', extra_line) is None:
                # could be an extra line with pycodestyle statistics
                continue
            _lineno, _col, message = extra_line.split(':', maxsplit=2)
            violations.append((int(_lineno), int(_col), message))

    for lineno, col, message in sorted(violations):
        print(f'{filename}:{lineno}:{col}: {message}')

    return ret


def traverse(tree: ModuleNode) -> Node:
    nodes = [NodeParent(tree, None)]

    while nodes:
        node_parent = nodes.pop()
        node = node_parent.node
        if node is None:
            continue

        child_attrs = set(copy.deepcopy(node.child_attrs))

        if isinstance(node, CClassDefNode):
            child_attrs.update(['bases', 'decorators'])
        elif isinstance(node, TypecastNode):
            child_attrs.add('base_type')
        elif isinstance(node, GeneratorExpressionNode):
            if hasattr(node, 'loop'):
                child_attrs.add('loop')
            else:  # pragma: no cover
                err_msg(node, 'GeneratorExpressionNode with loop attribute')
        elif isinstance(node, CFuncDefNode):
            child_attrs.add('decorators')
        elif isinstance(node, FusedTypeNode):
            child_attrs.add('types')
        elif isinstance(node, ForInStatNode):
            child_attrs.add('target')
        elif isinstance(node, NewExprNode):
            child_attrs.add('cppclass')
        elif isinstance(node, LambdaNode):
            child_attrs.update(['args', 'result_expr'])

        for attr in child_attrs:
            child = getattr(node, attr)
            if isinstance(child, list):
                nodes.extend([NodeParent(_child, node) for _child in child])
            else:
                nodes.append(NodeParent(child, node))
        yield node_parent


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    # default from black formatter
    parser.add_argument('--max-line-length', type=int, default=88)
    parser.add_argument('--no-pycodestyle', action='store_true')
    args = parser.parse_args(argv)
    ret = 0
    for path in args.paths:
        _, ext = os.path.splitext(path)
        if ext != '.pyx':
            continue
        try:
            with open(path, encoding='utf-8') as fd:
                content = fd.read()
        except UnicodeDecodeError:
            continue
        try:
            ret |= _main(
                content, path, line_length=args.max_line_length,
                no_pycodestyle=args.no_pycodestyle,
            )
        except CythonParseError:
            continue
    return ret


if __name__ == '__main__':
    sys.exit(main())
