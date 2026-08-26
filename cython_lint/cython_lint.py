from __future__ import annotations

import argparse
import collections
import contextlib
import copy
import os
import pathlib
import re
import subprocess
import sys
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import NoReturn
from typing import cast

from Cython.Compiler.Errors import init_thread

if sys.version_info >= (3, 11):  # pragma: no cover
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

with warnings.catch_warnings():
    # DeprecationWarning: 'cgi' is deprecated and slated for
    # removal in Python 3.13
    # needs fixing in Cython
    warnings.simplefilter("ignore", DeprecationWarning)
    from Cython import Tempita
    from Cython.Compiler.TreeFragment import StringParseContext
import Cython
from Cython.Compiler.ExprNodes import AttributeNode
from Cython.Compiler.ExprNodes import ComprehensionAppendNode
from Cython.Compiler.ExprNodes import ComprehensionNode
from Cython.Compiler.ExprNodes import DictComprehensionAppendNode
from Cython.Compiler.ExprNodes import DictNode
from Cython.Compiler.ExprNodes import ExprNode
from Cython.Compiler.ExprNodes import FloatNode
from Cython.Compiler.ExprNodes import FormattedValueNode
from Cython.Compiler.ExprNodes import GeneratorExpressionNode
from Cython.Compiler.ExprNodes import ImportNode
from Cython.Compiler.ExprNodes import IndexNode
from Cython.Compiler.ExprNodes import IntNode
from Cython.Compiler.ExprNodes import IteratorNode
from Cython.Compiler.ExprNodes import JoinedStrNode
from Cython.Compiler.ExprNodes import LambdaNode
from Cython.Compiler.ExprNodes import ListNode
from Cython.Compiler.ExprNodes import NameNode
from Cython.Compiler.ExprNodes import PrimaryCmpNode
from Cython.Compiler.ExprNodes import SequenceNode
from Cython.Compiler.ExprNodes import SetNode
from Cython.Compiler.ExprNodes import SimpleCallNode
from Cython.Compiler.ExprNodes import TupleNode
from Cython.Compiler.ExprNodes import UnicodeNode
from Cython.Compiler.Nodes import AssertStatNode
from Cython.Compiler.Nodes import CArgDeclNode
from Cython.Compiler.Nodes import CDeclaratorNode
from Cython.Compiler.Nodes import CFuncDeclaratorNode
from Cython.Compiler.Nodes import CFuncDefNode
from Cython.Compiler.Nodes import CImportStatNode
from Cython.Compiler.Nodes import CNameDeclaratorNode
from Cython.Compiler.Nodes import CSimpleBaseTypeNode
from Cython.Compiler.Nodes import CVarDefNode
from Cython.Compiler.Nodes import DefNode
from Cython.Compiler.Nodes import ExprStatNode
from Cython.Compiler.Nodes import ForInStatNode
from Cython.Compiler.Nodes import FromCImportStatNode
from Cython.Compiler.Nodes import FromImportStatNode
from Cython.Compiler.Nodes import FuncDefNode
from Cython.Compiler.Nodes import GlobalNode
from Cython.Compiler.Nodes import IfClauseNode
from Cython.Compiler.Nodes import Node
from Cython.Compiler.Nodes import SingleAssignmentNode
from Cython.Compiler.Nodes import StatListNode
from Cython.Compiler.Nodes import StatNode
from Cython.Compiler.TreeFragment import parse_from_strings
from tokenize_rt import src_to_tokens
from tokenize_rt import tokens_to_src

from cython_lint import __version__

# non-numeric parts (e.g. the 'b1' of '3.1.0b1') are dropped
CYTHON_VERSION = tuple(
    int(part)
    for part in Cython.__version__.split(".")  # type: ignore[attr-defined]
    if part.isdigit()
)
CYTHON_3 = (3,)
CYTHON_3_3 = (3, 3)
if TYPE_CHECKING:
    from collections.abc import Hashable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import MutableMapping
    from collections.abc import Sequence


EXCLUDES = (
    r"/("
    r"\.direnv|\.eggs|\.git|\.hg|\.ipynb_checkpoints|\.mypy_cache|\.nox|\.svn|"
    r"\.tox|\.venv|"
    r"_build|buck-out|build|dist|venv"
    r")/"
)

if CYTHON_VERSION >= CYTHON_3:
    from Cython.Compiler.ExprNodes import AnnotationNode  # type: ignore[assignment]
else:  # pragma: no cover

    class AnnotationNode:  # type: ignore[no-redef]
        pass


PRAGMA = r"#\s+no-cython-lint"

# generate these with python generate_pycodestyle_codes.py
PYCODESTYLE_CODES = frozenset(
    (
        "E121",
        "E123",
        "E126",
        "E133",
        "E203",
        "E211",
        "E225",
        "E226",
        "E227",
        "E241",
        "E242",
        "E251",
        "E271",
        "E272",
        "E275",
        "E4",
        "E704",
        "E9",
        "W5",
    )
)

CONSTANT_NODE = (UnicodeNode, IntNode, FloatNode)

MISSING_CHILD_ATTRS = frozenset(
    (
        "bases",
        "decorators",
        "base_type",
        "loop",
        "decorators",
        "types",
        "target",
        "cppclass",
        "args",
        "result_expr",
        "expr",
        "attribute",
        "base_type_node",
        "annotation",
    )
)


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
        f"Unexpected error, please report bug. Expected {expected}, got {node}\n{node}\n"
    )
    if hasattr(node, "pos"):
        msg += f"pos: {node.pos}\n"
    raise CythonLintError(
        msg,
    )


def visit_cvardef(
    node: CVarDefNode,
    lines: Mapping[int, str],
    violations: list[tuple[int, int, str]],
) -> None:
    _base = lines[node.pos[1]][node.pos[2] :]
    round_parens = 0
    square_parens = 0
    _base_type = ""
    for _ch in _base:
        if _ch == "(":
            round_parens += 1
        elif _ch == ")":
            round_parens -= 1
        elif _ch == "[":  # pragma: no cover
            square_parens += 1
        elif _ch == "]":  # pragma: no cover
            square_parens -= 1
        if _ch == " " and not round_parens and not square_parens:
            break
        _base_type += _ch
    if _base_type.endswith(","):  # pragma: no cover
        violations.append(
            (
                node.pos[1],
                node.pos[2],
                "comma after base type in definition",
            )
        )


def visit_funcdef(
    node: CFuncDefNode | DefNode,
    global_names: list[str],
    global_imports: list[Token],
    violations: list[tuple[int, int, str]],
) -> None:
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
        Token(_name_from_name_node(_child.lhs), *_child.lhs.pos[1:])
        for _child in children
        if isinstance(_child, SingleAssignmentNode)
        and isinstance(_child.lhs, NameNode)
        # e.g. import numpy as np
        and not isinstance(_child.rhs, ImportNode)
    ]
    tuple_assignments = []
    for _child in children:
        if (
            isinstance(_child, SingleAssignmentNode)
            and isinstance(_child.lhs, TupleNode)
            # e.g. import numpy as np
            and not isinstance(_child.rhs, ImportNode)
        ):
            tuple_assignments.extend(
                [
                    Token(_name_from_name_node(_arg), *_arg.pos[1:])
                    for _arg in _child.lhs.args
                    if isinstance(_arg, NameNode)
                ]
            )
    defs = [*defs, *simple_assignments, *tuple_assignments]

    names = [
        Token(_name_from_name_node(_child), *_child.pos[1:])
        for _child in children
        if isinstance(_child, NameNode)
    ]

    args: list[Token] = []
    for _child in children:
        if isinstance(_child, CArgDeclNode):
            args.extend(_args_from_cargdecl(_child))

    if isinstance(node, CFuncDefNode):
        _declarator: CDeclaratorNode = node.declarator  # type: ignore[assignment]
        func = _func_from_base(_declarator)
        func_name = _name_from_name_node(_name_from_base(func.base))  # type: ignore[attr-defined]
    else:
        func_name = _name_from_name_node(node)

    for _def in defs:
        # we don't report on unused function args
        if (
            _def[0] not in [_name[0] for _name in names if _def != _name]
            and _def[0] != func_name
            and _def[0] not in [i[0] for i in args]
            and not _def[0].startswith("_")
            and not _def[0].startswith("unused")
            and _def[0] not in global_names
        ):
            violations.append(
                (
                    _def[1],
                    _def[2] + 1,
                    f"'{_def[0]}' defined but unused (try prefixing with underscore?)",
                )
            )
        if _def[0] in [_import[0] for _import in global_imports]:
            _global_import = next(  # pragma: no cover
                _import for _import in global_imports if _import[0] == _def[0]
            )
            violations.append(
                (
                    _def[1],
                    _def[2] + 1,
                    f"'{_def[0]}' shadows global import on line "
                    f"{_global_import[1]} col {_global_import[2] + 1}",
                )
            )


# Helper functions to work around upstream issues.


def _name_from_name_node(node: NameNode | CSimpleBaseTypeNode | DefNode) -> str:
    return node.name  # type: ignore[attr-defined]


def _cond_from_assert_stat_node(node: AssertStatNode) -> ExprNode:
    return node.cond  # type: ignore[attr-defined]  # pragma: no cover


def _default_from_cargdecl_node(node: CArgDeclNode) -> ExprNode | None:
    return node.default  # type: ignore[attr-defined]


def _loop_from_loop_node(
    node: GeneratorExpressionNode | ComprehensionNode,
) -> ForInStatNode:
    return node.loop  # type: ignore[attr-defined]


def _target_from_for_in_stat_node(node: ForInStatNode) -> SequenceNode:
    return node.target  # type: ignore[attr-defined]


def _args_from_sequence_node(node: SequenceNode) -> list[ExprNode]:
    return node.args  # type: ignore[attr-defined]


def _value_from_dict_comprehension_append_node(
    node: DictComprehensionAppendNode,
) -> ExprNode:
    if CYTHON_VERSION >= CYTHON_3_3:
        # Cython 3.3 replaced key_expr/value_expr with a single DictItemNode
        return node.dict_item.value  # type: ignore[attr-defined]
    return node.value_expr  # type: ignore[attr-defined]  # pragma: no cover


def _rhs_from_single_assignment_node(node: SingleAssignmentNode) -> ExprNode:
    return node.rhs  # type: ignore[attr-defined]


def _operand1_from_primary_cmp_node(node: PrimaryCmpNode) -> ExprNode:
    return node.operand1  # type: ignore[attr-defined]


def _operand2_from_primary_cmp_node(node: PrimaryCmpNode) -> ExprNode:
    return node.operand2  # type: ignore[attr-defined]


def _value_from_unicode_node(node: UnicodeNode) -> str:
    return node.value  # type: ignore[attr-defined]


def _args_from_simple_call_node(node: SimpleCallNode) -> list[ExprNode]:
    return node.args  # type: ignore[attr-defined]


def _name_from_base(node: Node) -> NameNode:
    while not hasattr(node, "name"):
        if hasattr(node, "base"):
            node = node.base  # type: ignore[attr-defined]
        else:
            err_msg(node, "CNameDeclaratorNode")  # pragma: no cover
    return node  # type: ignore[return-type]


def _func_from_base(node: Node) -> CFuncDeclaratorNode | CFuncDefNode:
    while not isinstance(node, (CFuncDeclaratorNode, CFuncDefNode)):
        if hasattr(node, "base"):
            node = node.base  # type: ignore[attr-defined]
        else:
            err_msg(node, "CFuncDeclaratorNode")  # pragma: no cover
    return node


def _args_from_cargdecl(node: CArgDeclNode) -> Iterator[Token]:
    _declarator: CArgDeclNode = node.declarator  # type: ignore[assignment]
    if isinstance(_declarator, CFuncDeclaratorNode):
        # e.g. cdef foo(object (*operation)(int64_t value))
        _args: list[CArgDeclNode] = _declarator.args  # type: ignore[assignment]
        for _arg in _args:
            yield from _args_from_cargdecl(_arg)
        _base = _name_from_base(_declarator.base)
        yield Token(_name_from_name_node(_base), *_base.pos[1:])
    elif hasattr(_declarator, "base"):
        # e.g. cdef foo(vector[FrontierRecord]& frontier)
        # e.g. cdef foo(double x[])
        _base = _name_from_base(_declarator)
        yield Token(
            _name_from_name_node(_base),
            *_base.pos[1:],
        )
    # e.g. foo(int a), foo(int* a)
    _decl = _name_from_base(_declarator)
    yield Token(_name_from_name_node(_decl), *_decl.pos[1:])


def _record_imports(node: Node) -> Iterator[Token]:
    if isinstance(node, FromCImportStatNode):
        _imported_names: list[tuple[tuple[int, int, int], str, str]] = node.imported_names  # type: ignore[assignment]
        yield from (
            Token(imp[2] or imp[1], imp[0][1], imp[0][2]) for imp in _imported_names
        )
    elif isinstance(node, CImportStatNode):
        _as_name: str = node.as_name  # type: ignore[assignment]
        _module_name: str = node.module_name  # type: ignore[assignment]
        yield (Token(_as_name or _module_name, *node.pos[1:]))
    elif isinstance(node, SingleAssignmentNode) and isinstance(
        node.rhs,
        ImportNode,
    ):
        _lhs: NameNode = node.lhs  # type: ignore[assignment]
        # e.g. import numpy as np
        yield (Token(_name_from_name_node(_lhs), *_lhs.pos[1:]))
    elif isinstance(node, FromImportStatNode):
        # from numpy import array
        _items: list[tuple[str, NameNode]] = node.items  # type: ignore[assignment]
        yield from (
            Token(_name_from_name_node(imp[1]), *imp[1].pos[1:]) for imp in _items
        )


def visit_dict_node(
    node: DictNode,
    violations: list[tuple[int, int, str]],
) -> None:
    literal_counts: MutableMapping[
        Hashable,
        int,
    ] = collections.Counter()
    variable_counts: MutableMapping[
        Hashable,
        int,
    ] = collections.Counter()
    for key_value_pair in node.key_value_pairs:
        if getattr(key_value_pair.key, "value", None) is not None:
            literal_counts[key_value_pair.key.value] += 1
        elif getattr(key_value_pair.key, "name", None) is not None:
            variable_counts[key_value_pair.key.name] += 1
    for key, value in literal_counts.items():
        if value > 1:
            violations.append(
                (
                    node.pos[1],
                    node.pos[2],
                    f"dict key {key} repeated {value} times",
                ),
            )
    for key, value in variable_counts.items():
        if value > 1:
            violations.append(
                (
                    node.pos[1],
                    node.pos[2],
                    f"dict key variable {key} repeated {value} times",
                ),
            )


def _iter_target_name_nodes(target: Node) -> Iterator[NameNode]:
    if isinstance(target, NameNode):
        yield target
    elif isinstance(target, TupleNode):
        for arg in target.args:
            if isinstance(arg, NameNode):
                yield arg
    else:
        pass


def _traverse_loop_body(
    node: Node,
    loop_vars: frozenset[str],
) -> Iterator[Node]:
    """Yield nodes in a for-loop body without entering new scopes.

    Stops recursing into an inner ForInStatNode's body if its target variable
    shadows one of loop_vars (that inner loop will be checked on its own).
    """
    stack: list[Node] = [node]
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (FuncDefNode, LambdaNode, ComprehensionNode)):
            continue
        if isinstance(n, ForInStatNode) and any(
            _name_from_name_node(a) in loop_vars
            for a in _iter_target_name_nodes(n.target)
        ):
            continue
        for attr in getattr(n, "child_attrs", ()):
            child = getattr(n, attr, None)
            if child is None:
                continue
            if isinstance(child, list):
                for c in child:
                    if c is not None and hasattr(c, "child_attrs"):  # pragma: no cover
                        stack.append(c)  # noqa: PERF401
            elif hasattr(child, "child_attrs"):
                stack.append(child)
            else:  # pragma: no cover
                pass


def _traverse_file(  # noqa: PLR0915,PLR0913
    code: str,
    filename: str,
    lines: Mapping[int, str],
    *,
    skip_check: bool,
    violations: list[tuple[int, int, str]] | None,
    ban_relative_imports: bool,
) -> tuple[list[Token], list[Token], list[str]]:
    """
    skip_check: only for when traversing an included file
    """
    try:
        context = StringParseContext(filename)
        context.set_language_level(3)
        init_thread()
        tree = parse_from_strings(filename, code, context=context)
    except Exception as exp:  # pragma: no cover
        # If Cython can't parse this file, just skip it.
        print(
            f"Skipping file {filename}, as it cannot be parsed. Error: {exp!r}",
        )
        raise CythonParseError from exp
    nodes = list(traverse(tree))
    imported_names: list[Token] = []
    global_imports: list[Token] = []
    global_names: list[str] = []
    exported_imports: list[str] = []

    _body: ExprNode = tree.body  # type: ignore[assignment]
    if isinstance(_body, StatListNode):
        _stats: list[StatNode] = tree.body.stats  # type: ignore[assignment]
        for node in _stats:
            if isinstance(node, StatListNode):
                for _node in node.stats:
                    global_imports.extend(_record_imports(_node))
            global_imports.extend(_record_imports(node))

    names: list[Token] = []
    for node_parent in nodes:
        node = node_parent.node
        imported_names.extend(_record_imports(node))
        if isinstance(node, GlobalNode):
            _names: list[str] = node.names  # type: ignore[assignment]
            global_names.extend(_names)

    for node_parent in nodes:
        node = node_parent.node
        if isinstance(node, (NameNode, CSimpleBaseTypeNode)):
            # do we need node.module_path?
            names.append(Token(_name_from_name_node(node), *node.pos[1:]))
            # need this for:
            # ctypedef fused foo:
            #     bar.quox
            names.extend(
                [
                    Token(_module, *node.pos[1:])
                    for _module in getattr(node, "module_path", [])
                ]
            )

        if skip_check:
            continue
        assert violations is not None  # help mypy

        if isinstance(node, (CFuncDefNode, DefNode)):
            visit_funcdef(
                node,
                global_names,
                global_imports,
                violations=violations,
            )

        if isinstance(node, CVarDefNode):
            visit_cvardef(node, lines, violations)

        if isinstance(node, DictNode):
            visit_dict_node(node, violations)

        if isinstance(node, CImportStatNode):
            _module_name: str = node.module_name  # type: ignore[assignment]
            _as_name: str = node.as_name  # type: ignore[assignment]
            if _module_name == _as_name:
                violations.append(
                    (
                        node.pos[1],
                        node.pos[2] + 1,
                        "Found useless import alias",
                    ),
                )

        if isinstance(node, FromCImportStatNode):
            _imported_names: list[tuple[tuple[int, int], str, str]] = node.imported_names  # type: ignore[assignment]
            for _imported_name in _imported_names:
                if _imported_name[1] == _imported_name[2]:
                    violations.append(
                        (
                            _imported_name[0][1],
                            _imported_name[0][1],
                            "Found useless import alias",
                        ),
                    )
            if ban_relative_imports and node.relative_level:
                violations.append(
                    (
                        node.pos[1],
                        node.pos[2],
                        "Found relative import",
                    ),
                )

        if (
            ban_relative_imports
            and isinstance(node, FromImportStatNode)
            and isinstance(getattr(node, "module", None), ImportNode)
            and node.module.level
        ):
            violations.append(
                (
                    node.pos[1],
                    node.pos[2],
                    "Found relative import",
                ),
            )

        if isinstance(node, (IfClauseNode, AssertStatNode)):
            if CYTHON_VERSION >= CYTHON_3 or isinstance(
                node, IfClauseNode
            ):  # pragma: no cover
                test = isinstance(node.condition, TupleNode)
            else:  # pragma: no cover
                # Cython renamed this in version 3
                test = isinstance(_cond_from_assert_stat_node(node), TupleNode)

            if test:
                if isinstance(node, IfClauseNode):
                    statement = "if-statement"
                else:
                    statement = "assert statement"
                violations.append(
                    (
                        node.pos[1],
                        node.pos[2],
                        f"{statement} with tuple as condition is always "
                        "true - perhaps remove comma?",
                    ),
                )

        if (
            isinstance(node, JoinedStrNode)
            and not any(isinstance(_child, FormattedValueNode) for _child in node.values)
            and not isinstance(node_parent.parent, FormattedValueNode)
        ):
            violations.append(
                (
                    node.pos[1],
                    node.pos[2],
                    "f-string without any placeholders",
                ),
            )

        if (
            isinstance(node, CArgDeclNode)
            and not skip_check
            and isinstance(_default_from_cargdecl_node(node), (ListNode, DictNode))
        ):
            violations.append(
                (
                    node.pos[1],
                    node.pos[2] + 1,
                    "dangerous default value!",
                ),
            )

        if (
            isinstance(node, ComprehensionNode)
            and isinstance(node.loop.target, NameNode)
            and isinstance(node.loop.body, ComprehensionAppendNode)
        ):
            if isinstance(node.loop.body, DictComprehensionAppendNode):
                _expr = _value_from_dict_comprehension_append_node(node.loop.body)
            else:
                _expr = node.loop.body.expr
            if isinstance(_expr, LambdaNode) and not hasattr(_expr, "loop"):
                # GeneratorExpressionNode is a LambdaNode, and has a loop
                # attribute, so need to exclude it.
                _children = [j.node for j in traverse(_expr)]
                _names = [
                    _name_from_name_node(_child)
                    for _child in _children
                    if isinstance(
                        _child,
                        NameNode,
                    )
                ]
                if _name_from_name_node(node.loop.target) in _names:
                    violations.append(
                        (
                            node.pos[1],
                            node.pos[2] + 1,
                            "Late binding closure! Careful "
                            "https://docs.python-guide.org/writing/gotchas/"
                            "#late-binding-closures",
                        ),
                    )

        if (
            isinstance(node, ForInStatNode)
            and isinstance(node.target, NameNode)
            and isinstance(node.body, StatListNode)
        ):
            for _stat in node.body.stats:
                if isinstance(_stat, (DefNode, CFuncDefNode)):
                    expr: StatListNode = _stat.body  # type: ignore[assignment]
                    _children = [j.node for j in traverse(expr)]
                    _names = [
                        _name_from_name_node(i)
                        for i in _children
                        if isinstance(i, NameNode)
                    ]
                    if _name_from_name_node(node.target) in _names:
                        violations.append(
                            (
                                node.pos[1],
                                node.pos[2] + 1,
                                "Late binding closure! Careful "
                                "https://docs.python-guide.org/writing/gotchas"
                                "/#late-binding-closures",
                            ),
                        )

        if (
            isinstance(node, PrimaryCmpNode)
            and isinstance(node.operand1, CONSTANT_NODE)
            and isinstance(node.operand2, CONSTANT_NODE)
        ):
            violations.append(
                (
                    node.pos[1],
                    node.pos[2] + 1,
                    "Comparison between constants",
                ),
            )

        if isinstance(node, SetNode):
            args: list[ExprNode] = node.args  # type: ignore[assignment]
            counts: MutableMapping[object, int] = collections.Counter()
            for arg in args:
                if hasattr(arg, "value"):
                    counts[arg.value] += 1  # type: ignore[attr-defined]
            if counts and max(counts.values()) > 1:
                violations.append(
                    (
                        node.pos[1],
                        node.pos[2] + 1,
                        "Repeated element in set",
                    ),
                )

        if (
            isinstance(node, SimpleCallNode)
            and isinstance(node.function, AttributeNode)
            and node.function.attribute in {"strip", "rstrip", "lstrip"}
            and (
                node.args
                and isinstance(node.args[0], UnicodeNode)
                and len(set(_value_from_unicode_node(node.args[0])))
                != len(_value_from_unicode_node(node.args[0]))
            )
        ):
            violations.append(
                (
                    node.pos[1],
                    node.pos[2] + 1,
                    f"Using '{node.function.attribute}' with repeated elements",
                ),
            )

        if (
            isinstance(node, ExprStatNode)
            and isinstance(node.expr, UnicodeNode)
            and isinstance(node_parent.parent, StatListNode)
        ):
            try:
                idx = node_parent.parent.stats.index(node)
            except ValueError:  # pragma: no cover
                pass  # defensive check
            else:
                if not isinstance(
                    node_parent.parent.stats[idx - 1], SingleAssignmentNode
                ):
                    violations.append(
                        (
                            node.pos[1],
                            node.pos[2] + 1,
                            "pointless string statement",
                        ),
                    )

        if (
            isinstance(node, SimpleCallNode)
            and isinstance(node.function, NameNode)
            and hasattr(node.function, "name")
            and (args := cast("list[ExprNode]", node.args))
            and len(args) == 1
            and isinstance(args[0], (GeneratorExpressionNode, ComprehensionNode))
            and (
                (_func_name := _name_from_name_node(node.function)) in {"list", "set"}
                or (
                    _func_name == "dict"
                    and isinstance(
                        (
                            _target := _target_from_for_in_stat_node(
                                _loop_from_loop_node(args[0])
                            )
                        ),
                        TupleNode,
                    )
                    and len(_args_from_sequence_node(_target)) == 2
                )
            )
        ):
            violations.append(
                (
                    node.pos[1],
                    node.pos[2] + 1,
                    f"unnecessary {_func_name} + generator (just use a "
                    f"{_func_name} comprehension)",
                ),
            )

        if (
            isinstance(node, ForInStatNode)
            and isinstance(node.target, TupleNode)
            and len(node.target.args) == 2
            and isinstance(node.target.args[0], NameNode)
            and isinstance(node.target.args[1], NameNode)
            and isinstance(
                (sequence := cast("IteratorNode", node.iterator).sequence),
                SimpleCallNode,
            )
            and isinstance(sequence.function, NameNode)
            and _name_from_name_node(sequence.function) == "enumerate"
            and (args := cast("list[ExprNode]", sequence.args))
            and len(args) == 1
            and isinstance(args[0], NameNode)
        ):
            for _child in traverse(node.body):
                if isinstance(_child.node, SingleAssignmentNode) and isinstance(
                    _rhs_from_single_assignment_node(_child.node), IndexNode
                ):
                    index_node: IndexNode = _child.node.rhs  # type: ignore[assignment]
                elif isinstance(_child.node, PrimaryCmpNode) and (
                    isinstance(_operand1_from_primary_cmp_node(_child.node), IndexNode)
                ):
                    index_node = _child.node.operand1  # type: ignore[assignment]
                elif isinstance(_child.node, PrimaryCmpNode) and (
                    isinstance(_operand2_from_primary_cmp_node(_child.node), IndexNode)
                ):
                    index_node = _child.node.operand2  # type: ignore[assignment]
                elif (
                    isinstance(_child.node, SimpleCallNode)
                    and isinstance(_child.node.function, AttributeNode)
                    and isinstance(_child.node.function.obj, NameNode)
                    and _child.node.function.attribute == "append"
                    and len(_args_from_simple_call_node(_child.node)) == 1
                    and isinstance(_args_from_simple_call_node(_child.node)[0], IndexNode)
                ):
                    index_node = _args_from_simple_call_node(_child.node)[0]  # type: ignore[assignment]
                else:  # pragma: no cover
                    # This branch is definitely hit - bug in coverage?
                    continue
                if (
                    isinstance(index_node.base, NameNode)
                    and isinstance(index_node.index, NameNode)
                    and (
                        (_base_name := _name_from_name_node(index_node.base))
                        == _name_from_name_node(args[0])
                    )
                    and (
                        (_index_name := _name_from_name_node(index_node.index))
                        == _name_from_name_node(node.target.args[0])
                    )
                ):
                    _target_name = _name_from_name_node(node.target.args[1])
                    violations.append(
                        (
                            index_node.base.pos[1],
                            index_node.base.pos[2] + 1,
                            "unnecessary list index lookup: use "
                            f"`{_target_name.lstrip('_')}` instead of "
                            f"`{_base_name}[{_index_name}]`",
                        ),
                    )

        if (
            isinstance(node, SingleAssignmentNode)
            and isinstance(node.lhs, NameNode)
            and _name_from_name_node(node.lhs) == "__all__"
            and isinstance(node.rhs, ListNode)
        ):
            exported_imports.extend(
                _value_from_unicode_node(_import)
                for _import in node.rhs.args
                if isinstance(_import, UnicodeNode)
            )

        if isinstance(node, ForInStatNode):
            loop_vars: frozenset[str] = frozenset(
                _name_from_name_node(n) for n in _iter_target_name_nodes(node.target)
            )
            if loop_vars:
                for _child in _traverse_loop_body(node.body, loop_vars):
                    if isinstance(_child, ForInStatNode):
                        for _name_node in _iter_target_name_nodes(_child.target):
                            _inner_name = _name_from_name_node(_name_node)
                            if _inner_name in loop_vars and not _inner_name.startswith(
                                "_"
                            ):
                                violations.append(
                                    (
                                        _name_node.pos[1],
                                        _name_node.pos[2] + 1,
                                        f"Outer for loop variable '{_inner_name}' "
                                        "overwritten by inner for-loop target",
                                    ),
                                )

        if isinstance(node, ForInStatNode):
            _target_nodes = list(_iter_target_name_nodes(node.target))
            if _target_nodes:
                _body_names: frozenset[str] = frozenset(
                    _name_from_name_node(_child.node)
                    for _root in filter(None, [node.body, node.else_clause])
                    for _child in traverse(_root)
                    if isinstance(_child.node, NameNode)
                )
                for _target_node in _target_nodes:
                    _target_name = _name_from_name_node(_target_node)
                    if (
                        not _target_name.startswith("_")
                        and _target_name not in _body_names
                    ):
                        violations.append(
                            (
                                _target_node.pos[1],
                                _target_node.pos[2] + 1,
                                f"Loop control variable '{_target_name}' not used "
                                "within the loop body (if this is intended, start the "
                                "name with an underscore)",
                            ),
                        )

    return names, imported_names, exported_imports


def sanitise_input(
    code: str,
    filename: str,
) -> tuple[str, dict[int, str], list[str]]:
    tokens = src_to_tokens(code)
    exclude_lines = {
        token.line
        for token in tokens
        if token.name == "NAME" and token.src in ("include", "DEF")
    }
    code = tokens_to_src(tokens)
    lines = {}
    _dir = os.path.dirname(filename)
    included_texts = []
    for i, line in enumerate(code.splitlines(keepends=True), start=1):
        if i in exclude_lines:
            _file = os.path.join(_dir, line.split()[-1].strip("'\""))
            if os.path.exists(f"{_file}.in"):
                try:
                    with open(f"{_file}.in", encoding="utf-8") as fd:
                        content = fd.read()
                    pyxcontent = Tempita.sub(content)
                except Exception as exc:  # pragma: no cover
                    # If Cython can't parse this file, just skip it.
                    raise CythonParseError from exc
                included_texts.append(pyxcontent)
            elif os.path.exists(_file):
                try:
                    with open(_file, encoding="utf-8") as fd:
                        content = fd.read()
                except Exception as exc:  # pragma: no cover
                    # If Cython can't parse this file, just skip it.
                    raise CythonParseError from exc
                included_texts.append(content)
            lines[i] = "\n"
        else:
            lines[i] = line

    code = "".join(lines.values())
    return code, lines, included_texts


def run_ast_checks(
    code: str,
    filename: str,
    violations: list[tuple[int, int, str]],
    *,
    ban_relative_imports: bool,
) -> dict[int, str]:
    code, lines, included_texts = sanitise_input(code, filename)
    names, imported_names, exported_imports = _traverse_file(
        code,
        filename,
        lines,
        violations=violations,
        ban_relative_imports=ban_relative_imports,
        skip_check=False,
    )

    included_names = []
    for _code in included_texts:
        _code, _lines, __ = sanitise_input(_code, filename)
        _included_names, _, __ = _traverse_file(
            _code,
            filename,
            _lines,
            skip_check=True,
            violations=None,
            ban_relative_imports=False,
        )
        included_names.extend(_included_names)
    for _import in imported_names:
        if _import[0] == "*":
            continue
        if "." in _import[0]:
            # e.g. import foo.bar
            # foo.bar.bat()
            # skip for now so there's no false negative
            continue
        if (
            _import[0] not in [_name[0] for _name in names if _import != _name]
            and _import[0] not in [_name[0] for _name in included_names]
            and _import[0] not in exported_imports
        ):
            violations.append(
                (
                    _import[1],
                    _import[2] + 1,
                    f"'{_import[0]}' imported but unused",
                )
            )
    return lines


def run_pycodestyle(
    line_length: int,
    filename: str,
    violations: list[tuple[int, int, str]],
    ignore: set[str],
) -> None:
    output = subprocess.run(
        [
            "pycodestyle",
            f"--ignore={','.join(PYCODESTYLE_CODES | ignore)}",
            f"--max-line-length={line_length}",
            "--format=%(row)d:%(col)d:%(code)s %(text)s",
            filename,
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    extra_lines = output.stdout.splitlines()
    for extra_line in extra_lines:
        if re.search(r"^\d+:\d+:", extra_line) is None:
            # could be an extra line with pycodestyle statistics
            continue
        _lineno, _col, message = extra_line.split(":", maxsplit=2)
        violations.append((int(_lineno), int(_col), message))


def _main(  # noqa: PLR0913
    code: str,
    filename: str,
    *,
    ext: str,
    line_length: int = 88,
    no_pycodestyle: bool = False,
    ban_relative_imports: bool = False,
    ignore: set[str] | None = None,
) -> int:
    if ignore is None:
        ignore = set()
    assert ignore is not None  # help mypy
    violations: list[tuple[int, int, str]] = []
    if not no_pycodestyle:
        run_pycodestyle(line_length, filename, violations, ignore)

    lines = {}
    if ext == ".pyx":
        with contextlib.suppress(CythonParseError):
            lines = run_ast_checks(
                code, filename, violations, ban_relative_imports=ban_relative_imports
            )

    ret = 0
    for lineno, col, message in sorted(violations):
        if re.search(PRAGMA, lines.get(lineno, "")) is not None:
            continue
        print(f"{filename}:{lineno}:{col}: {message}")
        ret = 1

    return ret


def traverse(tree: Node) -> Iterator[NodeParent]:
    nodes = [NodeParent(tree, None)]

    while nodes:
        node_parent = nodes.pop()
        node = node_parent.node
        if node is None:
            continue
        if not hasattr(node, "child_attrs"):
            continue

        node_child_attrs = cast("list[str]", node.child_attrs)
        child_attrs = set(copy.deepcopy(node_child_attrs))
        for attr in MISSING_CHILD_ATTRS:
            if hasattr(node, attr):
                child_attrs.add(attr)

        for attr in child_attrs:
            child = getattr(node, attr)
            if isinstance(child, list):
                nodes.extend([NodeParent(_child, node) for _child in child])
            else:
                nodes.append(NodeParent(child, node))
        yield node_parent


def _get_config(paths: list[pathlib.Path]) -> dict[str, Any]:
    """Get the configuration from a config file

    Search for a .cython-lint.toml or pyproject.toml file in common parent
    directories of the given list of paths.
    """
    root = pathlib.Path(os.path.commonpath(paths))
    root = root.parent if root.is_file() else root

    while root != root.parent:
        for basename in (".cython-lint.toml", "pyproject.toml"):
            config_file = root / basename
            if config_file.is_file():
                config: dict[str, Any] = tomllib.loads(config_file.read_text())
                config = config.get("tool", {}).get("cython-lint", {})
                if config:
                    return config
                # .cython-lint.toml takes precedence over
                # pyproject.toml even when it is empty
                if basename == ".cython-lint.toml":
                    return config

        root = root.parent

    return {}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument(
        "--files",
        help="Regex pattern with which to match files to include",
        required=False,
        default=r"",
    )
    parser.add_argument(
        "--exclude",
        help="Regex pattern with which to match files to exclude",
        required=False,
        default=r"^$",
    )
    # default from black formatter
    parser.add_argument("--max-line-length", type=int, default=88)
    parser.add_argument("--no-pycodestyle", action="store_true")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--ban-relative-imports", action="store_true")
    parser.add_argument(
        "--ignore",
        nargs="*",
        default="",
        help="Comma-separated list of pycodestyle error codes to ignore",
    )
    args = parser.parse_args(argv)
    paths = [pathlib.Path(path).resolve() for path in args.paths]

    # Update defaults from pyproject.toml if present
    config = {k.replace("-", "_"): v for k, v in _get_config(paths).items()}
    parser.set_defaults(**config)
    args = parser.parse_args(argv)

    ret = 0

    if not isinstance(args.ignore, list):
        args.ignore = [args.ignore]
    ignore: set[str] = {code.strip() for s in args.ignore for code in s.split(",")}

    for path in paths:
        if path.is_file():
            filepaths = iter((path,))
        else:
            filepaths = (
                p
                for p in path.rglob("*")
                if re.search(args.files, str(p.as_posix()), re.VERBOSE)
                and not re.search(args.exclude, str(p.as_posix()), re.VERBOSE)
                and not re.search(EXCLUDES, str(p.as_posix()))
                and p.suffix in (".pyx", ".pxd", ".pxi")
            )

        for filepath in filepaths:
            ext = filepath.suffix
            try:
                with open(filepath, encoding="utf-8") as fd:
                    content = fd.read()
            except UnicodeDecodeError:
                continue
            ret |= _main(
                content,
                str(filepath),
                line_length=args.max_line_length,
                no_pycodestyle=args.no_pycodestyle,
                ext=ext,
                ban_relative_imports=args.ban_relative_imports,
                ignore=ignore,
            )
    return ret


if __name__ == "__main__":
    sys.exit(main())
