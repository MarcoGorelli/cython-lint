# type: ignore
# ruff: noqa
import copy  # pragma: no cover

from Cython.Compiler.TreeFragment import StringParseContext  # pragma: no cover

from cython_lint.cython_lint import MISSING_CHILD_ATTRS  # pragma: no cover


def _print(name, node, indent):  # pragma: no cover
    if node is None:
        return
    if hasattr(node, "pos"):
        print("    " * indent, name, node, node.pos)
    if hasattr(node, "child_attrs"):
        child_attrs = set(copy.deepcopy(node.child_attrs))
        for attr in MISSING_CHILD_ATTRS:
            if hasattr(node, attr):
                child_attrs.add(attr)

        for attr in child_attrs:
            children = getattr(node, attr)
            if isinstance(children, list):
                for child in children:
                    _print(attr, child, indent + 1)
            else:
                _print(attr, children, indent + 1)


def pretty_print(path):  # pragma: no cover
    from Cython.Compiler.TreeFragment import parse_from_strings

    with open(path, encoding="utf-8") as fd:
        content = fd.read()
    context = StringParseContext(path)
    context.set_language_level(3)
    tree = parse_from_strings(path, content, context=context)
    _print("tree", tree, indent=0)


if __name__ == "__main__":  # pragma: no cover
    import sys

    pretty_print(sys.argv[1])
