# type: ignore

def _print(name, node, indent):  # pragma: no cover
    if node is None:
        return
    if hasattr(node, 'pos'):
        print('    '*indent, name, node, node.pos)
    if hasattr(node, 'child_attrs'):
        for attr in node.child_attrs:
            children = getattr(node, attr)
            if isinstance(children, list):
                for child in children:
                    _print(attr, child, indent+1)
            else:
                _print(attr, children, indent+1)


def pretty_print(path):  # pragma: no cover
    from Cython.Compiler.TreeFragment import parse_from_strings
    with open(path, encoding='utf-8') as fd:
        content = fd.read()
    tree = parse_from_strings(path, content)
    _print('tree', tree, indent=0)


if __name__ == '__main__':
    import sys
    pretty_print(sys.argv[1])
