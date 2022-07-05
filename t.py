"""
todo:
    cdef a, b=0, c=0
should become
    cdef a, b, c = *[1, 2, 3]
is it possible to do this non-hackily?
I think so, via tokenization
could how many cvardefs there are?
should be possible to do something...
"""
import os
import argparse
import subprocess
import tempfile
from Cython.Compiler.TreeFragment import parse_from_strings
from Cython.Compiler.Nodes import (
    StatListNode,
    CVarDefNode,
    CFuncDefNode,
    CImportStatNode,
    FromCImportStatNode,
    CSimpleBaseTypeNode,
    MemoryViewSliceTypeNode,
    CPtrDeclaratorNode,
    GILStatNode,
    FusedTypeNode,
)
from Cython.Compiler.ExprNodes import TypecastNode, AmpersandNode
import ast
import re


def replace_cvardef(tokens, i, varnames):
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

    # replace the variables
    # find first variable
    j = i
    while not (tokens[j].name == 'NAME' and tokens[j].src == varnames[0]):
        j += 1
    assignment_idx = j
    assignment = f"{', '.join(varnames)} = {', '.join('0' for _ in range(len(varnames)))}\n"
    line = tokens[assignment_idx].line
    tokens[assignment_idx] = Token(name='NAME', src=assignment, line=line)
    tokens_in_line = []
    j = assignment_idx + 1
    while tokens[j].line is None or tokens[j].line == line:
        tokens_in_line.append(j)
        j += 1
    for j in tokens_in_line:
        tokens[j] = Token(name='PLACEHOLDER', src='')

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
    j = i+1
    # get to the end of the function declaration
    while not (tokens[j].name == 'OP' and tokens[j].src == ':'):
        j += 1
    j -= 1  # go back to statement before colon
    # ignore any whitespace before colon
    while not tokens[j].src.strip():
        j -= 1
    if tokens[j].name == 'NAME' and tokens[j].src == 'nogil':
        tokens[j] = Token(name='PLACEHOLDER', src='')
    # remove any extra whitespace
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j -= 1

def replace_cfuncarg(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='')
    j = i+1
    while not tokens[j].src.strip():  # TODO: tokenize whitespace?
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1

def replace_cdef(tokens, i):
    pass

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
    while not tokens[j].src.strip() and j<len(tokens)-1:
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
    j = i
    while not (tokens[j].name == 'OP' and set(tokens[j].src) == {'*'}):
        j -= 1
    tokens[j] = Token(name='PLACEHOLDER', src='')

def replace_gilstatnode(tokens, i):
    tokens[i] = Token(name='NAME', src='True')
    j = i-1
    while not tokens[j].src.strip():
        j -= 1
    if not tokens[j].name == 'NAME' and tokens[j].src == 'if':
        raise AssertionError('Please report a bug')
    tokens[j] = Token(name='NAME', src='if')


def replace_fusedtype(tokens, i):
    j = i
    while not (tokens[j].name=='OP' and tokens[j].src==':'):
        tokens[j] = Token(name='PLACEHOLDER', src='')
        j += 1
    tokens[j-1] = Token(name='NAME', src='if True')

def visit_cvardefnode(node):
    base_type = node.base_type
    varnames = []
    for declarator in node.declarators:
        while isinstance(declarator, CPtrDeclaratorNode):
            declarator = declarator.base
        varnames.append(declarator.name)
    yield (
        'cvardef',
        base_type.pos[1],
        base_type.pos[2],
        {'varnames': varnames},
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
    if node.name is None:
        # no C type declared.
        return
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

def visit_gilstatnode(node):
    yield (
        'gilstat',
        node.pos[1],
        node.pos[2],
    )

def visit_fusedtypenode(node):
    yield (
        'fusedtype',
        node.pos[1],
        node.pos[2],
    )
import collections
from tokenize_rt import src_to_tokens, tokens_to_src, reversed_enumerate, Token

# ok, no, let's do something totally different
# let's have...let's do...
# some list of replacements

def main(filename, append_config):
    with open(filename, encoding='utf-8') as fd:
        code = fd.read()
    tokens = src_to_tokens(code)
    exclude_lines = set()
    for token in tokens:
        if token.name == 'NAME' and token.src == 'include':
            exclude_lines.add(token.line)
    code = ''.join([line for i, line in enumerate(code.splitlines(keepends=True), start=1) if i not in exclude_lines])
    tree = parse_from_strings(filename, code)
    replacements = traverse(tree)
    tokens = src_to_tokens(code)

    for n, token in reversed_enumerate(tokens):
        key = (token.line, token.utf8_byte_offset)
        if key in replacements:
            for name, kwargs in replacements.pop(key):
                if name == 'cvardef':
                    replace_cvardef(tokens, n, kwargs[0]['varnames'])
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
                elif name == 'cconsttype':
                    replace_cconsttypenode(tokens, n)
                elif name == 'memoryviewslicetype':
                    replace_memoryviewslicetypenode(tokens, n)
                elif name == 'ampersand':
                    replace_ampersandnode(tokens, n)
                elif name == 'cptrdeclarator':
                    replace_cptrdeclaratornode(tokens, n)
                elif name == 'gilstat':
                    replace_gilstatnode(tokens, n)
                elif name == 'fusedtype':
                    replace_fusedtype(tokens, n)
    newsrc = tokens_to_src(tokens)
    import sys
    try:
        ast.parse(newsrc)
    except SyntaxError as exp:
        if str(exp).startswith('cannot assign to literal'):
            print('limitation of cython-lint, sorry')
        else:
            print(repr(exp))
        sys.exit(1)

    fd, path = tempfile.mkstemp(
        dir=os.path.dirname(filename),
        prefix=os.path.basename(filename),
        suffix='.py',
    )
    try:
        with open(fd, 'w', encoding='utf-8') as f:
            f.write(newsrc)
        command = ['python', '-m', 'flake8', path]
        if append_config is not None:
            command += ['--append-config', append_config]
        output = subprocess.run(command, capture_output=True, text=True)
        sys.stdout.write(
            output.stdout.replace(
                os.path.basename(path),
                os.path.basename(filename),
            ),
        )
    finally:
        os.remove(path)


    # want to get to: ast.parse(code)

def _run_flake8(filename: str) -> dict[int, set[str]]:
    cmd = (sys.executable, '-mflake8', filename)
    out, _ = subprocess.Popen(cmd, stdout=subprocess.PIPE).communicate()
    ret: dict[int, set[str]] = collections.defaultdict(set)
    for line in out.decode().splitlines():
        # TODO: use --no-show-source when that is released instead
        try:
            lineno, code = line.split('\t')
        except ValueError:
            pass  # ignore additional output besides our --format
        else:
            ret[int(lineno)].add(code)
    return ret

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
        'GILStatNode': visit_gilstatnode,
        'FusedTypeNode': visit_fusedtypenode,
    }

    while nodes:
        node = nodes.pop()
        if node is None:
            continue

        func = funcs.get(type(node).__name__)
        if func is not None:
            iterator = func(node)
            if iterator is not None:
                for name, line, col, *kwargs in iterator:
                    replacements[line, col].append((name, kwargs))

        child_attrs = node.child_attrs
        if 'declarator' not in child_attrs and hasattr(node, 'declarator'):
            # noticed this in size(int*)
            # bug?
            child_attrs.append('declarator')
        for attr in child_attrs:
            child = getattr(node, attr)
            if isinstance(child, list):
                nodes.extend(child)
            else:
                nodes.append(child)
    return replacements

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='*')
    parser.add_argument('--append-config', required=False)
    args = parser.parse_args()
    for path in args.paths:
        main(path, args.append_config)
