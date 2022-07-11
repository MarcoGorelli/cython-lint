"""
we're gonna have to find:
    position of the first declarator
    position of the first type
    delete everything in between?

one strategy could be:
    1. delete everything before first declarator


cvardef type var: this is just a cvardefnode, and var is the first declarator
cvardef type var, var2: also cvardefnode, with multiple declarators
cvardef public type var: first declarator is var

what about in a function? that cargdecl, so totally different, no need to worry?


need to get:
    cdef extern from "Python.h":
        Py_ssize_t PY_SSIZE_T_MAX

holy shit, also

    cdef:
        public int foo

is valid

wtf...need a more robust way to find these, just deleting the previous one won't do
we need to have the start and end, and delete in between

ok, that's enough for now though? is that all a cvardefnode?
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
    CClassDefNode,
    CArgDeclNode,
    CNameDeclaratorNode,
    CEnumDefNode,
    CTupleBaseTypeNode,
)
from Cython.Compiler.ExprNodes import TypecastNode, AmpersandNode
import ast
import re

def _delete_base_type(tokens, i):
    j = i
    open_paren = False
    while (tokens[j].src.strip() or open_paren):
        if tokens[j].name == 'OP' and tokens[j].src == '[':
            open_paren = True
        elif tokens[j].name == 'OP' and tokens[j].src == ']':
            open_paren = False
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
    # remove any trailing whitespace
    while not tokens[j].src.strip() and j<len(tokens)-1:
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def replace_cvardef(tokens, i, varnames):
    #_delete_base_type(tokens, i)
    j = i-1
    while not tokens[j].src.strip():
        j -= 1
    k = j-1
    while not tokens[k].src.strip():
        k -= 1
    m = k-1
    while not tokens[m].src.strip():
        m -= 1
    n = m-1
    while not tokens[n].src.strip():
        n -= 1
    o = n-1
    while not tokens[o].src.strip():
        o -= 1
    if (
        tokens[j].name == 'OP'
        and tokens[j].src == ':'
        and tokens[k].name == 'NAME'
        and tokens[k].src == 'cdef'
    ):
        tokens[k] = tokens[k]._replace(src='if True')
    elif (
        tokens[j].name == 'OP'
        and tokens[j].src == ':'
        and tokens[k].name == 'NAME'
        and tokens[k].src in ('readonly', 'public')
        and tokens[m].name == 'NAME'
        and tokens[m].src == 'cdef'
    ):
        tokens[m] = tokens[m]._replace(src='if True')
        tokens[k] = Token(name='PLACEHOLDER', src='', line=tokens[k].line, utf8_byte_offset=tokens[k].utf8_byte_offset)
        # todo: delete some whitespace
    elif (
        tokens[j].name == 'OP'
        and tokens[j].src == ':'
        and tokens[m].name == 'NAME'
        and tokens[m].src == 'from'
        and tokens[n].name == 'NAME'
        and tokens[n].src == 'extern'
        and tokens[o].name == 'NAME'
        and tokens[o].src == 'cdef'
    ):
        tokens[o] = tokens[o]._replace(src='if True')
        tokens[n] = Token(name='PLACEHOLDER', src='', line=tokens[n].line, utf8_byte_offset=tokens[n].utf8_byte_offset)
        tokens[m] = Token(name='PLACEHOLDER', src='', line=tokens[m].line, utf8_byte_offset=tokens[m].utf8_byte_offset)
        tokens[k] = Token(name='PLACEHOLDER', src='', line=tokens[k].line, utf8_byte_offset=tokens[k].utf8_byte_offset)
        # todo: delete some whitespace
    elif tokens[j].name == 'NAME' and tokens[j].src == 'cdef':
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j = j+1
        while not tokens[j].src.strip():
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j += 1

    # replace the variables
    # find first variable
    j = i
    while not (tokens[j].name == 'NAME' and tokens[j].src == varnames[0]):
        j += 1
    assignment_idx = j
    assignment = f"{', '.join(varnames)} = {', '.join('0' for _ in range(len(varnames)))}\n"
    line = tokens[assignment_idx].line
    tokens[assignment_idx] = tokens[assignment_idx]._replace(src=assignment)
    tokens_in_line = []
    j = assignment_idx + 1
    while tokens[j].line is None or tokens[j].line == line:
        tokens_in_line.append(j)
        j += 1
    for j in tokens_in_line:
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)

def replace_cfuncdef(tokens, i, declarator):
    # let's get to the opening paren
    j = i
    while not (tokens[j].line == declarator[1] and tokens[j].utf8_byte_offset == declarator[2]):
        j += 1
    # go back to name before paren
    j -= 1
    while not tokens[j].name == 'NAME':
        j -= 1
    for k in range(i, j):
        tokens[k] = Token(name='PLACEHOLDER', src='', line=tokens[k].line, utf8_byte_offset=tokens[k].utf8_byte_offset)
    j = i
    while not (tokens[j].name == 'NAME' and tokens[j].src in ('cdef', 'cpdef')):
        j -= 1
    tokens[j] = tokens[j]._replace(src='def')
    # get to the end of the function declaration
    while not (tokens[j].name == 'OP' and tokens[j].src == ':'):
        j += 1
    # go back to closing paren
    j -= 1
    while not (tokens[j].name=='OP' and tokens[j].src == ')'):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j -= 1

def replace_cfuncarg(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='', line=tokens[i].line, utf8_byte_offset=tokens[i].utf8_byte_offset)
    j = i+1
    while not tokens[j].src.strip():  # TODO: tokenize whitespace?
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def replace_cdef(tokens, i):
    pass

def replace_cdefblock(tokens, i):
    j = i-1
    while not (tokens[j].name=='NAME' and tokens[j].src=='cdef'):
        j -= 1
    tokens[j] = tokens[j]._replace(src='if True')
    j += 1
    while not (tokens[j].name == 'OP' and tokens[j].src == ':'):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def replace_typecast(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='', line=tokens[i].line, utf8_byte_offset=tokens[i].utf8_byte_offset)
    j = i+1
    while not (tokens[j].name == 'OP' and tokens[j].src == '>'):
        j += 1
    for _i in range(i, j+1):
        tokens[_i] = Token(name='PLACEHOLDER', src='', line=tokens[_i].line, utf8_byte_offset=tokens[_i].utf8_byte_offset)


def replace_fromcimportstat(tokens, i):
    j = i+1
    while not (tokens[j].name=='NAME' and tokens[j].src=='cimport'):
        j += 1
    tokens[j] = tokens[j]._replace(src='import')

def replace_cimportstat(tokens, i):
    j = i-1
    while not (tokens[j].name=='NAME' and tokens[j].src=='cimport'):
        j -= 1
    tokens[j] = tokens[j]._replace(src='import')

def replace_templatedtype(tokens, i):
    return
    j = i
    while not (tokens[j].name=='OP' and tokens[j].src==']'):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def replace_csimplebasetype(tokens, i):
    _delete_base_type(tokens, i)

def replace_cconsttypenode(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='', line=tokens[i].line, utf8_byte_offset=tokens[i].utf8_byte_offset)
    j = i+1
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def replace_memoryviewslicetypenode(tokens, i):
    return
    j = i
    while not (tokens[j].name=='OP' and tokens[j].src=='['):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j -= 1
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
    j = i+1
    while not (tokens[j].name=='OP' and tokens[j].src==']'):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
    j = j+1
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def replace_ampersandnode(tokens, i):
    tokens[i] = Token(name='PLACEHOLDER', src='', line=tokens[i].line, utf8_byte_offset=tokens[i].utf8_byte_offset)

def replace_cptrdeclaratornode(tokens, i):
    j = i
    #while not (tokens[j].name == 'OP' and set(tokens[j].src) == {'*'}):
    #    j -= 1
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)

def replace_gilstatnode(tokens, i):
    tokens[i] = tokens[i]._replace(src='True')
    j = i-1
    while not tokens[j].src.strip():
        j -= 1
    if not tokens[j].name == 'NAME' and tokens[j].src == 'if':
        raise AssertionError('Please report a bug')
    tokens[j] = tokens[j]._replace(src='if')


def replace_fusedtype(tokens, i):
    j = i
    while not (tokens[j].name=='OP' and tokens[j].src==':'):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
    tokens[j-1] = tokens[j-1]._replace(src='if True')

def replace_cclassdefnode(tokens, i):
    j = i-1
    while not (tokens[j].name == 'NAME' and tokens[j].src == 'cdef'):
        j -= 1
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
    j += 1
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def replace_cenumdefnode(tokens, i):
    tokens[i] = tokens[i]._replace(src='class')
    j = i-1
    while not (tokens[j].name == 'NAME' and tokens[j].src in ('cdef', 'cpdef')):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j -= 1
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)

def replace_ctuplebasetypenode(tokens, i):
    pass

def visit_cvardefnode(node):
    base_type = node.base_type
    varnames = []
    for declarator in node.declarators:
        while isinstance(declarator, CPtrDeclaratorNode):
            declarator = declarator.base
        varnames.append(declarator.name)
    yield (
        'cvardef',
        node.pos[1],
        node.pos[2],
        {'varnames': varnames},
    )
    if hasattr(node, '_parent'):
        yield (
            'cdefblock',
            node.pos[1],
            node.pos[2],
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
        {'declarator': node.declarator.pos},
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
    if node.name is None or node.is_self_arg:
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

def visit_cclassdefnode(node):
    yield (
        'cclassdef',
        node.pos[1],
        node.pos[2],
    )

def visit_cenumdefnode(node):
    yield (
        'cenumdef',
        node.pos[1],
        node.pos[2],
    )
def visit_ctuplebasetypenode(node):
    yield (
        'ctuplebasetype',
        node.pos[1],
        node.pos[2],
    )
import collections
from tokenize_rt import src_to_tokens, tokens_to_src, reversed_enumerate, Token

# ok, no, let's do something totally different
# let's have...let's do...
# some list of replacements

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
                elif name == 'cdefblock':
                    replace_cdefblock(tokens, n)
                elif name == 'typecast':
                    replace_typecast(tokens, n)
                elif name == 'cfuncdef':
                    replace_cfuncdef(tokens, n, kwargs[0]['declarator'])
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
                elif name == 'cclassdef':
                    replace_cclassdefnode(tokens, n)
                elif name == 'cenumdef':
                    replace_cenumdefnode(tokens, n)
                elif name == 'ctuplebasetype':
                    replace_ctuplebasetypenode(tokens, n)
    newsrc = tokens_to_src(tokens)
    return newsrc

def main(code, filename, append_config):
    newsrc = transform(code, filename)
    if False:
        with open(filename, 'w') as fd:
            fd.write(newsrc)
    print(newsrc)
    import sys
    try:
        ast.parse(newsrc)
    except SyntaxError as exp:
        if str(exp).startswith('cannot assign to literal'):
            print('limitation of cython-lint, sorry')
        else:
            print(f'{filename}: {repr(exp)}')
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


def skip_typeless_arg(child, parent):
    if (
        isinstance(child, CSimpleBaseTypeNode)
        and isinstance(parent, CArgDeclNode)
        and isinstance(parent.declarator, CNameDeclaratorNode)
        and not parent.declarator.name
    ):
        return True
    return False


def traverse(tree):
    # check if child isn't []
    nodes = [tree]
    breakpoint()
    replacements = collections.defaultdict(list)

    funcs = {
        'CVarDefNode': visit_cvardefnode,
        'CFuncDefNode': visit_cfuncdefnode,
        'TypecastNode': visit_typecastnode,
        'FromCImportStatNode': visit_fromcimportstatnode,
        #'StatListNode': visit_statlistnode,
        'CImportStatNode': visit_cimportstatnode,
        'TemplatedTypeNode': visit_templatedtypenode,
        'CSimpleBaseTypeNode': visit_csimplebasetypenode,
        'CConstTypeNode': visit_cconsttypenode,
        'MemoryViewSliceTypeNode': visit_memoryviewslicetypenode,
        'AmpersandNode': visit_ampersandnode,
        'CPtrDeclaratorNode': visit_cptrdeclaratornode,
        'GILStatNode': visit_gilstatnode,
        'FusedTypeNode': visit_fusedtypenode,
        'CClassDefNode': visit_cclassdefnode,
        'CEnumDefNode': visit_cenumdefnode,
        'CTupleBaseTypeNode': visit_ctuplebasetypenode,
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

            if skip_typeless_arg(child, node):
                continue
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
        with open(path, encoding='utf-8') as fd:
            content = fd.read()
        main(content, path, args.append_config)
