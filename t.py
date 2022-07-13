"""
probably, remove replace_cenum, and do it with TDD

what to do about
    enum: A
?
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
    CFuncDeclaratorNode,
    TemplatedTypeNode,
    CStructOrUnionDefNode,
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

def replace_cvardef(tokens, i, declarator, varnames, end_pos):
    j = i-1
    while not tokens[j].src.strip():
        j -= 1
    if tokens[j].name == 'NAME' and tokens[j].src == 'cdef':
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
        while not tokens[j].src.strip():
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j += 1
    j = i
    while not (tokens[j].line == declarator[1] and tokens[j].utf8_byte_offset == declarator[2]):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

    # replace the variables
    # find first variable
    j = i
    while not (tokens[j].name == 'NAME' and tokens[j].src == varnames[0]):
        j += 1
    assignment_idx = j

    assignment = f"{', '.join(varnames)} = {', '.join('0' for _ in range(len(varnames)))}"
    tokens[assignment_idx] = tokens[assignment_idx]._replace(src=assignment)
    if tokens[assignment_idx].line == end_pos[1] and tokens[assignment_idx].utf8_byte_offset == end_pos[2]:
        pass
        # return
    # we've ruled out multi-line assignments, so...just go on until the end of the line
    j = assignment_idx + 1
    while not tokens[j].name == 'NEWLINE':
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

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
    while not (tokens[j].name=='OP' and tokens[j].src=='['):
        j += 1
    while not (tokens[j].name=='OP' and tokens[j].src==']'):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)

def replace_csimplebasetype(tokens, i, name):
    return
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


def replace_fusedtype(tokens, i, name):
    j = i
    while not (tokens[j].name=='NAME' and tokens[j].src==name):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
    tokens[j-1] = tokens[j-1]._replace(src='class ')

def replace_fusedtype_child(tokens, i, name):
    j = i
    tokens[j] = tokens[j]._replace(src=f'{name} = 0')
    j += 1
    while not tokens[j].name == 'NEWLINE':
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1


def replace_cclassdefnode(tokens, i, objstructname, module_name, class_name, as_name):
    j = i-1
    while not (tokens[j].name == 'NAME' and tokens[j].src in ('cdef', 'ctypedef')):
        j -= 1
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
    j += 1
    while not tokens[j].src.strip():
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
    if objstructname is not None:
        # find location
        j = i+1
        while not (tokens[j].name == 'NAME' and tokens[j].src == objstructname):
            j += 1
        objstruct_pos = j
        j = objstruct_pos
        while not (tokens[j].name == 'OP' and tokens[j].src == '['):
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j -= 1
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        while not tokens[j].src.strip():
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j -= 1
        j = objstruct_pos
        while not (tokens[j].name == 'OP' and tokens[j].src == ']'):
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j += 1
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)

        # use as_name
        if module_name is not None and module_name != '':
            j = i
            while not (tokens[j].name == 'NAME' and tokens[j].src == module_name):
                j += 1
            while not (tokens[j].name == 'NAME' and tokens[j].src == class_name):
                tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
                j += 1



def replace_cenumdefnode(tokens, i, name):
    if name is None:
        tokens[i] = tokens[i]._replace(src='lambda')
        return
    if tokens[i].name == 'NAME' and tokens[i].src == 'enum':
        # bug in Cython?
        j = i
        while not (tokens[j].name == 'NAME' and tokens[j].src in ('cdef', 'def', 'ctypedef')):
            j -= 1
        i = j
    tokens[i] = tokens[i]._replace(src='class ')
    j = i+1
    while not (tokens[j].name == 'NAME' and tokens[j].src == name):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def replace_ctuplebasetypenode(tokens, i):
    return
    j = i
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
    while not (tokens[j].name == 'NAME' and tokens[j].src in ('cdef', 'cpdef')):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j -= 1

def replace_cargdeclnode(tokens, i, declarator, not_none):
    j = i
    tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
    while not (tokens[j].line == declarator[1] and tokens[j].utf8_byte_offset == declarator[2]):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1
    if not_none:
        j += 1
        # skip whitespaces
        while not tokens[j].src.strip():
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j += 1
        if tokens[j].name == 'NAME' and tokens[j].src == 'not':
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j += 1
        else:
            raise AssertionError('please report bug')
        while not tokens[j].src.strip():
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j += 1
        if tokens[j].name == 'NAME' and tokens[j].src == 'None':
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j += 1
        else:
            raise AssertionError('please report bug')
        while not tokens[j].src.strip():
            tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
            j += 1

def replace_cstructoruniondefnode(tokens, i, name):
    j = i
    tokens[j] = tokens[j]._replace(src='class ')
    j += 1
    while not (tokens[j].name == 'NAME' and tokens[j].src == name):
        tokens[j] = Token(name='PLACEHOLDER', src='', line=tokens[j].line, utf8_byte_offset=tokens[j].utf8_byte_offset)
        j += 1

def visit_cvardefnode(node):
    base_type = node.base_type
    varnames = []
    first_declarator = None
    for declarator in node.declarators:
        while isinstance(declarator, CPtrDeclaratorNode):
            declarator = declarator.base
        if isinstance(declarator, CFuncDeclaratorNode):
            declarator = declarator.base
        while isinstance(declarator, CPtrDeclaratorNode):
            declarator = declarator.base
        varnames.append(declarator.name)
        if first_declarator is None:
            first_declarator = declarator
    yield (
         'cvardef',
         node.pos[1],
         node.pos[2],
         {
             'varnames': varnames,
             'first_declarator': first_declarator.pos,
             'end_pos': node.end_pos(),
         },
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
        {'name': node.name},
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
    typenames = []
    for _type in node.types:
        if isinstance(_type, TemplatedTypeNode):
            _type = _type.base_type_node
        yield (
            'fusedtype_child',
            _type.pos[1],
            _type.pos[2],
            {'name': _type.name},
        )
    yield (
        'fusedtype',
        node.pos[1],
        node.pos[2],
        {
            'name': node.name,
        },
    )

def visit_cclassdefnode(node):
    yield (
        'cclassdef',
        node.pos[1],
        node.pos[2],
        {
            'objstruct_name': getattr(node, 'objstruct_name', None),
            'module_name': getattr(node, 'module_name', None),
            'class_name': getattr(node, 'class_name', None),
            'as_name': getattr(node, 'as_name', None),
        },
    )

def visit_cenumdefnode(node):
    yield (
        'cenumdef',
        node.pos[1],
        node.pos[2],
        {'name': node.name},
    )
def visit_ctuplebasetypenode(node):
    yield (
        'ctuplebasetype',
        node.pos[1],
        node.pos[2],
    )
def visit_cargdeclnode(node):
    if (
        node.annotation is not None 
        or (
            isinstance(node.base_type, CSimpleBaseTypeNode)
            and (
                node.base_type.name is None
                or node.base_type.name == 'self'
            )
        )
        or (
            isinstance(node.declarator, CNameDeclaratorNode)
            and node.declarator.name == ''
            
        )
    ):
        # has annotation, so no type to delete
        return
    if node.is_self_arg:
        # don't replace self
        return
    yield (
        'cargdecl',
        node.pos[1],
        node.pos[2],
        {
            'declarator': node.declarator.pos,
            'not_none': node.not_none,
        },
    )
def visit_cstructoruniondefnode(node):
    yield (
        'cstructorunion',
        node.pos[1],
        node.pos[2],
        {'name': node.name},
    )
import collections
from tokenize_rt import src_to_tokens, tokens_to_src, reversed_enumerate, Token

# ok, no, let's do something totally different
# let's have...let's do...
# some list of replacements

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
    replacements = traverse(tree, filename)
    tokens = src_to_tokens(code)
    tokenize_replacements(tokens)

    for n, token in reversed_enumerate(tokens):
        key = (token.line, token.utf8_byte_offset)
        if key in replacements:
            for name, kwargs in replacements.pop(key):
                if name == 'cvardef':
                    try:
                        replace_cvardef(
                            tokens,
                            n,
                            kwargs[0]['first_declarator'],
                            kwargs[0]['varnames'],
                            kwargs[0]['end_pos'],
                        )
                    except:
                        print(filename)
                elif name == 'cdef':
                    replace_cdef(tokens, n)
                elif name == 'typecast':
                    replace_typecast(tokens, n)
                elif name == 'cfuncdef':
                    replace_cfuncdef(
                    tokens,
                    n,
                    kwargs[0]['declarator'],
                )
                elif name == 'cfuncarg':
                    replace_cfuncarg(tokens, n)
                elif name == 'fromcimport':
                    replace_fromcimportstat(tokens, n)
                elif name == 'cimport':
                    replace_cimportstat(tokens, n)
                elif name == 'templatedtype':
                    replace_templatedtype(tokens, n)
                elif name == 'csimplebasetype':
                    replace_csimplebasetype(tokens, n, kwargs[0]['name'])
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
                    replace_fusedtype(tokens, n, kwargs[0]['name'])
                elif name == 'fusedtype_child':
                    replace_fusedtype_child(tokens, n, kwargs[0]['name'])
                elif name == 'cclassdef':
                    replace_cclassdefnode(
                        tokens,
                        n,
                        kwargs[0]['objstruct_name'],
                        kwargs[0]['module_name'],
                        kwargs[0]['class_name'],
                        kwargs[0]['as_name'],
                    )
                elif name == 'cenumdef':
                    replace_cenumdefnode(tokens, n, kwargs[0]['name'])
                elif name == 'ctuplebasetype':
                    replace_ctuplebasetypenode(tokens, n)
                elif name == 'cargdecl':
                    replace_cargdeclnode(
                        tokens,
                        n,
                        kwargs[0]['declarator'],
                        kwargs[0]['not_none'],
                    )
                elif name == 'cstructorunion':
                    replace_cstructoruniondefnode(
                        tokens,
                        n,
                        kwargs[0]['name'],
                    )
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



def traverse(tree, filename):
    # check if child isn't []
    nodes = [tree]
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
        'CArgDeclNode': visit_cargdeclnode,
        'CStructOrUnionDefNode': visit_cstructoruniondefnode,
    }
    while nodes:
        node = nodes.pop()
        if node is None:
            continue

        if isinstance(node, CVarDefNode):
            if node.pos[1] != node.end_pos()[1]:
                raise NotImplementedError(
                    f'{filename}:{node.pos[1]}:{node.pos[2]} '
                    'Variable declarations spanning multiple '
                    'line are not yet supported. '
                    'You might want to declare the variable '
                    'on one line, and assign it on another.'
                )


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
        if isinstance(node, FusedTypeNode):
            child_attrs.append('types')
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
