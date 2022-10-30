import re  # pragma: no cover

from cython_lint import PYCODESTYLE_CODES  # pragma: no cover

if __name__ == '__main__':  # pragma: no cover
    with open('README.md', encoding='utf-8') as fd:
        content = fd.read()

    codes = re.findall(r' ([E|W]\d+) ', content)

    if sorted(set(codes)) != sorted(set(PYCODESTYLE_CODES)):
        for i in sorted(set(codes)):
            print(f'    "{i}",')
        import sys
        sys.exit(1)
