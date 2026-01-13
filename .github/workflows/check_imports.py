#!/usr/bin/env python3

import argparse
import importlib.util
import pathlib
import sys


def check_imports(args: argparse.Namespace) -> None:
    src_path = pathlib.Path(args.src_path)
    num_files = 0
    for pyfile in src_path.rglob('*.py'):
        print('Trying to import', pyfile)
        module_name = pyfile.with_suffix('').as_posix().replace('/', '.')
        spec = importlib.util.spec_from_file_location(
            name=module_name,
            location=pyfile,
        )
        if spec is None:
            print(f'Failed to load {pyfile}', file=sys.stderr)
            raise SystemExit(1)

        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            num_files += 1
        except Exception as err:
            print(f'Error importing {module_name}: {err}', file=sys.stderr)
            raise SystemExit(1) from err

    print(f'All {num_files} imports checked successfully.')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Check imports in Python files',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--src_path',
        type=str,
        help='Path to the source directory',
        default='src',
    )
    args = parser.parse_args()
    check_imports(args)


if __name__ == '__main__':
    main()
