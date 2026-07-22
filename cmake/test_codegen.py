import argparse
import re
import sys

ALL_ARCHS = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-file",
        nargs=1,
        required=True,
        help="Absolute path to the codegen test source file",
    )
    parser.add_argument(
        "--cuobjdmp-bin",
        nargs=1,
        required=True,
        help="Absolute path to cuobjdump binary",
    )
    parser.add_argument(
        "--filecheck-bin",
        nargs=1,
        required=True,
        help="Absolute path to FileCheck binary",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.source_file) as src_file:
        SRC = src_file.read()

    re.search("(^|\n)[ \t]*//[ \t]*CODE:[ \t]*([^\r\n]*)", SRC)

    return 0


if __name__ == "__main__":
    sys.exit(main())
