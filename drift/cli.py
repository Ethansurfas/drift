"""Drift CLI — drift run, drift build, drift check."""
import sys
import os

from drift.lexer import Lexer
from drift.parser import Parser
from drift.transpiler import Transpiler
from drift.errors import DriftError


def main():
    if len(sys.argv) < 2:
        print("Usage: drift <command> [file.drift]", file=sys.stderr)
        print("Commands: run, build, check", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command in ("run", "build", "check"):
        if len(sys.argv) < 3:
            print(f"Usage: drift {command} <file.drift>", file=sys.stderr)
            sys.exit(1)
        filepath = sys.argv[2]
        if not os.path.exists(filepath):
            print(f"Error: file not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        with open(filepath) as f:
            source = f.read()

        try:
            tokens = Lexer(source).tokenize()
            tree = Parser(tokens).parse()

            if command == "check":
                print(f"OK: {filepath}")
                sys.exit(0)

            python_code = Transpiler(tree).transpile()

            if command == "build":
                out_path = filepath.replace(".drift", ".py")
                with open(out_path, "w") as out:
                    out.write(python_code)
                print(f"Built: {out_path}")
                sys.exit(0)

            if command == "run":
                exec(python_code, {"__name__": "__main__"})

        except DriftError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
