"""Compare one or more convolution binaries with an independent direct solution."""

import argparse
from pathlib import Path
import shlex
import subprocess
import tempfile

N = 512


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="+",
                        help="quoted executable command, optionally with an MPI launcher")
    args = parser.parse_args()

    # Sparse, asymmetric inputs expose wraparound and transpose errors.
    left = {(0, 0): 2, (2, 6): -3, (511, 510): 5}
    right = {(0, 1): 7, (3, 9): 4, (509, 5): -2}
    expected = [0] * (N * N)
    for (i, j), x in left.items():
        for (k, l), y in right.items():
            expected[((i + k) % N) * N + (j + l) % N] += x * y

    with tempfile.TemporaryDirectory() as directory:
        first, second, output = (Path(directory) / name for name in ("a", "b", "out"))
        for path, pixels in ((first, left), (second, right)):
            with path.open("w") as file:
                for i in range(N):
                    print(*(pixels.get((i, j), 0) for j in range(N)), file=file)

        for command in args.command:
            output.unlink(missing_ok=True)
            subprocess.run(shlex.split(command) + [first, second, output], check=True)
            rows = [line.split() for line in output.read_text().splitlines()]
            if len(rows) != N or any(len(row) != N for row in rows):
                raise SystemExit(f"{command}: expected a {N} x {N} output")
            actual = (float(value) for row in rows for value in row)
            errors = [abs(a - b) for a, b in zip(actual, expected)]
            if not all(error <= 0.001 for error in errors):
                raise SystemExit(f"{command}: direct convolution comparison failed")
            print(f"{command}: max absolute error {max(errors):.4g}")


if __name__ == "__main__":
    main()
