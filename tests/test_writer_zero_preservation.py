"""Regression tests for CHGCAR-like FukuiGrid output with genuine zero values.

Run in the repository root:
    python3 -m unittest discover -s tests -v

FukuiGrid's declared NumPy/SciPy/pandas/Matplotlib dependencies must be installed.
"""
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from FukuiGrid import write_fukui_file


def make_chgcar_header(path, shape):
    # Minimal VASP-like CHGCAR header: one H atom, Direct coordinates.
    # write_fukui_file copies exactly NATOMS+10=11 header lines.
    lines = [
        "Synthetic CHGCAR for writer regression\n",
        "1.0\n",
        "1.0 0.0 0.0\n",
        "0.0 1.0 0.0\n",
        "0.0 0.0 1.0\n",
        "H\n",
        "1\n",
        "Direct\n",
        "0.0 0.0 0.0\n",
        "\n",
        f"{shape[0]} {shape[1]} {shape[2]}\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")
    return lines


class TestFukuiWriter(unittest.TestCase):
    def check_round_trip(self, shape, data):
        with TemporaryDirectory() as d:
            source = Path(d) / "CHGCAR"
            target = Path(d) / "FUKUI"
            header = make_chgcar_header(source, shape)
            arr = np.asarray(data, dtype=float)
            self.assertEqual(arr.size, math.prod(shape))
            write_fukui_file(str(source), 1, arr, str(target))
            lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
            self.assertEqual(lines[:11], header)
            payload = lines[11:]
            # Five data points per full row, no padding in a short final row.
            self.assertEqual(len(payload), (len(data) + 4) // 5)
            self.assertEqual([len(line.split()) for line in payload],
                             [min(5, len(data) - i) for i in range(0, len(data), 5)])
            values = [float(token) for line in payload for token in line.split()]
            self.assertEqual(len(values), len(data))
            np.testing.assert_allclose(values, data, atol=1e-11, rtol=1e-11)
            self.assertEqual(sum(x == 0 for x in values),
                             sum(x == 0 for x in data))

    def test_true_zeros_negative_values_and_short_last_row(self):
        self.check_round_trip((2, 2, 2), [0.125, 0.0, -0.25, 0.5, 0.0, 1.0, -0.75, 0.375])

    def test_exactly_five_points_with_zero(self):
        self.check_round_trip((1, 1, 5), [0.0, -1.0, 2.0, 0.0, 3.0])

    def test_all_zero_grid(self):
        self.check_round_trip((2, 2, 1), [0.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
