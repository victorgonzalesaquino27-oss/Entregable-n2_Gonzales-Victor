from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analisis_modal import AnalysisInputError, parse_floor_data  # noqa: E402


class FloorDataParserTests(unittest.TestCase):
    def test_parses_spanish_semicolon_template(self) -> None:
        parsed = parse_floor_data(
            "Piso;Altura;Masa;Rigidez\n"
            "1;11;2.16;108\n"
            "2;11;1.92;90\n"
        )
        self.assertEqual(parsed["floor"].tolist(), [1.0, 2.0])
        np.testing.assert_allclose(parsed["height"], [11.0, 11.0])
        np.testing.assert_allclose(parsed["mass"], [2.16, 1.92])
        np.testing.assert_allclose(parsed["stiffness"], [108.0, 90.0])

    def test_parses_english_tab_data_without_floor_column(self) -> None:
        parsed = parse_floor_data("Height\tMass\tStiffness\n3.2\t1200\t45000\n")
        self.assertEqual(parsed["floor"].tolist(), [1.0])
        self.assertEqual(parsed.loc[0, "height"], 3.2)

    def test_rejects_nonsequential_floor_numbers(self) -> None:
        with self.assertRaises(AnalysisInputError):
            parse_floor_data(
                "Piso;Altura;Masa;Rigidez\n1;3;1000;50000\n3;3;1000;50000\n"
            )


if __name__ == "__main__":
    unittest.main()
