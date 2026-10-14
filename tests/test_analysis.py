from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analisis_modal import (  # noqa: E402
    AnalysisInputError,
    SpectrumParameters,
    analyze_building,
    evaluate_spectrum,
    shear_building_stiffness,
)
from analisis_modal.data_io import convert_floor_data  # noqa: E402
from analisis_modal.plots import make_figures  # noqa: E402
from analisis_modal.report import generate_pdf_report  # noqa: E402
import pandas as pd  # noqa: E402


class ModalAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.masses = np.array([2.16, 1.92, 1.68])
        self.stiffnesses = np.array([108.0, 90.0, 60.0])
        self.heights = np.array([11.0, 11.0, 11.0])
        self.spectrum = SpectrumParameters(
            model="exercise",
            coefficient_g_seconds=0.5,
        )

    def test_reference_case_matches_pdf_periods_and_participation(self) -> None:
        results = analyze_building(
            self.masses,
            self.stiffnesses,
            self.heights,
            self.spectrum,
        )

        np.testing.assert_allclose(results["periods"], [1.961, 0.808, 0.541], atol=0.001)
        np.testing.assert_allclose(
            results["mass_percentages"],
            [86.824, 10.667, 2.509],
            atol=0.002,
        )
        np.testing.assert_allclose(
            results["spectral_accelerations"],
            [98.455, 238.971, 356.636],
            atol=0.03,
        )
        np.testing.assert_allclose(
            results["modal_forces"][0],
            [101.123, 175.599, 215.656],
            atol=0.002,
        )
        np.testing.assert_allclose(
            results["modal_displacements"][0],
            [4.559, 8.906, 12.501],
            atol=0.002,
        )
        np.testing.assert_allclose(
            results["srss_displacements"],
            [4.781, 8.978, 12.587],
            atol=0.002,
        )
        np.testing.assert_allclose(
            results["srss_accelerations_g"],
            [0.296, 0.333, 0.409],
            atol=0.002,
        )

    def test_builds_seven_charts_and_downloadable_pdf(self) -> None:
        results = analyze_building(
            self.masses,
            self.stiffnesses,
            self.heights,
            self.spectrum,
        )
        units = {
            "height": "in",
            "mass": "kip·s²/in",
            "stiffness": "kip/in",
            "displacement": "in",
            "force": "kip",
            "acceleration": "in/s²",
        }
        figures = make_figures(
            results,
            self.spectrum,
            9.80665 / 0.0254,
            units,
            6.0,
        )
        try:
            pdf = generate_pdf_report(
                results,
                figures,
                units,
                "Sa = 0.5·g/T",
            )
        finally:
            import matplotlib.pyplot as plt

            for figure in figures.values():
                plt.close(figure)

        self.assertEqual(len(figures), 7)
        structure_labels = [text.get_text() for text in figures["structure"].axes[0].texts]
        self.assertIn("K1 = 108 kip/in", structure_labels)
        self.assertIn("W1 = 834 kip", structure_labels)
        self.assertIn("h1 = 11 in", structure_labels)
        self.assertTrue(pdf.startswith(b"%PDF-"))

    def test_stiffness_matrix_uses_bottom_to_top_story_springs(self) -> None:
        actual = shear_building_stiffness([108.0, 90.0, 60.0])
        expected = np.array(
            [
                [198.0, -90.0, 0.0],
                [-90.0, 150.0, -60.0],
                [0.0, -60.0, 60.0],
            ]
        )
        np.testing.assert_allclose(actual, expected)

    def test_metric_and_imperial_inputs_preserve_periods_and_displacements(self) -> None:
        results_imperial = analyze_building(
            self.masses,
            self.stiffnesses,
            self.heights,
            self.spectrum,
            "imperial",
        )
        imperial_data = pd.DataFrame(
            {
                "floor": [1, 2, 3],
                "height": self.heights,
                "mass": self.masses,
                "stiffness": self.stiffnesses,
            }
        )
        metric_data = convert_floor_data(imperial_data, "imperial", "metric")
        results_metric = analyze_building(
            metric_data["mass"],
            metric_data["stiffness"],
            metric_data["height"],
            self.spectrum,
            "metric",
        )

        np.testing.assert_allclose(
            results_metric["periods"],
            results_imperial["periods"],
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            results_metric["srss_displacements"],
            results_imperial["srss_displacements"] * 0.0254,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            results_metric["srss_story_shears"],
            results_imperial["srss_story_shears"] * 4.4482216152605,
            rtol=1e-9,
        )

    def test_four_corner_spectrum_has_expected_branches(self) -> None:
        spectrum = SpectrumParameters(
            model="four_corner",
            sds_g=1.0,
            sd1_g_seconds=0.5,
            long_period_seconds=6.0,
        )
        values = evaluate_spectrum(
            [0.0 + 0.05, 0.1, 0.5, 1.0, 6.0, 12.0],
            spectrum,
            10.0,
        )
        np.testing.assert_allclose(
            values,
            [7.0, 10.0, 10.0, 5.0, 0.8333333, 0.2083333],
            atol=1e-7,
        )

    def test_four_corner_spectrum_rejects_inconsistent_long_period(self) -> None:
        spectrum = SpectrumParameters(
            model="four_corner",
            sds_g=1.0,
            sd1_g_seconds=0.5,
            long_period_seconds=0.25,
        )
        with self.assertRaises(AnalysisInputError):
            evaluate_spectrum([1.0], spectrum, 9.80665)

    def test_e030_spectrum_uses_zone_factors_and_three_period_branches(self) -> None:
        spectrum = SpectrumParameters(
            model="e030_peru",
            zone_factor=0.35,
            importance_factor=1.0,
            soil_factor=1.15,
            response_reduction_factor=8.0,
            tp_seconds=0.6,
            tl_seconds=2.0,
        )
        values_g = evaluate_spectrum(
            [0.3, 0.6, 1.0, 2.0, 4.0],
            spectrum,
            10.0,
        ) / 10.0
        plateau = 0.35 * 1.0 * 1.15 * 2.5 / 8.0
        np.testing.assert_allclose(
            values_g,
            [plateau, plateau, plateau * 0.6, plateau * 0.6 / 2.0,
             plateau * 0.6 * 2.0 / 4.0**2],
        )

    def test_e030_spectrum_rejects_long_period_before_short_period(self) -> None:
        spectrum = SpectrumParameters(
            model="e030_peru",
            tp_seconds=2.0,
            tl_seconds=0.6,
        )
        with self.assertRaises(AnalysisInputError):
            evaluate_spectrum([1.0], spectrum, 9.80665)

    def test_inconsistent_floor_data_is_rejected(self) -> None:
        with self.assertRaises(AnalysisInputError):
            analyze_building(
                [1.0, 2.0],
                [10.0],
                [3.0, 3.0],
                self.spectrum,
            )


if __name__ == "__main__":
    unittest.main()
