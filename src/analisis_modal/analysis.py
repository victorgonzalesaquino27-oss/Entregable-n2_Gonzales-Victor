"""Numerical routines for shear-building modal response analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

UnitSystem = Literal["metric", "imperial"]
SpectrumModel = Literal["exercise", "four_corner", "e030_peru"]

GRAVITY = {"metric": 9.80665, "imperial": 9.80665 / 0.0254}
STIFFNESS_TO_BASE_UNITS = {"metric": 1000.0, "imperial": 1.0}
FORCE_FROM_BASE_UNITS = {"metric": 1000.0, "imperial": 1.0}


class AnalysisInputError(ValueError):
    """Raised when model or spectrum inputs are invalid."""


@dataclass(frozen=True)
class SpectrumParameters:
    """Editable parameters for supported pseudo-acceleration spectra."""

    model: SpectrumModel = "e030_peru"
    coefficient_g_seconds: float = 0.5
    sds_g: float = 1.0
    sd1_g_seconds: float = 0.5
    long_period_seconds: float = 6.0
    zone_factor: float = 0.35
    importance_factor: float = 1.0
    soil_factor: float = 1.15
    response_reduction_factor: float = 8.0
    tp_seconds: float = 0.6
    tl_seconds: float = 2.0


def _positive_vector(values: ArrayLike, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise AnalysisInputError(f"{name} debe contener al menos un valor.")
    if not np.all(np.isfinite(vector)) or np.any(vector <= 0):
        raise AnalysisInputError(f"Todos los valores de {name} deben ser finitos y positivos.")
    return vector


def shear_building_stiffness(story_stiffnesses: ArrayLike) -> NDArray[np.float64]:
    """Build the floor-ordered stiffness matrix from bottom-to-top story springs."""
    stiffnesses = _positive_vector(story_stiffnesses, "las rigideces")
    floors = stiffnesses.size
    matrix = np.zeros((floors, floors), dtype=float)

    for floor, stiffness in enumerate(stiffnesses):
        matrix[floor, floor] += stiffness
        if floor > 0:
            matrix[floor - 1, floor - 1] += stiffness
            matrix[floor, floor - 1] = -stiffness
            matrix[floor - 1, floor] = -stiffness

    return matrix


def evaluate_spectrum(
    periods_seconds: ArrayLike,
    parameters: SpectrumParameters,
    gravity: float,
) -> NDArray[np.float64]:
    """Evaluate pseudo-spectral acceleration in the selected acceleration units."""
    periods = np.asarray(periods_seconds, dtype=float)
    if not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise AnalysisInputError("Los periodos para evaluar el espectro deben ser positivos.")
    if not np.isfinite(gravity) or gravity <= 0:
        raise AnalysisInputError("La aceleración de gravedad debe ser positiva.")

    if parameters.model == "exercise":
        coefficient = parameters.coefficient_g_seconds
        if not np.isfinite(coefficient) or coefficient <= 0:
            raise AnalysisInputError("El coeficiente C del espectro debe ser positivo.")
        return coefficient * gravity / periods

    if parameters.model == "e030_peru":
        factors = (
            parameters.zone_factor,
            parameters.importance_factor,
            parameters.soil_factor,
            parameters.response_reduction_factor,
            parameters.tp_seconds,
            parameters.tl_seconds,
        )
        if not all(np.isfinite(value) and value > 0 for value in factors):
            raise AnalysisInputError(
                "Z, U, S, R, Tp y TL deben ser finitos y positivos."
            )
        if parameters.tl_seconds < parameters.tp_seconds:
            raise AnalysisInputError("TL debe ser igual o mayor que Tp.")

        periods_at_or_below_tp = periods <= parameters.tp_seconds
        periods_between_corners = (
            (periods > parameters.tp_seconds) & (periods <= parameters.tl_seconds)
        )
        periods_above_tl = periods > parameters.tl_seconds
        coefficient = np.empty_like(periods)
        coefficient[periods_at_or_below_tp] = 2.5
        coefficient[periods_between_corners] = (
            2.5 * parameters.tp_seconds / periods[periods_between_corners]
        )
        coefficient[periods_above_tl] = (
            2.5
            * parameters.tp_seconds
            * parameters.tl_seconds
            / periods[periods_above_tl] ** 2
        )
        scale = (
            parameters.zone_factor
            * parameters.importance_factor
            * parameters.soil_factor
            / parameters.response_reduction_factor
        )
        return scale * coefficient * gravity

    if parameters.model != "four_corner":
        raise AnalysisInputError("El modelo de espectro seleccionado no es válido.")

    sds = parameters.sds_g
    sd1 = parameters.sd1_g_seconds
    long_period = parameters.long_period_seconds
    if not all(np.isfinite(value) and value > 0 for value in (sds, sd1, long_period)):
        raise AnalysisInputError("SDS, SD1 y TL deben ser finitos y positivos.")

    short_corner = 0.2 * sd1 / sds
    plateau_end = sd1 / sds
    if long_period < plateau_end:
        raise AnalysisInputError(
            "TL debe ser igual o mayor que Ts = SD1/SDS para cerrar el espectro."
        )
    ordinates_g = np.empty_like(periods)

    initial = periods < short_corner
    plateau = (periods >= short_corner) & (periods <= plateau_end)
    descending = (periods > plateau_end) & (periods <= long_period)
    long_period_branch = periods > long_period

    ordinates_g[initial] = sds * (
        0.4 + 0.6 * periods[initial] / short_corner
    )
    ordinates_g[plateau] = sds
    ordinates_g[descending] = sd1 / periods[descending]
    ordinates_g[long_period_branch] = (
        sd1 * long_period / periods[long_period_branch] ** 2
    )
    return ordinates_g * gravity


def analyze_building(
    masses: ArrayLike,
    story_stiffnesses: ArrayLike,
    story_heights: ArrayLike,
    spectrum: SpectrumParameters,
    unit_system: UnitSystem = "imperial",
) -> dict[str, NDArray[np.float64] | float | str]:
    """Compute modal and SRSS response for a lumped-mass shear-building model.

    Floor arrays are ordered from the base upward. Metric input uses kg,
    kN/m, and m; imperial input uses kip*s²/in, kip/in, and in.
    """
    if unit_system not in GRAVITY:
        raise AnalysisInputError("El sistema de unidades debe ser metric o imperial.")

    mass_values = _positive_vector(masses, "las masas")
    stiffness_values = _positive_vector(story_stiffnesses, "las rigideces")
    height_values = _positive_vector(story_heights, "las alturas")
    floor_count = mass_values.size
    if stiffness_values.size != floor_count or height_values.size != floor_count:
        raise AnalysisInputError(
            "Debe ingresar una masa, rigidez y altura para cada piso."
        )

    mass_matrix = np.diag(mass_values)
    stiffness_matrix = shear_building_stiffness(stiffness_values)
    stiffness_consistent = stiffness_matrix * STIFFNESS_TO_BASE_UNITS[unit_system]

    root_mass = np.sqrt(mass_values)
    symmetric_operator = stiffness_consistent / np.outer(root_mass, root_mass)
    eigenvalues, mass_scaled_modes = np.linalg.eigh(symmetric_operator)
    if np.any(eigenvalues <= 0):
        raise AnalysisInputError(
            "El modelo no tiene frecuencias naturales positivas; revise masas y rigideces."
        )

    omega = np.sqrt(eigenvalues)
    periods = 2.0 * np.pi / omega
    modes = mass_scaled_modes / root_mass[:, np.newaxis]
    modes /= np.max(np.abs(modes), axis=0, keepdims=True)

    influence = np.ones(floor_count, dtype=float)
    participation_factors = np.empty(floor_count, dtype=float)
    effective_modal_masses = np.empty(floor_count, dtype=float)
    for mode_number in range(floor_count):
        shape = modes[:, mode_number]
        numerator = float(shape @ mass_matrix @ influence)
        denominator = float(shape @ mass_matrix @ shape)
        if numerator < 0:
            modes[:, mode_number] *= -1.0
            numerator *= -1.0
        elif np.isclose(numerator, 0.0):
            largest_component = int(np.argmax(np.abs(shape)))
            if shape[largest_component] < 0:
                modes[:, mode_number] *= -1.0
        participation_factors[mode_number] = -numerator / denominator
        effective_modal_masses[mode_number] = numerator**2 / denominator

    mass_percentages = effective_modal_masses / np.sum(mass_values) * 100.0
    cumulative_mass_percentages = np.cumsum(mass_percentages)
    gravity = GRAVITY[unit_system]
    spectral_accelerations = evaluate_spectrum(periods, spectrum, gravity)

    modal_displacements = np.empty((floor_count, floor_count), dtype=float)
    modal_drifts = np.empty_like(modal_displacements)
    modal_forces = np.empty_like(modal_displacements)
    for mode_number in range(floor_count):
        shape = modes[:, mode_number]
        modal_displacements[mode_number] = (
            -participation_factors[mode_number]
            * shape
            * spectral_accelerations[mode_number]
            / eigenvalues[mode_number]
        )
        modal_drifts[mode_number] = np.diff(
            np.concatenate(([0.0], modal_displacements[mode_number]))
        )
        modal_forces[mode_number] = (
            -(mass_matrix @ shape)
            * participation_factors[mode_number]
            * spectral_accelerations[mode_number]
            / FORCE_FROM_BASE_UNITS[unit_system]
        )

    modal_drift_ratios = modal_drifts / height_values[np.newaxis, :]
    modal_accelerations = (
        modal_forces
        * FORCE_FROM_BASE_UNITS[unit_system]
        / mass_values[np.newaxis, :]
    )
    modal_accelerations_g = modal_accelerations / gravity
    modal_story_shears = np.cumsum(modal_forces[:, ::-1], axis=1)[:, ::-1]

    def srss(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(np.sum(values**2, axis=0))

    return {
        "unit_system": unit_system,
        "mass_matrix": mass_matrix,
        "stiffness_matrix": stiffness_matrix,
        "eigenvalues": eigenvalues,
        "omega": omega,
        "periods": periods,
        "modes": modes,
        "participation_factors": participation_factors,
        "effective_modal_masses": effective_modal_masses,
        "mass_percentages": mass_percentages,
        "cumulative_mass_percentages": cumulative_mass_percentages,
        "spectral_accelerations": spectral_accelerations,
        "modal_displacements": modal_displacements,
        "modal_drifts": modal_drifts,
        "modal_drift_ratios": modal_drift_ratios,
        "modal_forces": modal_forces,
        "modal_story_shears": modal_story_shears,
        "modal_accelerations": modal_accelerations,
        "modal_accelerations_g": modal_accelerations_g,
        "srss_displacements": srss(modal_displacements),
        "srss_drifts": srss(modal_drifts),
        "srss_drift_ratios": srss(modal_drift_ratios),
        "srss_forces": srss(modal_forces),
        "srss_story_shears": srss(modal_story_shears),
        "srss_accelerations": srss(modal_accelerations),
        "srss_accelerations_g": srss(modal_accelerations_g),
        "total_mass": float(np.sum(mass_values)),
        "total_weight": float(
            np.sum(mass_values) * gravity / FORCE_FROM_BASE_UNITS[unit_system]
        ),
        "story_heights": height_values,
        "story_stiffnesses": stiffness_values,
        "masses": mass_values,
    }
