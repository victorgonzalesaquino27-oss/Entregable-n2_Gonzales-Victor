"""Figures shared by the Streamlit interface and PDF report."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from .analysis import SpectrumParameters, evaluate_spectrum

COLORS = ("#00A6A6", "#F4A261", "#6675D9", "#DC6483", "#62A371")
INK = "#172238"
GRID = "#DEE5EF"


def _style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def spectrum_figure(
    results: dict,
    parameters: SpectrumParameters,
    gravity: float,
    maximum_period: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    periods = np.linspace(0.02, maximum_period, 500)
    acceleration_g = evaluate_spectrum(periods, parameters, gravity) / gravity
    ax.plot(
        periods,
        acceleration_g,
        color=COLORS[0],
        linewidth=2.6,
        label="Espectro Sa",
    )
    ax.scatter(
        results["periods"],
        results["spectral_accelerations"] / gravity,
        color=COLORS[1],
        edgecolor="white",
        linewidth=0.8,
        s=60,
        zorder=3,
        label="Periodos modales",
    )
    for index, (period, value) in enumerate(
        zip(results["periods"], results["spectral_accelerations"])
    ):
        ax.annotate(
            f"T{index + 1} = {period:.3f} s",
            (period, value / gravity),
            xytext=(6, 8 if index % 2 == 0 else -16),
            textcoords="offset points",
            fontsize=8,
        )
    if parameters.model == "e030_peru":
        plateau_sa_g = (
            2.5
            * parameters.zone_factor
            * parameters.importance_factor
            * parameters.soil_factor
            / parameters.response_reduction_factor
        )
        ax.vlines(
            parameters.tp_seconds,
            0,
            plateau_sa_g,
            color="#DC6483",
            linestyle="--",
            linewidth=1.5,
            label=f"Tp = {parameters.tp_seconds:g} s",
        )
        long_period_sa_g = plateau_sa_g * parameters.tp_seconds / parameters.tl_seconds
        ax.vlines(
            parameters.tl_seconds,
            0,
            long_period_sa_g,
            color="#62A371",
            linestyle="--",
            linewidth=1.5,
            label=f"TL = {parameters.tl_seconds:g} s",
        )
    ax.set(
        title="Espectro de pseudoaceleración",
        xlabel="Periodo, T (s)",
        ylabel="Sa (g)",
        xlim=(0, maximum_period),
    )
    ax.legend(frameon=False)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def mode_shapes_figure(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    floors = np.arange(1, len(results["masses"]) + 1)
    for index in range(results["modes"].shape[1]):
        shape = results["modes"][:, index]
        ax.plot(
            np.r_[0.0, shape],
            np.r_[0, floors],
            marker="o",
            linewidth=2.2,
            markersize=5,
            color=COLORS[index % len(COLORS)],
            label=f"Modo {index + 1} · T={results['periods'][index]:.3f} s",
        )
    ax.set(
        title="Formas modales normalizadas",
        xlabel="Desplazamiento modal relativo",
        ylabel="Piso (de abajo hacia arriba)",
        yticks=np.arange(0, len(floors) + 1),
    )
    ax.legend(frameon=False, loc="best")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def participation_figure(results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    modes = np.arange(1, len(results["mass_percentages"]) + 1)
    bars = ax.bar(
        modes,
        results["mass_percentages"],
        color=[COLORS[index % len(COLORS)] for index in range(len(modes))],
        width=0.62,
        label="Masa modal efectiva",
    )
    for bar, value in zip(bars, results["mass_percentages"]):
        ax.annotate(
            f"{value:.2f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    cumulative_ax = ax.twinx()
    cumulative_ax.plot(
        modes,
        results["cumulative_mass_percentages"],
        marker="o",
        color=INK,
        linestyle="--",
        label="Acumulado",
    )
    cumulative_ax.set_ylabel("Masa acumulada (%)")
    cumulative_ax.set_ylim(0, 110)
    cumulative_ax.spines["top"].set_visible(False)
    cumulative_ax.spines["right"].set_color(GRID)
    cumulative_ax.tick_params(colors=INK)
    ax.set(
        title="Participación de masa por modo",
        xlabel="Modo",
        ylabel="Masa modal efectiva (%)",
        xticks=modes,
        ylim=(0, max(110, float(np.max(results["mass_percentages"])) * 1.2)),
    )
    ax.legend([bars], ["Masa modal efectiva"], frameon=False, loc="upper left")
    cumulative_ax.legend(frameon=False, loc="upper right")
    _style_axes(ax)
    fig.tight_layout()
    return fig


def displacement_drift_figure(
    results: dict,
    displacement_unit: str,
) -> plt.Figure:
    floors = np.arange(1, len(results["masses"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    axes[0].bar(
        floors,
        results["srss_displacements"],
        color=COLORS[0],
        width=0.62,
    )
    axes[0].set(
        title="Desplazamiento combinado · SRSS",
        xlabel="Piso",
        ylabel=f"Desplazamiento ({displacement_unit})",
        xticks=floors,
    )
    axes[1].bar(
        floors,
        results["srss_drift_ratios"] * 100,
        color=COLORS[1],
        width=0.62,
    )
    axes[1].set(
        title="Deriva relativa combinada · SRSS",
        xlabel="Piso",
        ylabel="Deriva / altura de entrepiso (%)",
        xticks=floors,
    )
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    return fig


def force_shear_figure(results: dict, force_unit: str) -> plt.Figure:
    floors = np.arange(1, len(results["masses"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    for index in range(results["modal_forces"].shape[0]):
        axes[0].plot(
            results["modal_forces"][index],
            floors,
            marker="o",
            linewidth=1.8,
            color=COLORS[index % len(COLORS)],
            label=f"Modo {index + 1}",
        )
    axes[0].set(
        title="Fuerzas modales por nivel",
        xlabel=f"Fuerza ({force_unit})",
        ylabel="Piso",
        yticks=floors,
    )
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].barh(floors, results["srss_story_shears"], color=COLORS[2], height=0.62)
    axes[1].set(
        title="Cortante por nivel · SRSS",
        xlabel=f"Cortante ({force_unit})",
        ylabel="Piso",
        yticks=floors,
    )
    for ax in axes:
        _style_axes(ax)
    fig.tight_layout()
    return fig


def acceleration_figure(results: dict) -> plt.Figure:
    floors = np.arange(1, len(results["masses"]) + 1)
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.bar(
        floors,
        results["srss_accelerations_g"],
        color=COLORS[3],
        width=0.62,
    )
    ax.set(
        title="Aceleración por nivel · combinación SRSS",
        xlabel="Piso",
        ylabel="Aceleración (g)",
        xticks=floors,
    )
    _style_axes(ax)
    fig.tight_layout()
    return fig


def structural_model_figure(
    results: dict,
    units: dict[str, str],
    gravity: float,
) -> plt.Figure:
    """Draw a floor-by-floor frame with story stiffness, height, and weight."""
    floor_count = len(results["masses"])
    heights = results["story_heights"]
    elevations = np.concatenate(([0.0], np.cumsum(heights)))
    force_divisor = 1000.0 if results["unit_system"] == "metric" else 1.0
    weights = results["masses"] * gravity / force_divisor
    figure_height = max(5.2, min(13.0, 3.0 + floor_count * 0.32))
    fig, ax = plt.subplots(figsize=(9.5, figure_height))

    frame_left, frame_right = 4.0, 5.8
    floor_center = (frame_left + frame_right) / 2
    spring_center = 1.35
    ax.plot(
        [frame_left, frame_left],
        [0, elevations[-1]],
        color=INK,
        linewidth=2.2,
    )
    ax.plot(
        [frame_right, frame_right],
        [0, elevations[-1]],
        color=INK,
        linewidth=2.2,
    )
    ax.plot([frame_left - 0.25, frame_right + 0.25], [0, 0], color=INK, linewidth=2.2)

    for index in range(floor_count):
        lower = elevations[index]
        upper = elevations[index + 1]
        height = heights[index]
        middle = (lower + upper) / 2
        spring_bottom = lower + height * 0.12
        spring_top = upper - height * 0.12
        zigzag_y = np.linspace(spring_bottom, spring_top, 11)
        zigzag_x = np.full_like(zigzag_y, spring_center)
        zigzag_x[1:-1:2] += 0.16
        zigzag_x[2:-1:2] -= 0.16

        ax.plot(
            [spring_center, spring_center],
            [lower, spring_bottom],
            color=COLORS[2],
            linewidth=1.4,
        )
        ax.plot(zigzag_x, zigzag_y, color=COLORS[2], linewidth=1.6)
        ax.plot(
            [spring_center, spring_center],
            [spring_top, upper],
            color=COLORS[2],
            linewidth=1.4,
        )
        ax.text(
            0.05,
            middle,
            f"K{index + 1} = {results['story_stiffnesses'][index]:,.4g} "
            f"{units['stiffness']}",
            va="center",
            fontsize=9,
            color=INK,
        )

        ax.plot(
            [frame_left, frame_right],
            [upper, upper],
            color=COLORS[0],
            linewidth=2.0,
        )
        ax.scatter(
            [floor_center],
            [upper],
            s=54,
            facecolor="white",
            edgecolor=COLORS[1],
            linewidth=1.8,
            zorder=4,
        )
        arrow_start = upper - height * 0.08
        arrow_end = upper - height * 0.42
        ax.annotate(
            "",
            xy=(floor_center, arrow_end),
            xytext=(floor_center, arrow_start),
            arrowprops={"arrowstyle": "-|>", "color": COLORS[1], "lw": 1.4},
        )
        ax.text(
            6.15,
            upper - height * 0.24,
            f"W{index + 1} = {weights[index]:,.4g} {units['force']}",
            va="center",
            fontsize=9,
            color=INK,
        )

        dimension_x = 8.5
        ax.annotate(
            "",
            xy=(dimension_x, upper),
            xytext=(dimension_x, lower),
            arrowprops={"arrowstyle": "<->", "color": "#75859A", "lw": 1.0},
        )
        ax.text(
            dimension_x + 0.2,
            middle,
            f"h{index + 1} = {height:g} {units['height']}",
            va="center",
            fontsize=9,
            color=INK,
        )

    ax.set_title("Esquema referencial por nivel", loc="left", color=INK, pad=14)
    ax.text(
        0.05,
        elevations[-1] + max(elevations[-1] * 0.025, 0.15),
        "Resorte: rigidez de entrepiso  ·  Flecha: peso sísmico por nivel",
        fontsize=9,
        color="#526174",
        va="bottom",
    )
    ax.set_xlim(0, 11.5)
    ax.set_ylim(-max(elevations[-1] * 0.04, 0.2), elevations[-1] * 1.1)
    ax.set_ylabel(f"Elevación ({units['height']})")
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK)
    fig.tight_layout()
    return fig


def make_figures(
    results: dict,
    parameters: SpectrumParameters,
    gravity: float,
    units: dict[str, str],
    maximum_period: float,
) -> dict[str, plt.Figure]:
    """Create the result charts shown in the app and exported to PDF."""
    return {
        "spectrum": spectrum_figure(
            results,
            parameters,
            gravity,
            maximum_period,
        ),
        "modes": mode_shapes_figure(results),
        "participation": participation_figure(results),
        "displacements": displacement_drift_figure(
            results,
            units["displacement"],
        ),
        "forces": force_shear_figure(results, units["force"]),
        "accelerations": acceleration_figure(results),
        "structure": structural_model_figure(results, units, gravity),
    }
