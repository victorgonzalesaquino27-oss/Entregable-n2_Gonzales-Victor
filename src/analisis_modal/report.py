"""PDF report generation."""

from __future__ import annotations

from io import BytesIO

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def _text_page(title: str, paragraphs: list[str]) -> plt.Figure:
    figure = plt.figure(figsize=(8.27, 11.69))
    axis = figure.add_axes([0.08, 0.08, 0.84, 0.84])
    axis.axis("off")
    axis.text(0.0, 1.0, title, fontsize=20, fontweight="bold", va="top")
    y = 0.91
    for paragraph in paragraphs:
        axis.text(0.0, y, paragraph, fontsize=11, va="top", wrap=True)
        y -= 0.055
    return figure


def _table_page(title: str, headers: list[str], rows: list[list[str]]) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(11.69, 8.27))
    axis.axis("off")
    axis.set_title(title, fontsize=17, fontweight="bold", loc="left", pad=18)
    table = axis.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)
    for column in range(len(headers)):
        table[(0, column)].set_facecolor("#172238")
        table[(0, column)].set_text_props(color="white", weight="bold")
    figure.tight_layout()
    return figure


def _add_table_pages(
    pdf: PdfPages,
    title: str,
    headers: list[str],
    rows: list[list[str]],
) -> None:
    page_size = 18
    for start in range(0, len(rows), page_size):
        page_number = start // page_size + 1
        page_title = title if len(rows) <= page_size else f"{title} · página {page_number}"
        page = _table_page(page_title, headers, rows[start : start + page_size])
        pdf.savefig(page, bbox_inches="tight")
        plt.close(page)


def generate_pdf_report(
    results: dict,
    figures: dict[str, plt.Figure],
    units: dict[str, str],
    spectrum_description: str,
) -> bytes:
    """Export inputs, key results, and all charts as a multi-page PDF."""
    buffer = BytesIO()
    with PdfPages(
        buffer,
        metadata={
            "Title": "Reporte de análisis modal y respuesta espectral",
            "Author": "Entregable 2 - Aplicación Streamlit",
            "Subject": "Análisis de estructura de corte",
        },
    ) as pdf:
        cover = _text_page(
            "REPORTE DE ANÁLISIS MODAL",
            [
                "Análisis lineal de una estructura idealizada de corte.",
                (
                    f"Niveles: {len(results['masses'])}  |  "
                    f"Sistema: {'Métrico' if results['unit_system'] == 'metric' else 'Imperial'}"
                ),
                (
                    f"Masa total: {results['total_mass']:.4g} {units['mass']}  |  "
                    f"Peso total: {results['total_weight']:.4g} {units['force']}"
                ),
                f"Espectro: {spectrum_description}",
                (
                    f"Periodo fundamental: {results['periods'][0]:.4f} s  |  "
                    f"Cortante SRSS en la base: "
                    f"{results['srss_story_shears'][0]:.4g} {units['force']}"
                ),
                "",
                "PROPIEDADES MODALES",
                (
                    "Modo | lambda (1/s²) | omega (rad/s) | T (s) | "
                    "Gamma | Masa modal (%) | Sa"
                ),
                "",
                (
                    "Las propiedades de todos los modos y los resultados por nivel "
                    "se detallan en las tablas y gráficas siguientes."
                ),
            ],
        )
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        mode_rows = [
            [
                str(index + 1),
                f"{results['eigenvalues'][index]:.5g}",
                f"{results['omega'][index]:.5g}",
                f"{results['periods'][index]:.5g}",
                f"{results['participation_factors'][index]:.5g}",
                f"{results['effective_modal_masses'][index]:.5g}",
                f"{results['mass_percentages'][index]:.3f}%",
                f"{results['cumulative_mass_percentages'][index]:.3f}%",
                f"{results['spectral_accelerations'][index]:.5g}",
            ]
            for index in range(len(results["periods"]))
        ]
        _add_table_pages(
            pdf,
            "Resumen de propiedades modales",
            [
                "Modo",
                "lambda (1/s²)",
                "omega (rad/s)",
                "T (s)",
                "Gamma",
                f"Masa modal ({units['mass']})",
                "Participación",
                "Acumulada",
                f"Sa ({units['acceleration']})",
            ],
            mode_rows,
        )

        floor_rows = [
            [
                str(index + 1),
                f"{results['story_heights'][index]:.5g}",
                f"{results['masses'][index]:.5g}",
                f"{results['story_stiffnesses'][index]:.5g}",
                f"{results['srss_displacements'][index]:.5g}",
                f"{results['srss_drifts'][index]:.5g}",
                f"{results['srss_drift_ratios'][index] * 100:.4g}%",
                f"{results['srss_forces'][index]:.5g}",
                f"{results['srss_story_shears'][index]:.5g}",
                f"{results['srss_accelerations_g'][index]:.5g}",
            ]
            for index in range(len(results["masses"]))
        ]
        _add_table_pages(
            pdf,
            "Resultados combinados por nivel · SRSS",
            [
                "Piso",
                f"Altura ({units['height']})",
                f"Masa ({units['mass']})",
                f"Rigidez ({units['stiffness']})",
                f"u ({units['displacement']})",
                f"Deriva ({units['displacement']})",
                "Deriva relativa",
                f"Fuerza ({units['force']})",
                f"Cortante ({units['force']})",
                "Aceleración (g)",
            ],
            floor_rows,
        )

        for figure in figures.values():
            pdf.savefig(figure, bbox_inches="tight")

    return buffer.getvalue()
