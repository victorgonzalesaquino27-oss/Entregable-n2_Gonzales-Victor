from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from analisis_modal import (  # noqa: E402
    AnalysisInputError,
    SpectrumParameters,
    analyze_building,
    parse_floor_data,
)
from analisis_modal.analysis import GRAVITY  # noqa: E402
from analisis_modal.data_io import convert_floor_data, template_text  # noqa: E402
from analisis_modal.plots import make_figures  # noqa: E402
from analisis_modal.report import generate_pdf_report  # noqa: E402

st.set_page_config(
    page_title="Análisis modal · Entregable 2",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stHeader"] { background: rgba(245,247,251,.92); }
    [data-testid="stMain"] {
      background: #f5f7fb;
      color: #172238;
    }
    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4,
    [data-testid="stMain"] h5,
    [data-testid="stMain"] h6,
    [data-testid="stMain"] p,
    [data-testid="stMain"] label {
      color: #172238;
    }
    [data-testid="stMain"] [data-testid="stCaptionContainer"] p {
      color: #526174;
    }
    [data-testid="stMain"] input,
    [data-testid="stMain"] textarea {
      background-color: #ffffff;
      color: #172238;
    }
    [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #edf4fb 0%, #dce8f4 100%);
      color: #172238;
    }
    [data-testid="stSidebar"] * { color: #172238; }
    [data-testid="stSidebar"] [data-baseweb="select"] * {
      color: #172238 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
      background-color: #ffffff;
      border-color: #aeb9c8;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
      background-color: #ffffff;
      border: 1px dashed #aeb9c8;
    }
    [data-testid="stSidebar"] button {
      background-color: #f4f7fb;
      color: #172238 !important;
      border-color: #aeb9c8;
    }
    [data-testid="stSidebar"] button:hover {
      background-color: #e7edf5;
      color: #172238 !important;
      border-color: #8392a6;
    }
    .sidebar-logo {
      color: #101b2d;
      font-size: 2rem;
      font-weight: 800;
      letter-spacing: 0.22em;
      line-height: 1;
      margin: 0.35rem 0 1.6rem;
      padding: 0.8rem 0.9rem;
      text-align: center;
      border: 1px solid rgba(23, 34, 56, 0.15);
      border-radius: 12px;
      background: linear-gradient(135deg, rgba(255,255,255,.72), rgba(220,232,244,.55));
      box-shadow: 0 4px 12px rgba(23, 34, 56, 0.08);
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
      background-color: #ffffff !important;
      color: #172238 !important;
    }
    [data-testid="stSidebar"] [role="combobox"] {
      color: #172238 !important;
    }
    .hero {
      background: linear-gradient(115deg, #172238, #263d5c);
      padding: 1.65rem 1.9rem; border-radius: 16px; color: white;
      margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #d7e3f2; margin: .35rem 0 0; }
    [data-testid="stMetric"] {
      background: white; border: 1px solid #e1e8f0; padding: 1rem;
      border-radius: 12px;
    }
    div[data-testid="stDownloadButton"] button {
      background: #00a6a6; color: white; border: 0; border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SYSTEM_NAMES = {"Métrico (SI)": "metric", "Imperial (kip-in)": "imperial"}
UNIT_LABELS = {
    "metric": {
        "height": "m",
        "mass": "kg",
        "stiffness": "kN/m",
        "displacement": "m",
        "force": "kN",
        "acceleration": "m/s²",
    },
    "imperial": {
        "height": "in",
        "mass": "kip·s²/in",
        "stiffness": "kip/in",
        "displacement": "in",
        "force": "kip",
        "acceleration": "in/s²",
    },
}
IMPERIAL_REFERENCE = pd.DataFrame(
    {
        "floor": [1, 2, 3],
        "height": [11.0, 11.0, 11.0],
        "mass": [2.16, 1.92, 1.68],
        "stiffness": [108.0, 90.0, 60.0],
    }
)
DEFAULTS = {
    "metric": pd.DataFrame(
        {
            "floor": [1, 2, 3],
            "height": [0.2794, 0.2794, 0.2794],
            "mass": [
                2.16 * 4448.2216152605 / 0.0254,
                1.92 * 4448.2216152605 / 0.0254,
                1.68 * 4448.2216152605 / 0.0254,
            ],
            "stiffness": [
                108.0 * 4448.2216152605 / 0.0254 / 1000,
                90.0 * 4448.2216152605 / 0.0254 / 1000,
                60.0 * 4448.2216152605 / 0.0254 / 1000,
            ],
        }
    ),
    "imperial": IMPERIAL_REFERENCE,
}


def prepare_floor_data(unit_system: str, floor_count: int) -> pd.DataFrame:
    current = st.session_state["floor_data"].copy()
    current = current.sort_values("floor").reset_index(drop=True)
    current = current.iloc[:floor_count].copy()

    if len(current) < floor_count:
        defaults = DEFAULTS[unit_system]
        while len(current) < floor_count:
            row_index = min(len(current), len(defaults) - 1)
            row = defaults.iloc[row_index].copy()
            row["floor"] = len(current) + 1
            current.loc[len(current)] = row
    current["floor"] = np.arange(1, floor_count + 1)
    return current[["floor", "height", "mass", "stiffness"]].reset_index(drop=True)


def spectrum_description(parameters: SpectrumParameters) -> str:
    if parameters.model == "e030_peru":
        return (
            "E.030 (Perú): Sa = Z·U·S·C(T)·g/R; "
            f"Z={parameters.zone_factor:g}, "
            f"U={parameters.importance_factor:g}, "
            f"S={parameters.soil_factor:g}, "
            f"R={parameters.response_reduction_factor:g}, "
            f"Tp={parameters.tp_seconds:g} s, "
            f"TL={parameters.tl_seconds:g} s"
        )
    if parameters.model == "exercise":
        return f"Sa = {parameters.coefficient_g_seconds:g}·g/T"
    return (
        f"4 tramos: SDS={parameters.sds_g:g} g, "
        f"SD1={parameters.sd1_g_seconds:g} g·s, "
        f"TL={parameters.long_period_seconds:g} s"
    )


with st.sidebar:
    st.markdown('<div class="sidebar-logo">VICOG</div>', unsafe_allow_html=True)
    st.markdown("## Configuración")
    system_name = st.selectbox(
        "Sistema de unidades",
        list(SYSTEM_NAMES),
        index=1,
        key="unit_system_name",
    )
    unit_system = SYSTEM_NAMES[system_name]

    if "floor_data" not in st.session_state:
        st.session_state["floor_data"] = DEFAULTS[unit_system].copy()
        st.session_state["previous_unit_system"] = unit_system
    elif st.session_state["previous_unit_system"] != unit_system:
        st.session_state["floor_data"] = convert_floor_data(
            st.session_state["floor_data"],
            st.session_state["previous_unit_system"],
            unit_system,
        )
        st.session_state["previous_unit_system"] = unit_system

    st.markdown("### Carga desde archivo")
    uploaded_file = st.file_uploader(
        "Archivo TXT con pisos, alturas, masas y rigideces",
        type=["txt"],
        help="Usa los encabezados Piso, Altura, Masa y Rigidez. Los valores deben estar en el sistema elegido.",
        key="floor_data_upload",
    )
    if st.button("Importar datos TXT", width="stretch"):
        if uploaded_file is None:
            st.warning("Primero selecciona un archivo TXT.")
        else:
            try:
                parsed_data = parse_floor_data(
                    uploaded_file.getvalue().decode("utf-8-sig")
                )
            except (UnicodeDecodeError, AnalysisInputError) as error:
                st.error(f"No se importaron los datos: {error}")
            else:
                st.session_state["floor_data"] = parsed_data
                st.session_state["floor_count"] = len(parsed_data)
                editor_key = f"floor_editor_{unit_system}_{len(parsed_data)}"
                st.session_state.pop(editor_key, None)
                st.success(f"Se cargaron {len(parsed_data)} niveles.")

    st.download_button(
        "Descargar plantilla TXT",
        data=template_text(),
        file_name="plantilla_datos_estructura.txt",
        mime="text/plain",
        width="stretch",
    )
    st.caption("Los datos del TXT se interpretan en las unidades seleccionadas.")

    st.session_state.setdefault("floor_count", 3)
    floor_count = st.number_input(
        "Número de pisos",
        min_value=1,
        max_value=30,
        step=1,
        key="floor_count",
    )
    st.markdown("### Espectro de pseudoaceleración")
    st.caption("Espectro de diseño E.030 (Perú)")
    zone_factor = st.number_input(
        "Z · Zona",
        min_value=0.001,
        max_value=10.0,
        value=0.35,
        step=0.05,
        help="Factor de zona sísmica, según la ubicación del proyecto.",
    )
    importance_factor = st.number_input(
        "U · Uso",
        min_value=0.001,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Factor de uso o importancia de la edificación.",
    )
    soil_factor = st.number_input(
        "S · Suelo",
        min_value=0.001,
        max_value=10.0,
        value=1.15,
        step=0.05,
        help="Factor de suelo definido para el perfil correspondiente.",
    )
    response_reduction_factor = st.number_input(
        "R · Reducción",
        min_value=0.001,
        max_value=100.0,
        value=8.0,
        step=0.5,
        help="Coeficiente de reducción de la respuesta sísmica.",
    )
    tp_seconds = st.number_input(
        "Tp · Periodo corto (s)",
        min_value=0.001,
        max_value=30.0,
        value=0.6,
        step=0.1,
    )
    tl_seconds = st.number_input(
        "TL · Periodo largo (s)",
        min_value=0.001,
        max_value=60.0,
        value=2.0,
        step=0.1,
    )
    maximum_period = st.number_input(
        "Periodo máximo del gráfico (s)",
        min_value=0.5,
        max_value=30.0,
        value=6.0,
        step=0.5,
    )
    spectrum_parameters = SpectrumParameters(
        model="e030_peru",
        zone_factor=zone_factor,
        importance_factor=importance_factor,
        soil_factor=soil_factor,
        response_reduction_factor=response_reduction_factor,
        tp_seconds=tp_seconds,
        tl_seconds=tl_seconds,
    )

    st.info(
        "C(T) = 2.5 para T ≤ Tp; C(T) = 2.5·Tp/T para Tp < T ≤ TL; "
        "C(T) = 2.5·Tp·TL/T² para T > TL. "
        "Sa = Z·U·S·C(T)·g/R. El gráfico muestra Sa en unidades de g."
    )

unit_labels = UNIT_LABELS[unit_system]
floor_data = prepare_floor_data(unit_system, int(floor_count))
editor_key = f"floor_editor_{unit_system}_{int(floor_count)}"
st.markdown(
    """
    <div class="hero">
      <h1>Análisis modal y respuesta espectral</h1>
      <p>Edificio de corte · parámetros editables · combinación modal SRSS</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown("### Modelo de la estructura")
st.caption(
    "Ingresa una fila por piso, de abajo hacia arriba. "
    "Altura, masa y rigidez son editables; el TXT y la tabla usan las mismas unidades."
)

edited_data = st.data_editor(
    floor_data,
    key=editor_key,
    hide_index=True,
    width="stretch",
    num_rows="fixed",
    column_config={
        "floor": st.column_config.NumberColumn("Piso", disabled=True, format="%d"),
        "height": st.column_config.NumberColumn(
            f"Altura ({unit_labels['height']})",
            min_value=0.000001,
            step=0.1,
            format="%.4f",
        ),
        "mass": st.column_config.NumberColumn(
            f"Masa ({unit_labels['mass']})",
            min_value=0.000001,
            step=0.1,
            format="%.4f",
        ),
        "stiffness": st.column_config.NumberColumn(
            f"Rigidez ({unit_labels['stiffness']})",
            min_value=0.000001,
            step=1.0,
            format="%.4f",
        ),
    },
)
st.session_state["floor_data"] = edited_data.copy()

try:
    results = analyze_building(
        masses=edited_data["mass"].to_numpy(dtype=float),
        story_stiffnesses=edited_data["stiffness"].to_numpy(dtype=float),
        story_heights=edited_data["height"].to_numpy(dtype=float),
        spectrum=spectrum_parameters,
        unit_system=unit_system,
    )
except AnalysisInputError as error:
    st.error(f"No se pudo ejecutar el análisis: {error}")
    st.stop()

gravity = GRAVITY[unit_system]
figures = make_figures(
    results,
    spectrum_parameters,
    gravity,
    unit_labels,
    max(
        float(maximum_period),
        float(np.max(results["periods"])) * 1.15,
        spectrum_parameters.tl_seconds * 1.05
        if spectrum_parameters.model == "e030_peru"
        else 0.0,
    ),
)
st.markdown("### Esquema referencial de la estructura")
st.caption(
    "El esquema se actualiza con los datos de la tabla o del TXT. "
    "Muestra rigidez de entrepiso, altura y peso sísmico de cada nivel."
)
st.pyplot(figures["structure"], width="stretch")

pdf_report = generate_pdf_report(
    results,
    figures,
    unit_labels,
    spectrum_description(spectrum_parameters),
)

st.markdown("### Resumen del análisis")
metric_columns = st.columns(4)
metric_columns[0].metric("Niveles", len(results["masses"]))
metric_columns[1].metric("Periodo fundamental", f"{results['periods'][0]:.3f} s")
metric_columns[2].metric(
    "Masa participante · modo 1",
    f"{results['mass_percentages'][0]:.2f}%",
)
metric_columns[3].metric(
    "Cortante SRSS en la base",
    f"{results['srss_story_shears'][0]:,.3f} {unit_labels['force']}",
)

st.download_button(
    "Exportar resultados en PDF",
    data=pdf_report,
    file_name="reporte_analisis_modal.pdf",
    mime="application/pdf",
    type="primary",
)
st.caption(
    f"Espectro utilizado: {spectrum_description(spectrum_parameters)}. "
    "Las filas de niveles están ordenadas de la base hacia la azotea."
)

tabs = st.tabs(
    ["Espectro", "Modos y participación", "Respuesta por nivel", "Matrices"]
)
with tabs[0]:
    st.pyplot(figures["spectrum"], width="stretch")
    modes_table = pd.DataFrame(
        {
            "Modo": np.arange(1, len(results["periods"]) + 1),
            "Autovalor (1/s²)": results["eigenvalues"],
            "Frecuencia (rad/s)": results["omega"],
            "Periodo (s)": results["periods"],
            "Factor Γ": results["participation_factors"],
            "Sa (g)": results["spectral_accelerations"] / gravity,
        }
    )
    st.dataframe(modes_table, hide_index=True, width="stretch")

with tabs[1]:
    left, right = st.columns(2)
    with left:
        st.pyplot(figures["modes"], width="stretch")
    with right:
        st.pyplot(figures["participation"], width="stretch")
    st.markdown("#### Formas modales normalizadas")
    mode_shapes = pd.DataFrame(
        results["modes"].T,
        index=[f"Modo {index}" for index in range(1, len(results["periods"]) + 1)],
        columns=[f"Piso {index}" for index in range(1, len(results["masses"]) + 1)],
    )
    st.dataframe(mode_shapes, width="stretch")

with tabs[2]:
    left, right = st.columns(2)
    with left:
        st.pyplot(figures["displacements"], width="stretch")
    with right:
        st.pyplot(figures["forces"], width="stretch")
    st.pyplot(figures["accelerations"], width="stretch")
    floor_results = pd.DataFrame(
        {
            "Piso": np.arange(1, len(results["masses"]) + 1),
            f"Desplazamiento SRSS ({unit_labels['displacement']})": results[
                "srss_displacements"
            ],
            f"Deriva SRSS ({unit_labels['displacement']})": results["srss_drifts"],
            "Deriva relativa SRSS (%)": results["srss_drift_ratios"] * 100,
            f"Fuerza SRSS ({unit_labels['force']})": results["srss_forces"],
            f"Cortante SRSS ({unit_labels['force']})": results["srss_story_shears"],
            "Aceleración SRSS (g)": results["srss_accelerations_g"],
        }
    )
    st.dataframe(floor_results, hide_index=True, width="stretch")

with tabs[3]:
    st.markdown("#### Matriz de masas")
    st.dataframe(
        pd.DataFrame(results["mass_matrix"]),
        hide_index=True,
        width="stretch",
    )
    st.markdown(f"#### Matriz de rigidez ({unit_labels['stiffness']})")
    st.dataframe(
        pd.DataFrame(results["stiffness_matrix"]),
        hide_index=True,
        width="stretch",
    )

for figure in figures.values():
    plt.close(figure)
