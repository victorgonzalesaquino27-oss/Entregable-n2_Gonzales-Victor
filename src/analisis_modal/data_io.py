"""Input parsing and unit conversion helpers."""

from __future__ import annotations

import csv
import io
import re
import unicodedata

import numpy as np
import pandas as pd

from .analysis import AnalysisInputError

IMPERIAL_TO_METRIC = {
    "height": 0.0254,
    "mass": 4448.2216152605 / 0.0254,
    "stiffness": 4448.2216152605 / 0.0254 / 1000.0,
}

FLOOR_COLUMNS = {
    "floor": ("piso", "nivel", "floor", "level", "story"),
    "height": ("altura", "height", "h"),
    "mass": ("masa", "mass", "m"),
    "stiffness": ("rigidez", "stiffness", "stiff", "k"),
}
REQUIRED_COLUMNS = ("height", "mass", "stiffness")


def _normalize_header(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.strip().lower())
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]", "", text)


def _identify_columns(header: list[str]) -> dict[str, int]:
    normalized = [_normalize_header(value) for value in header]
    found: dict[str, int] = {}
    for field, aliases in FLOOR_COLUMNS.items():
        for index, value in enumerate(normalized):
            if any(value == alias or value.startswith(alias) for alias in aliases):
                found[field] = index
                break
    missing = [field for field in REQUIRED_COLUMNS if field not in found]
    if missing:
        names = {"height": "Altura", "mass": "Masa", "stiffness": "Rigidez"}
        expected = ", ".join(names[field] for field in missing)
        raise AnalysisInputError(
            f"Al TXT le faltan encabezados para: {expected}. "
            "Use las columnas Piso (opcional), Altura, Masa y Rigidez."
        )
    if len(set(found.values())) != len(found):
        raise AnalysisInputError("Cada columna del TXT debe tener un encabezado distinto.")
    return found


def _parse_number(value: str, line_number: int) -> float:
    cleaned = value.strip().replace(",", ".")
    try:
        number = float(cleaned)
    except ValueError as error:
        raise AnalysisInputError(
            f"El valor '{value}' de la línea {line_number} no es numérico."
        ) from error
    if not np.isfinite(number):
        raise AnalysisInputError(f"La línea {line_number} contiene un valor no finito.")
    return number


def parse_floor_data(text: str) -> pd.DataFrame:
    """Parse a header-based semicolon, comma, tab, or whitespace-separated TXT."""
    rows = [
        (line_number, line.strip())
        for line_number, line in enumerate(text.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(rows) < 2:
        raise AnalysisInputError(
            "El TXT debe incluir una fila de encabezados y al menos un nivel."
        )

    _, header_line = rows[0]
    if ";" in header_line:
        delimiter = ";"
    elif "\t" in header_line:
        delimiter = "\t"
    elif "," in header_line:
        delimiter = ","
    else:
        delimiter = None

    def split(line: str) -> list[str]:
        return (
            next(csv.reader([line], delimiter=delimiter, skipinitialspace=True))
            if delimiter
            else re.split(r"\s+", line.strip())
        )

    header_cells = split(header_line)
    columns = _identify_columns(header_cells)
    expected_cells = len(header_cells)
    parsed: list[dict[str, float]] = []
    for line_number, line in rows[1:]:
        cells = split(line)
        if len(cells) != expected_cells:
            raise AnalysisInputError(
                f"La línea {line_number} tiene {len(cells)} valores; "
                f"se esperaban {expected_cells}."
            )
        floor = (
            _parse_number(cells[columns["floor"]], line_number)
            if "floor" in columns
            else float(len(parsed) + 1)
        )
        parsed.append(
            {
                "floor": floor,
                "height": _parse_number(cells[columns["height"]], line_number),
                "mass": _parse_number(cells[columns["mass"]], line_number),
                "stiffness": _parse_number(cells[columns["stiffness"]], line_number),
            }
        )

    parsed.sort(key=lambda row: row["floor"])
    floor_numbers = np.asarray([row["floor"] for row in parsed], dtype=float)
    expected = np.arange(1, len(parsed) + 1, dtype=float)
    if not np.array_equal(floor_numbers, expected):
        raise AnalysisInputError(
            "Los pisos deben estar numerados consecutivamente desde 1, de abajo hacia arriba."
        )
    for field in REQUIRED_COLUMNS:
        values = np.asarray([row[field] for row in parsed], dtype=float)
        if np.any(values <= 0):
            raise AnalysisInputError(
                f"Todos los valores de {field} deben ser positivos (revisar el TXT)."
            )

    return pd.DataFrame(parsed, columns=["floor", "height", "mass", "stiffness"])


def convert_floor_data(
    floor_data: pd.DataFrame,
    from_system: str,
    to_system: str,
) -> pd.DataFrame:
    """Convert height, mass, and stiffness while preserving the physical model."""
    if from_system not in ("metric", "imperial") or to_system not in (
        "metric",
        "imperial",
    ):
        raise AnalysisInputError("No se reconoce el sistema de unidades.")
    converted = floor_data.copy()
    if from_system == to_system:
        return converted

    for column in IMPERIAL_TO_METRIC:
        multiplier = IMPERIAL_TO_METRIC[column]
        if from_system == "imperial":
            converted[column] *= multiplier
        else:
            converted[column] /= multiplier
    return converted


def template_text() -> str:
    """Return a sample TXT layout for the file uploader."""
    return io.StringIO(
        "Piso;Altura;Masa;Rigidez\n"
        "1;11;2.16;108\n"
        "2;11;1.92;90\n"
        "3;11;1.68;60\n"
    ).getvalue()
