"""
Wspolne narzedzia dla warstwy wizualizacji.

Ustawia backend "Agg" (bezekranowy, renderowanie do pliku/bajtow) --
niezbedne, zeby matplotlib dzialal poprawnie w procesie serwera Flask
(bez dostepu do wyswietlacza) oraz w skryptach uruchamianych z linii
komend bez srodowiska graficznego.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (musi byc po matplotlib.use)


def figure_to_base64(fig: plt.Figure, dpi: int = 110) -> str:
    """
    Konwertuje figure matplotlib do base64-owanego PNG, gotowego do
    osadzenia bezposrednio w szablonie HTML jako:
    <img src="data:image/png;base64,{{ encoded }}">

    Zamyka figure po konwersji, zeby nie zostawiac otwartych obiektow
    matplotlib w dlugo dzialajacym procesie serwera (wyciek pamieci
    przy wielu zadaniach HTTP pod rzad).
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    """Zapisuje figure do pliku PNG (uzywane przez experiments/run_comparison.py)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
