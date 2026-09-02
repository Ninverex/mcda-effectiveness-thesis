"""
Wykres radarowy (profile wydajnosci alternatyw) -- warstwa View z
opisu architektury pracy: "Profile wydajnosci alternatyw (wykresy
radarowe uwidaczniajace slabe i mocne strony wariantow)".

Kazde kryterium jest normalizowane niezaleznie do przedzialu [0, 1],
gdzie 1 oznacza wartosc najlepsza wsrod aktywnych alternatyw (z
uwzglednieniem kierunku optymalizacji: dla kryterium kosztowego
najnizsza wartosc dostaje 1, dla kryterium zysku -- najwyzsza).
Dzieki temu kryteria o roznych jednostkach i skalach (np. mln PLN
i m3/h) sa porownywalne na jednym wykresie.
"""

from __future__ import annotations

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.visualization._utils import plt


def normalized_profiles(problem: DecisionProblem) -> np.ndarray:
    """
    Zwraca macierz (m x n) znormalizowanych profili w [0, 1], gdzie
    1 = najlepsza wartosc danego kryterium wsrod aktywnych alternatyw,
    0 = najgorsza. Uzywane zarowno przez wykres radarowy, jak i
    (posrednio) do interpretacji GAIA.
    """
    X = problem.active_matrix
    n = X.shape[1]
    normed = np.zeros_like(X)

    for j, direction in enumerate(problem.directions):
        col = X[:, j]
        lo, hi = col.min(), col.max()
        span = hi - lo
        if span == 0:
            # Wszystkie alternatywy maja ta sama wartosc -- brak
            # zroznicowania, neutralna wartosc srodkowa.
            normed[:, j] = 0.5
            continue
        if direction == "max":
            normed[:, j] = (col - lo) / span
        else:
            normed[:, j] = (hi - col) / span

    return normed


def plot_radar_chart(
    problem: DecisionProblem,
    alternative_names: list[str] | None = None,
    figsize: tuple[float, float] = (6.5, 6.5),
):
    """
    Rysuje wykres radarowy profili alternatyw.

    Parameters
    ----------
    problem : DecisionProblem
    alternative_names : list[str] | None
        Podzbior alternatyw do narysowania (np. tylko czolowka
        rankingu, zeby uniknac zagraconego wykresu przy wielu
        alternatywach). Domyslnie: wszystkie aktywne alternatywy.
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure
    """
    profiles = normalized_profiles(problem)
    all_names = problem.active_names
    criteria = problem.criterion_names
    n_criteria = len(criteria)

    if alternative_names is None:
        selected_idx = list(range(len(all_names)))
    else:
        selected_idx = [all_names.index(name) for name in alternative_names]

    angles = np.linspace(0, 2 * np.pi, n_criteria, endpoint=False).tolist()
    angles += angles[:1]  # zamkniecie petli wykresu

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for plot_i, idx in enumerate(selected_idx):
        values = profiles[idx].tolist()
        values += values[:1]
        color = color_cycle[plot_i % len(color_cycle)]
        ax.plot(angles, values, linewidth=2, label=all_names[idx], color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="gray")
    ax.set_title(
        "Profile wydajnosci alternatyw\n(1.0 = najlepsza wartosc danego kryterium)",
        fontsize=10,
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)

    fig.tight_layout()
    return fig
