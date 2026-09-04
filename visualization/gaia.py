"""
Plaszczyzna GAIA (Geometrical Analysis for Interactive Aid),
Brans i Mareschal -- geometryczna interpretacja wynikow PROMETHEE.

Wejsciem jest macierz (m x n) jednokryterialnych przeplywow netto
(PrometheeStrategy.unicriterion_net_flows) -- kazdy wiersz to profil
alternatywy w przestrzeni kryteriow, kazda kolumna to os odpowiadajaca
jednemu kryterium. PCA rzutuje ta przestrzen do 2D, zachowujac jak
najwiecej wariancji (czyli jak najwiecej informacji o roznicach
miedzy alternatywami).

Interpretacja plaszczyzny GAIA:
- Alternatywy blisko siebie na wykresie sa do siebie podobne pod
  wzgledem profilu ocen.
- Kryteria (rysowane jako wektory/promienie z centrum) wskazujace w
  podobnym kierunku sa ze soba zgodne (niesprzeczne); kryteria
  wskazujace w przeciwnych kierunkach sa ze soba w konflikcie.
- Alternatywa lezaca "w kierunku" danego kryterium jest w nim dobra.

Referencja: Behzadian i in. "PROMETHEE: A comprehensive literature
review...", 2010 -- pozycja z bibliografii pracy.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.promethee import PrometheeStrategy
from mcdm.visualization._utils import plt


def plot_gaia_plane(
    problem: DecisionProblem,
    promethee_strategy: PrometheeStrategy | None = None,
    figsize: tuple[float, float] = (7.5, 5.5),
):
    """
    Rysuje plaszczyzne GAIA: alternatywy jako punkty, kryteria jako
    wektory z centrum ukladu.

    Parameters
    ----------
    problem : DecisionProblem
    promethee_strategy : PrometheeStrategy | None
        Instancja strategii (progi/parametry funkcji preferencji).
        Domyslnie: nowa instancja z domyslnymi parametrami.
    figsize : tuple[float, float]

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        Gdy problem ma mniej niz 3 kryteria -- PCA do 2D wymaga co
        najmniej tylu wymiarow, zeby rzutowanie mialo sens
        (przy 2 kryteriach plaszczyzna GAIA pokrywalaby sie z
        oryginalna przestrzenia i nie wnosi dodatkowej wartosci).
    """
    if problem.n_criteria < 3:
        raise ValueError(
            "Plaszczyzna GAIA wymaga co najmniej 3 kryteriow "
            f"(otrzymano {problem.n_criteria}) -- przy 2 kryteriach "
            "rzutowanie PCA nie wnosi nic ponad oryginalny wykres."
        )

    strategy = promethee_strategy or PrometheeStrategy()
    flows = strategy.unicriterion_net_flows(problem)  # (m, n)

    # Centrowanie (PCA wymaga danych o srednim 0), bez skalowania
    # wariancji -- w GAIA dlugosc wektora kryterium ma znaczenie
    # (odzwierciedla, jak bardzo dane kryterium roznicuje alternatywy).
    centered = flows - flows.mean(axis=0)

    pca = PCA(n_components=2)
    alt_coords = pca.fit_transform(centered)  # (m, 2)

    # Wspolrzedne kryteriow w przestrzeni GAIA: rzutowanie osi
    # jednostkowych oryginalnej przestrzeni przez te sama transformacje.
    criteria_coords = pca.components_.T * np.sqrt(pca.explained_variance_)
    # Skalowanie wektorow kryteriow tak, by byly czytelne na tle
    # rozrzutu punktow alternatyw.
    scale = np.max(np.linalg.norm(alt_coords, axis=1)) / (
        np.max(np.linalg.norm(criteria_coords, axis=1)) + 1e-12
    )
    criteria_coords_scaled = criteria_coords * scale * 0.9

    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
    ax.axvline(0, color="lightgray", linewidth=0.8, zorder=0)

    ax.scatter(alt_coords[:, 0], alt_coords[:, 1], s=70, color="#4c5fe0", zorder=3)

    x_span = alt_coords[:, 0].max() - alt_coords[:, 0].min()
    x_center = (alt_coords[:, 0].max() + alt_coords[:, 0].min()) / 2

    def _label_offset(x: float, index: int, magnitude: int) -> tuple[tuple[int, int], str]:
        """Etykieta skierowana w strone srodka wykresu (nie w margines),
        z naprzemiennym przesunieciem pionowym wg indeksu -- redukuje
        nakladanie sie tekstu przy punktach lezacych blisko siebie."""
        dx = magnitude if x <= x_center else -magnitude
        dy = 10 if index % 2 == 0 else -16
        ha = "left" if dx > 0 else "right"
        return (dx, dy), ha

    for i, (name, (x, y)) in enumerate(zip(problem.active_names, alt_coords)):
        offset, ha = _label_offset(x, i, magnitude=7)
        ax.annotate(
            name, (x, y), textcoords="offset points", xytext=offset,
            fontsize=8, ha=ha,
        )

    for i, (name, (x, y)) in enumerate(
        zip(problem.criterion_names, criteria_coords_scaled)
    ):
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#e07a1f", lw=1.6),
            zorder=2,
        )
        # Etykieta umieszczona DALEJ wzdluz tego samego kierunku co
        # wektor kryterium (a nie stalym przesunieciem w pikselach) --
        # naturalnie oddala ja od skupiska punktow alternatyw blisko
        # poczatku ukladu, zamiast tylko przemiennie gora/dol.
        label_x, label_y = x * 1.22, y * 1.22
        dy = 8 if i % 2 == 0 else -22
        ha = "left" if label_x >= 0 else "right"
        ax.annotate(
            name,
            (label_x, label_y),
            textcoords="offset points",
            xytext=(0, dy),
            fontsize=8,
            color="#b35c12",
            fontweight="bold",
            ha=ha,
        )

    explained = pca.explained_variance_ratio_.sum() * 100
    ax.set_title(
        f"Plaszczyzna GAIA (wyjasniona wariancja: {explained:.0f}%)",
        fontsize=10,
    )
    ax.set_xlabel("Skladowa glowna 1")
    ax.set_ylabel("Skladowa glowna 2")

    # Granice osi z niezaleznym marginesem dla X i Y (nie wymuszamy
    # scisle rownych proporcji 1:1). W klasycznym ujeciu GAIA rowne
    # proporcje sa pozadane, bo katy miedzy wektorami kryteriow niosa
    # informacje o ich zgodnosci/konflikcie. W praktyce jednak, gdy
    # 1. skladowa PCA silnie dominuje wariancje (czesty przypadek przy
    # niewielkiej liczbie alternatyw), wymuszenie 1:1 razem z
    # przycinaniem figury do faktycznej tresci (bbox_inches="tight")
    # zapadalo caly wykres do nieczytelnego, plaskiego paska.
    # Priorytetyzujemy tu czytelnosc etykiet -- ogolny kierunek
    # (zgodnosc vs konflikt kryteriow) pozostaje widoczny, tylko bez
    # scislej metrycznej dokladnosci katow.
    all_points = np.vstack([alt_coords, criteria_coords_scaled, [[0, 0]]])
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    x_pad = (x_max - x_min) * 0.3 + 1e-6
    y_pad = (y_max - y_min) * 0.6 + 1e-6
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    fig.tight_layout()
    return fig
