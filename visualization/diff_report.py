"""
Raport roznicowy odwrocenia rankingu (rank reversal) -- warstwa View
z opisu architektury pracy: "Raporty roznicowe eksponujace zjawisko
odwrocenia rankingu", odpowiadajace UC PU4 (widok diff po symulacji
usuniecia alternatywy).

Udostepnia dwie formy tego samego raportu:
- build_diff_html_table() -- tabela HTML z kolorowaniem wierszy,
  gotowa do wstawienia bezposrednio w szablon Jinja2 (przez |safe).
- plot_rank_reversal_diff() -- wykres typu "slope chart" (wykres
  nachylen), laczacy pozycje alternatyw przed i po liniami; kolor
  linii sygnalizuje, czy dana alternatywa zmienila pozycje.
"""

from __future__ import annotations

from mcdm.evaluation.rank_reversal import RankReversalReport
from mcdm.visualization._utils import plt


def build_diff_html_table(report: RankReversalReport) -> str:
    """
    Buduje tabele HTML (bez zaleznosci od konkretnego frameworka
    szablonow) prezentujaca zmiany pozycji alternatyw przed/po
    usunieciu jednej z nich. Wiersze sa kolorowane:
    - szary/przekreslony: alternatywa usunieta z rankingu,
    - zielony: pozycja bez zmian,
    - bursztynowy: pozycja zmieniona (przesuniecie w gore lub w dol).

    Zwracany string jest gotowym fragmentem HTML -- w Jinja2 nalezy
    go wstawic z filtrem |safe, np. {{ diff_table_html|safe }}.
    """
    rows_html = []
    for name, (before, after) in report.position_changes.items():
        if after is None:
            row_class = "diff-row-removed"
            after_display = "usunieta"
            change_display = "&mdash;"
        elif after == before:
            row_class = "diff-row-unchanged"
            after_display = str(after)
            change_display = "bez zmian"
        else:
            row_class = "diff-row-changed"
            after_display = str(after)
            delta = before - after
            arrow = "&uarr;" if delta > 0 else "&darr;"
            change_display = f"{arrow} {abs(delta)}"

        rows_html.append(
            f'<tr class="{row_class}">'
            f"<td>{name}</td>"
            f"<td>{before}</td>"
            f"<td>{after_display}</td>"
            f"<td>{change_display}</td>"
            f"</tr>"
        )

    return (
        '<table class="data-table diff-table">'
        "<thead><tr>"
        "<th>Alternatywa</th><th>Pozycja przed</th>"
        "<th>Pozycja po</th><th>Zmiana</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table>"
    )


def plot_rank_reversal_diff(
    report: RankReversalReport, figsize: tuple[float, float] = (5.5, 4.5)
):
    """
    Rysuje wykres nachylen (slope chart): dla kazdej alternatywy
    (poza usunieta) linia laczy jej pozycje w rankingu bazowym z
    pozycja po usunieciu jednej z alternatyw. Pozycja 1 (najlepsza)
    jest na gorze wykresu.

    Kolory:
    - szary: pozycja bez zmian,
    - bursztynowy: pozycja zmieniona (widoczne odwrocenie rankingu
      w obreble analizowanych alternatyw).

    Returns
    -------
    matplotlib.figure.Figure
    """
    entries = [
        (name, before, after)
        for name, (before, after) in report.position_changes.items()
        if after is not None
    ]
    # Sortowanie po pozycji bazowej -- czytelniejszy wykres
    entries.sort(key=lambda e: e[1])

    fig, ax = plt.subplots(figsize=figsize)

    max_pos = max(max(e[1], e[2]) for e in entries)

    for name, before, after in entries:
        changed = before != after
        color = "#e07a1f" if changed else "#9aa3b2"
        linewidth = 2.2 if changed else 1.2
        ax.plot([0, 1], [before, after], color=color, linewidth=linewidth, zorder=2)
        ax.scatter([0, 1], [before, after], color=color, s=28, zorder=3)
        label_side_offset = -0.03
        ax.annotate(
            name,
            (0, before),
            textcoords="offset points",
            xytext=(-14, 0),
            ha="right",
            va="center",
            fontsize=8,
        )
        ax.annotate(
            name,
            (1, after),
            textcoords="offset points",
            xytext=(8, 0),
            ha="left",
            va="center",
            fontsize=8,
        )

    ax.set_xlim(-0.9, 1.6)
    ax.set_ylim(max_pos + 0.7, 0.3)  # odwrocona os Y: pozycja 1 na gorze
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Przed", f"Po usunieciu\n'{report.removed_alternative}'"])
    ax.set_ylabel("Pozycja w rankingu")
    ax.set_yticks(range(1, max_pos + 1))
    ax.tick_params(axis="y", pad=10)
    title_suffix = " -- WYKRYTO odwrocenie" if report.reversal_detected else ""
    ax.set_title(f"{report.method_name}: zmiana pozycji{title_suffix}", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return fig
