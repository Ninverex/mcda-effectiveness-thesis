"""
Miary zgodnosci (korelacji) rankingow -- rozdzial 4.5 pracy.

Uzywane do ilosciowej oceny, jak bardzo rozne metody MCDM zgadzaja
sie ze soba na tym samym zbiorze danych. Wysoka korelacja (blisko 1)
oznacza duza zgodnosc rankingow; wartosci bliskie 0 lub ujemne
sygnalizuja istotne rozbieznosci -- co jest jednym z glownych
przedmiotow badania w tej pracy.
"""

from __future__ import annotations

from itertools import combinations

from scipy.stats import kendalltau, spearmanr

from mcdm.strategies.base import RankingResult


def compare_rankings(ranking_a: list[int], ranking_b: list[int]) -> dict:
    """
    Porownuje dwa rankingi (listy indeksow alternatyw, od najlepszej
    do najgorszej) przy uzyciu wspolczynnikow Kendalla tau i
    Spearmana rho.

    Oba rankingi musza dotyczyc tego samego zbioru alternatyw
    (te same indeksy, mozliwie w innej kolejnosci).

    Returns
    -------
    dict z kluczami "kendall_tau", "kendall_p_value",
    "spearman_rho", "spearman_p_value".
    """
    if set(ranking_a) != set(ranking_b):
        raise ValueError(
            "Rankingi musza dotyczyc dokladnie tego samego zbioru "
            f"alternatyw. Otrzymano {set(ranking_a)} vs {set(ranking_b)}."
        )

    # Zamiana "kolejnosci" (ranking) na "pozycje" (rank) per alternatywa,
    # zeby porownac zgodnosc pozycji, a nie surowej listy indeksow.
    positions_a = _ranking_to_positions(ranking_a)
    positions_b = _ranking_to_positions(ranking_b)

    n = len(ranking_a)
    ordered_ids = range(n)
    a_positions = [positions_a[i] for i in ordered_ids]
    b_positions = [positions_b[i] for i in ordered_ids]

    tau, tau_p = kendalltau(a_positions, b_positions)
    rho, rho_p = spearmanr(a_positions, b_positions)

    return {
        "kendall_tau": tau,
        "kendall_p_value": tau_p,
        "spearman_rho": rho,
        "spearman_p_value": rho_p,
    }


def compare_all_pairs(results: dict[str, RankingResult]) -> dict[tuple[str, str], dict]:
    """
    Porownuje wszystkie unikalne pary wynikow (np. AHP vs TOPSIS,
    AHP vs PROMETHEE, ...) z dict {method_name: RankingResult}.

    Returns
    -------
    dict {(method_a, method_b): {"kendall_tau": ..., "spearman_rho": ...}}
    """
    output = {}
    for name_a, name_b in combinations(results.keys(), 2):
        output[(name_a, name_b)] = compare_rankings(
            results[name_a].ranking, results[name_b].ranking
        )
    return output


def _ranking_to_positions(ranking: list[int]) -> dict[int, int]:
    """np. ranking=[2,0,1] -> {2:0, 0:1, 1:2} (indeks alternatywy -> pozycja)"""
    return {alt_idx: position for position, alt_idx in enumerate(ranking)}
