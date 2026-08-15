"""
Symulacja zjawiska odwrocenia rankingu (rank reversal) -- rozdzial
4.4 pracy oraz UC PU4 z opisu przypadkow uzycia.

Mechanizm wykorzystuje flage `active_mask` (Status_Aktywnosci) z
DecisionProblem: alternatywa jest "wylaczana" z obliczen bez
fizycznego usuwania jej z danych, co pozwala szybko i powtarzalnie
przeliczac model dla roznych podzbiorow alternatyw.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.base import ICalculationStrategy, RankingResult


@dataclass
class RankReversalReport:
    """Wynik porownania rankingu przed i po usunieciu alternatywy."""

    method_name: str
    removed_alternative: str
    baseline_order: list[str]
    modified_order: list[str]
    position_changes: dict[str, tuple[int, int | None]]
    reversal_detected: bool

    def __repr__(self) -> str:
        status = "WYKRYTO odwrocenie" if self.reversal_detected else "brak odwrocenia"
        return (
            f"RankReversalReport({self.method_name}, usunieto="
            f"'{self.removed_alternative}', {status})"
        )


def simulate_rank_reversal(
    problem: DecisionProblem,
    strategy: ICalculationStrategy,
    alternative_name: str,
    top_k: int = 3,
) -> RankReversalReport:
    """
    Usuwa (dezaktywuje) wskazana alternatywe z problemu, ponownie
    przelicza ranking i porownuje kolejnosc w scislej czolowce
    (`top_k` pozycji) przed i po.

    Parameters
    ----------
    problem : DecisionProblem
        Oryginalny problem decyzyjny (z pelnym active_mask).
    strategy : ICalculationStrategy
        Metoda MCDM, ktorej stabilnosc badamy (typowo AHP lub TOPSIS,
        zgodnie z hipoteza badawcza pracy o ich podatnosci na rank
        reversal, w kontrascie do PROMETHEE/ELECTRE).
    alternative_name : str
        Nazwa alternatywy do usuniecia z macierzy decyzyjnej.
    top_k : int
        Liczba czolowych pozycji rankingu branych pod uwage przy
        wykrywaniu odwrocenia.

    Returns
    -------
    RankReversalReport
    """
    if alternative_name not in problem.alternative_names:
        raise ValueError(f"Nieznana alternatywa: '{alternative_name}'")

    remove_idx = problem.alternative_names.index(alternative_name)
    if not problem.active_mask[remove_idx]:
        raise ValueError(
            f"Alternatywa '{alternative_name}' jest juz nieaktywna w problemie."
        )

    baseline_result = strategy.calculate_ranking(problem)
    baseline_order = baseline_result.as_ordered_names()

    new_mask = problem.active_mask.copy()
    new_mask[remove_idx] = False
    modified_problem = problem.copy_with(active_mask=new_mask.tolist())

    modified_result = strategy.calculate_ranking(modified_problem)
    modified_order = modified_result.as_ordered_names()

    position_changes = _compare_positions(baseline_order, modified_order)

    reversal_detected = _has_top_k_reversal(
        baseline_order, modified_order, alternative_name, top_k
    )

    return RankReversalReport(
        method_name=strategy.name,
        removed_alternative=alternative_name,
        baseline_order=baseline_order,
        modified_order=modified_order,
        position_changes=position_changes,
        reversal_detected=reversal_detected,
    )


def _compare_positions(
    baseline_order: list[str], modified_order: list[str]
) -> dict[str, tuple[int, int | None]]:
    """
    Dla kazdej alternatywy (poza usunieta) zwraca krotke
    (pozycja_przed, pozycja_po). Pozycje liczone od 1.
    """
    baseline_pos = {name: pos + 1 for pos, name in enumerate(baseline_order)}
    modified_pos = {name: pos + 1 for pos, name in enumerate(modified_order)}

    changes = {}
    for name in baseline_order:
        if name in modified_pos:
            changes[name] = (baseline_pos[name], modified_pos[name])
        else:
            changes[name] = (baseline_pos[name], None)  # to byla usunieta alternatywa
    return changes


def _has_top_k_reversal(
    baseline_order: list[str],
    modified_order: list[str],
    removed_name: str,
    top_k: int,
) -> bool:
    """
    Sprawdza, czy wzgledna kolejnosc alternatyw w scislej czolowce
    (top_k, po wykluczeniu usunietej alternatywy z obu list) zmienila
    sie po usunieciu jednej z pozostalych (slabszych) alternatyw.
    To jest klasyczna definicja rank reversal: usuniecie alternatywy
    spoza czolowki nie powinno zmieniac wzajemnej kolejnosci lepszych
    alternatyw.
    """
    baseline_top = [n for n in baseline_order if n != removed_name][:top_k]
    modified_top = [n for n in modified_order if n != removed_name][:top_k]
    return baseline_top != modified_top
