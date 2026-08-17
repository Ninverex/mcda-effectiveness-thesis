"""
Analiza wrazliwosci rankingow na zmiany wag kryteriow -- rozdzial
4.3 pracy oraz scenariusze 2.4.1 z metodyki badan.

Dla wskazanego kryterium generowana jest seria wag (perturbacja +/-
wokol wartosci bazowej), a pozostale wagi sa proporcjonalnie
przeskalowywane tak, by suma nadal wynosila 1.0. Dla kazdego wariantu
wag uruchamiana jest wskazana strategia MCDM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.base import ICalculationStrategy


@dataclass
class SensitivityPoint:
    """Pojedynczy punkt analizy wrazliwosci."""

    delta: float
    weights: np.ndarray
    ranking_names: list[str]
    scores: np.ndarray


@dataclass
class SensitivityReport:
    method_name: str
    criterion_name: str
    points: list[SensitivityPoint]

    def top_alternative_series(self) -> list[str]:
        """Lider rankingu dla kazdego punktu -- szybki podglad stabilnosci."""
        return [p.ranking_names[0] for p in self.points]

    def is_stable(self) -> bool:
        """True, jesli lider rankingu nie zmienia sie w calym zakresie."""
        series = self.top_alternative_series()
        return len(set(series)) == 1


def weight_sensitivity_analysis(
    problem: DecisionProblem,
    strategy: ICalculationStrategy,
    criterion_name: str,
    delta_range: np.ndarray | None = None,
) -> SensitivityReport:
    """
    Parameters
    ----------
    problem : DecisionProblem
    strategy : ICalculationStrategy
    criterion_name : str
        Nazwa kryterium, ktorego waga bedzie modyfikowana.
    delta_range : np.ndarray | None
        Zakres przesuniec wagi bazowej (np. od -0.2 do +0.2). Wartosc
        wagi jest przycinana do [0, 1) przed renormalizacja.
        Domyslnie: 9 punktow rownomiernie w [-0.2, 0.2].

    Returns
    -------
    SensitivityReport
    """
    if criterion_name not in problem.criterion_names:
        raise ValueError(f"Nieznane kryterium: '{criterion_name}'")

    if delta_range is None:
        delta_range = np.linspace(-0.2, 0.2, 9)

    idx = problem.criterion_names.index(criterion_name)
    points = []

    for delta in delta_range:
        new_weights = _perturb_and_renormalize(problem.weights, idx, delta)
        modified_problem = problem.copy_with(weights=new_weights.tolist())
        result = strategy.calculate_ranking(modified_problem)

        points.append(
            SensitivityPoint(
                delta=float(delta),
                weights=new_weights,
                ranking_names=result.as_ordered_names(),
                scores=result.scores,
            )
        )

    return SensitivityReport(
        method_name=strategy.name,
        criterion_name=criterion_name,
        points=points,
    )


def _perturb_and_renormalize(
    weights: np.ndarray, criterion_idx: int, delta: float
) -> np.ndarray:
    """
    Przesuwa wage wskazanego kryterium o `delta`, a nadwyzke/deficyt
    rozklada proporcjonalnie na pozostale wagi tak, aby suma nadal
    wynosila 1.0 i zadna waga nie byla ujemna.
    """
    weights = np.asarray(weights, dtype=float).copy()
    n = len(weights)

    new_value = np.clip(weights[criterion_idx] + delta, 0.0, 0.999999)
    actual_delta = new_value - weights[criterion_idx]

    others_idx = [i for i in range(n) if i != criterion_idx]
    others_sum = weights[others_idx].sum()

    if others_sum <= 1e-12:
        # Skrajny przypadek: pozostale wagi sa (prawie) zerowe --
        # rozklad rownomierny nadwyzki.
        weights[others_idx] -= actual_delta / len(others_idx)
    else:
        # Proporcjonalne przeskalowanie pozostalych wag
        scale = (others_sum - actual_delta) / others_sum
        weights[others_idx] *= scale

    weights[criterion_idx] = new_value
    weights = np.clip(weights, 0.0, None)
    weights /= weights.sum()  # korekta bledow numerycznych, suma dokladnie 1.0

    return weights
