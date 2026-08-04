"""
Walidacja danych wejsciowych (wymaganie niefunkcjonalne WN1).

Sprawdzane sa m.in.:
- brak wartosci pustych / NaN w macierzy decyzyjnej,
- poprawnosc typow (wartosci liczbowe),
- suma wag kryteriow rowna 1.0 (z tolerancja),
- spojnosc wymiarow (liczba wag == liczba kryteriow, liczba nazw
  alternatyw == liczba wierszy macierzy itd.),
- obecnosc wymaganych progow (q, p, v) gdy uzywana jest metoda
  z rodziny ELECTRE/PROMETHEE.
"""

from __future__ import annotations

import numpy as np

from mcdm.models.decision_problem import DecisionProblem


class ValidationError(Exception):
    """Podnoszony gdy dane wejsciowe nie spelniaja wymagan WN1."""


def validate_decision_problem(
    problem: DecisionProblem,
    weight_tolerance: float = 1e-6,
    require_thresholds: bool = False,
) -> None:
    """
    Waliduje DecisionProblem. Rzuca ValidationError z precyzyjnym
    komunikatem wskazujacym blad (zgodnie z UC PU1: "System wyswietla
    precyzyjny komunikat o bledzie ze wskazaniem blednej komorki").
    """
    _validate_shapes(problem)
    _validate_no_missing_values(problem)
    _validate_weights(problem, weight_tolerance)
    if require_thresholds:
        _validate_thresholds(problem)


def _validate_shapes(problem: DecisionProblem) -> None:
    m, n = problem.matrix.shape

    if len(problem.alternative_names) != m:
        raise ValidationError(
            f"Liczba nazw alternatyw ({len(problem.alternative_names)}) "
            f"nie zgadza sie z liczba wierszy macierzy ({m})."
        )
    if len(problem.criterion_names) != n:
        raise ValidationError(
            f"Liczba nazw kryteriow ({len(problem.criterion_names)}) "
            f"nie zgadza sie z liczba kolumn macierzy ({n})."
        )
    if len(problem.weights) != n:
        raise ValidationError(
            f"Liczba wag ({len(problem.weights)}) nie zgadza sie "
            f"z liczba kryteriow ({n})."
        )
    if len(problem.directions) != n:
        raise ValidationError(
            f"Liczba kierunkow optymalizacji ({len(problem.directions)}) "
            f"nie zgadza sie z liczba kryteriow ({n})."
        )
    if len(problem.active_mask) != m:
        raise ValidationError(
            f"Dlugosc active_mask ({len(problem.active_mask)}) "
            f"nie zgadza sie z liczba alternatyw ({m})."
        )
    if problem.n_alternatives < 2:
        raise ValidationError(
            "Do przeprowadzenia rankingu potrzebne sa co najmniej "
            "2 aktywne alternatywy."
        )


def _validate_no_missing_values(problem: DecisionProblem) -> None:
    nan_positions = np.argwhere(np.isnan(problem.matrix))
    if nan_positions.size > 0:
        cells = ", ".join(
            f"(wiersz={r}, kolumna={c})" for r, c in nan_positions[:10]
        )
        raise ValidationError(
            f"Macierz decyzyjna zawiera brakujace wartosci w komorkach: {cells}"
            + (" ... (i wiecej)" if nan_positions.shape[0] > 10 else "")
        )


def _validate_weights(problem: DecisionProblem, tolerance: float) -> None:
    if np.any(problem.weights < 0):
        raise ValidationError("Wagi kryteriow nie moga byc ujemne.")

    total = float(np.sum(problem.weights))
    if abs(total - 1.0) > tolerance:
        raise ValidationError(
            f"Suma wag kryteriow musi wynosic 1.0, otrzymano {total:.6f}. "
            "Znormalizuj wektor wag przed uruchomieniem obliczen."
        )


def _validate_thresholds(problem: DecisionProblem) -> None:
    if not problem.thresholds:
        raise ValidationError(
            "Wybrana metoda (ELECTRE/PROMETHEE) wymaga zdefiniowania "
            "progow indyferencji (q), preferencji (p) i weta (v)."
        )
    for key in ("q", "p", "v"):
        if key not in problem.thresholds:
            continue  # v bywa opcjonalne w niektorych wariantach ELECTRE
        values = problem.thresholds[key]
        if values is not None and len(values) != problem.n_criteria:
            raise ValidationError(
                f"Prog '{key}' musi byc zdefiniowany dla kazdego z "
                f"{problem.n_criteria} kryteriow (otrzymano {len(values)})."
            )
