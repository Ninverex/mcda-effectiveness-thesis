"""
Warstwa Controller (MVC).

Przechwytuje "akcje analityka" (np. zadanie przeliczenia macierzy
przy uzyciu metody TOPSIS), uruchamia walidacje wejscia (WN1) i
deleguje obliczenia do wlasciwej strategii (ICalculationStrategy),
nie znajac jej szczegolow implementacyjnych.
"""

from __future__ import annotations

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.base import ICalculationStrategy, RankingResult
from mcdm.validation.validators import validate_decision_problem, ValidationError


class MCDMController:
    """Koordynator: Problem_Decyzyjny + Strategia -> RankingResult."""

    def __init__(self):
        # Rejestr dostepnych strategii, dodawany dynamicznie -- pozwala
        # na rejestracje nowych metod (np. VIKOR, MACBETH) bez zmian
        # w rdzeniu kontrolera (WN3 -- modulowosc).
        self._strategies: dict[str, ICalculationStrategy] = {}

    def register_strategy(self, strategy: ICalculationStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def available_methods(self) -> list[str]:
        return list(self._strategies.keys())

    def run(
        self,
        problem: DecisionProblem,
        method_name: str,
        require_thresholds: bool = False,
    ) -> RankingResult:
        if method_name not in self._strategies:
            raise ValueError(
                f"Nieznana metoda '{method_name}'. Dostepne: "
                f"{self.available_methods()}"
            )

        try:
            validate_decision_problem(problem, require_thresholds=require_thresholds)
        except ValidationError as exc:
            # W realnym UI (PU1) tutaj nastapilaby prezentacja bledu
            # uzytkownikowi ze wskazaniem konkretnej komorki.
            raise

        strategy = self._strategies[method_name]
        return strategy.calculate_ranking(problem)

    def run_all(
        self, problem: DecisionProblem, require_thresholds: bool = False
    ) -> dict[str, RankingResult]:
        """Uruchamia wszystkie zarejestrowane metody dla tego samego problemu."""
        return {
            name: self.run(problem, name, require_thresholds=require_thresholds)
            for name in self._strategies
        }
