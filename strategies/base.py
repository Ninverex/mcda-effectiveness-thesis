"""
Wzorzec Strategia dla metod MCDM.

Kazda metoda (AHP, TOPSIS, PROMETHEE, ELECTRE) implementuje ten sam
interfejs ICalculationStrategy, dzięki czemu warstwa Controller i
modul ewaluacji (Kendall/Spearman, rank reversal, sensitivity)
mogą operować na wynikach w sposób jednolity, niezależny od
konkretnego algorytmu (Open/Closed Principle).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from mcdm.models.decision_problem import DecisionProblem


@dataclass
class RankingResult:
    """
    Ujednolicony wynik zwracany przez kazda strategie obliczeniowa.

    Attributes
    ----------
    method_name : str
        Nazwa metody, np. "TOPSIS", "AHP".
    scores : np.ndarray
        Wynik koncowy per alternatywa (im wyzszy, tym lepiej),
        w kolejnosci zgodnej z problem.active_names.
    ranking : list[int]
        Indeksy alternatyw (wzgledem active_names) posortowane
        od najlepszej do najgorszej.
    alternative_names : list[str]
        Nazwy alternatyw odpowiadajace pozycjom w scores/ranking.
    intermediate : dict
        Wyniki posrednie specyficzne dla metody, np.:
        - AHP: {"CI": ..., "CR": ..., "eigenvector": ...}
        - TOPSIS: {"PIS": ..., "NIS": ..., "D_plus": ..., "D_minus": ...}
        - PROMETHEE: {"phi_plus": ..., "phi_minus": ..., "phi_net": ...}
        - ELECTRE: {"concordance": ..., "discordance": ..., "outranking": ...}
    """

    method_name: str
    scores: np.ndarray
    ranking: list[int]
    alternative_names: list[str]
    intermediate: dict = field(default_factory=dict)

    def as_ordered_names(self) -> list[str]:
        """Nazwy alternatyw uszeregowane od najlepszej do najgorszej."""
        return [self.alternative_names[i] for i in self.ranking]

    def top(self, k: int = 1) -> list[str]:
        return self.as_ordered_names()[:k]

    def __repr__(self) -> str:
        order = " > ".join(self.as_ordered_names())
        return f"RankingResult({self.method_name}): {order}"


class ICalculationStrategy(ABC):
    """Wspólny interfejs dla wszystkich metod MCDM."""

    #: Nazwa wyświetlana metody - nadpisywana w klasach potomnych.
    name: str = "AbstractStrategy"

    @abstractmethod
    def calculate_ranking(self, problem: DecisionProblem) -> RankingResult:
        """
        Wykonuje pełne obliczenia dla podanego problemu decyzyjnego
        i zwraca ujednolicony RankingResult.
        """
        raise NotImplementedError
