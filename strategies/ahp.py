"""
Implementacja metody AHP (Analytic Hierarchy Process, Saaty 1980).

W tej implementacji AHP jest uzywane w wariancie "jednopoziomowym":
- wagi kryteriow pochodza z macierzy porownan parami (skala Saaty'ego 1-9),
- oceny alternatyw per kryterium pochodza wprost z macierzy decyzyjnej
  problem.matrix (znormalizowanej kolumnowo), a nie z kolejnych macierzy
  porownan parami alternatywa-vs-alternatywa dla kazdego kryterium.

Jest to swiadome uproszczenie (typowe w pracach porownawczych MCDM,
por. Zanakis i in. 1998), ktore pozwala uruchomic AHP na tych samych
danych wejsciowych (macierz X, kierunki optymalizacji) co TOPSIS,
PROMETHEE i ELECTRE -- co jest kluczowe dla eksperymentow
porownawczych w rozdziale 4 pracy. Pelny "klasyczny" AHP z
macierzami porownan parami alternatyw jest tez dostepny przez
metode `rank_from_pairwise_matrices`, gdyby promotor wymagal
scislejszej wiernosci oryginalnej metodzie Saaty'ego.

Kroki (3.2 w spisie tresci):
1. Macierz porownan parami kryteriow A (n x n), A[i,j] = waznosc
   kryterium i wzgledem j w skali Saaty'ego 1-9 (lub 1/1..1/9).
2. Wektor wag = znormalizowany glowny wektor wlasny A (odpowiadajacy
   najwiekszej wartosci wlasnej lambda_max).
3. CI = (lambda_max - n) / (n - 1).
4. CR = CI / RI, gdzie RI to losowy indeks spojnosci Saaty'ego.
5. Jesli CR > 0.10 -> ostrzezenie o braku spojnosci osadow (PU2).
6. Ranking koncowy: wazona suma znormalizowanych ocen alternatyw
   (klasyczna synteza addytywna AHP).
"""

from __future__ import annotations

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.base import ICalculationStrategy, RankingResult


# Losowy Indeks Spojnosci (Random Index) wg Saaty'ego, dla n = 1..15
RANDOM_INDEX = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}

CR_THRESHOLD = 0.10


class ConsistencyWarning(UserWarning):
    """Ostrzezenie o przekroczeniu progu CR > 0.10 (PU2 - blokada obliczen)."""


class AhpStrategy(ICalculationStrategy):

    name = "AHP"

    def __init__(self, pairwise_criteria_matrix: np.ndarray | None = None,
                 raise_on_inconsistency: bool = False):
        """
        Parameters
        ----------
        pairwise_criteria_matrix : np.ndarray | None
            Macierz porownan parami kryteriow (n x n) w skali Saaty'ego.
            Jesli None, wagi sa brane bezposrednio z problem.weights
            (traktowane jako juz wyznaczony wektor priorytetow) --
            przydatne w eksperymentach porownawczych, gdzie ten sam
            wektor wag ma byc uzyty przez wszystkie metody.
        raise_on_inconsistency : bool
            Jesli True, CR > 0.10 podnosi wyjatek zamiast tylko
            zapisac ostrzezenie w intermediate["consistency_ok"].
        """
        self.pairwise_criteria_matrix = pairwise_criteria_matrix
        self.raise_on_inconsistency = raise_on_inconsistency

    def calculate_ranking(self, problem: DecisionProblem) -> RankingResult:
        X = problem.active_matrix
        directions = problem.directions

        if self.pairwise_criteria_matrix is not None:
            weights, ci, cr, eigenvector = self._weights_from_pairwise_matrix(
                self.pairwise_criteria_matrix
            )
            if cr > CR_THRESHOLD:
                message = (
                    f"Wspolczynnik spojnosci CR={cr:.4f} przekracza prog 0.10 -- "
                    "osady eksperckie sa niespojne, nalezy je zrewidowac."
                )
                if self.raise_on_inconsistency:
                    raise ValueError(message)
        else:
            # Wagi juz wyznaczone (np. wprost z problem.weights) --
            # brak macierzy porownan parami do liczenia CI/CR.
            weights = problem.weights
            ci = cr = None
            eigenvector = weights

        # Normalizacja kolumnowa macierzy decyzyjnej (suma kolumny = 1),
        # z uwzglednieniem kierunku optymalizacji -- klasyczny krok
        # syntezy AHP dla ocen alternatyw wzgledem kryteriow.
        normalized = np.zeros_like(X)
        for j, direction in enumerate(directions):
            col = X[:, j]
            if direction == "max":
                total = col.sum()
                total = total if total != 0 else 1e-12
                normalized[:, j] = col / total
            else:  # "min" -- odwracamy tak, by wieksza wartosc = lepsza
                inv = 1.0 / np.where(col == 0, 1e-12, col)
                total = inv.sum()
                total = total if total != 0 else 1e-12
                normalized[:, j] = inv / total

        # Synteza: wazona suma priorytetow
        scores = normalized @ weights
        ranking = list(np.argsort(-scores))

        return RankingResult(
            method_name=self.name,
            scores=scores,
            ranking=ranking,
            alternative_names=problem.active_names,
            intermediate={
                "criteria_weights": weights,
                "eigenvector": eigenvector,
                "CI": ci,
                "CR": cr,
                "consistency_ok": (cr is None) or (cr <= CR_THRESHOLD),
                "normalized_matrix": normalized,
            },
        )

    # ------------------------------------------------------------------
    # Pomocnicze: klasyczna procedura wektora wlasnego + CI/CR
    # ------------------------------------------------------------------

    @staticmethod
    def _weights_from_pairwise_matrix(A: np.ndarray):
        A = np.asarray(A, dtype=float)
        n = A.shape[0]
        if A.shape != (n, n):
            raise ValueError("Macierz porownan parami musi byc kwadratowa.")

        eigenvalues, eigenvectors = np.linalg.eig(A)
        max_idx = int(np.argmax(eigenvalues.real))
        lambda_max = eigenvalues.real[max_idx]

        eigenvector = eigenvectors[:, max_idx].real
        # Wektor wlasny moze wyjsc ze znakiem ujemnym -- normalizujemy
        # do wartosci dodatnich i sumy = 1.
        eigenvector = np.abs(eigenvector)
        weights = eigenvector / eigenvector.sum()

        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
        ri = RANDOM_INDEX.get(n, RANDOM_INDEX[15])
        cr = ci / ri if ri > 0 else 0.0

        return weights, ci, cr, eigenvector

    @classmethod
    def compute_consistency(cls, pairwise_matrix: np.ndarray) -> dict:
        """Publiczny helper -- przydatny do UC PU2 (podglad CR przed obliczeniami)."""
        weights, ci, cr, eigenvector = cls._weights_from_pairwise_matrix(pairwise_matrix)
        return {
            "weights": weights,
            "eigenvector": eigenvector,
            "CI": ci,
            "CR": cr,
            "consistent": cr <= CR_THRESHOLD,
        }
