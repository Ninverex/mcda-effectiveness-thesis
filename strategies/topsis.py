"""
Implementacja metody TOPSIS
(Technique for Order Preference by Similarity to Ideal Solution).

Kroki (zgodnie z 3.3 w spisie tresci pracy):
1. Normalizacja wektorowa macierzy decyzyjnej.
2. Wazenie znormalizowanej macierzy.
3. Wyznaczenie rozwiazania idealnego (PIS) i anty-idealnego (NIS)
   z uwzglednieniem kierunku optymalizacji kryterium (zysk/koszt).
4. Obliczenie odleglosci euklidesowych D+ i D- do PIS/NIS.
5. Wskaznik bliskosci C_i = D_i- / (D_i+ + D_i-).

Referencja: Hwang, C.L., Yoon, K. "Multiple attribute decision
making: an introduction.", 1995 -- pozycja z bibliografii pracy.
"""

from __future__ import annotations

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.numeric_utils import safe_denominator, safe_divide
from mcdm.strategies.base import ICalculationStrategy, RankingResult


class TopsisStrategy(ICalculationStrategy):

    name = "TOPSIS"

    def calculate_ranking(self, problem: DecisionProblem) -> RankingResult:
        X = problem.active_matrix          # (m, n)
        w = problem.weights                # (n,)
        directions = problem.directions    # lista "max"/"min"

        # 1. Normalizacja wektorowa: r_ij = x_ij / sqrt(sum_i x_ij^2)
        norm_denominator = np.sqrt(np.sum(X ** 2, axis=0))
        R = X / safe_denominator(norm_denominator)  # unik dzielenia przez 0

        # 2. Wazenie: v_ij = w_j * r_ij
        V = R * w

        # 3. Rozwiazanie idealne (PIS) i anty-idealne (NIS)
        pis = np.zeros(V.shape[1])
        nis = np.zeros(V.shape[1])
        for j, direction in enumerate(directions):
            if direction == "max":
                pis[j] = V[:, j].max()
                nis[j] = V[:, j].min()
            else:  # "min" -- kryterium kosztowe
                pis[j] = V[:, j].min()
                nis[j] = V[:, j].max()

        # 4. Odleglosci euklidesowe do PIS i NIS
        d_plus = np.sqrt(np.sum((V - pis) ** 2, axis=1))
        d_minus = np.sqrt(np.sum((V - nis) ** 2, axis=1))

        # 5. Wskaznik bliskosci wzgledem rozwiazania idealnego
        closeness = safe_divide(d_minus, d_plus + d_minus)

        ranking = list(np.argsort(-closeness))  # malejaco: najlepszy pierwszy

        return RankingResult(
            method_name=self.name,
            scores=closeness,
            ranking=ranking,
            alternative_names=problem.active_names,
            intermediate={
                "normalized_matrix": R,
                "weighted_matrix": V,
                "PIS": pis,
                "NIS": nis,
                "D_plus": d_plus,
                "D_minus": d_minus,
            },
        )
