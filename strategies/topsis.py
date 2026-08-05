"""
Implementacja metody TOPSIS
(Technique for Order Preference by Similarity to Ideal Solution).

Kroki:
1. Normalizacja wektorowa macierzy decyzyjnej.
2. Wazenie znormalizowanej macierzy.
3. Wyznaczenie rozwiazania idealnego (PIS) i anty-idealnego (NIS)
   z uwzglednieniem kierunku optymalizacji kryterium (zysk/koszt).
4. Obliczenie odleglosci euklidesowych D+ i D- do PIS/NIS.
5. Wskaznik bliskosci C_i = D_i- / (D_i+ + D_i-).

"""

from __future__ import annotations

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.base import ICalculationStrategy, RankingResult


class TopsisStrategy(ICalculationStrategy):

    name = "TOPSIS"

    def calculate_ranking(self, problem: DecisionProblem) -> RankingResult:
        X = problem.active_matrix          # (m, n)
        w = problem.weights                # (n,)
        directions = problem.directions    # lista "max"/"min"

        # 1. Normalizacja wektorowa: r_ij = x_ij / sqrt(sum_i x_ij^2)
        norm_denominator = np.sqrt(np.sum(X ** 2, axis=0))
        norm_denominator[norm_denominator == 0] = 1e-12  # unik dzielenia przez 0
        R = X / norm_denominator

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
        denom = d_plus + d_minus
        denom[denom == 0] = 1e-12
        closeness = d_minus / denom

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
