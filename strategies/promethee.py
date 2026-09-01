"""
Implementacja metody PROMETHEE (Preference Ranking Organization
METHod for Enrichment Evaluations), Brans i Vincke 1985.

Kroki (3.4 w spisie tresci):
1. Funkcja preferencji P_j(a,b) per kryterium -- w tej implementacji
   uzyty jest typ V ("linear with indifference and preference
   thresholds", typ 5 wg klasyfikacji Bransa), definiowany progami
   q (indyferencji) i p (preferencji) z problem.thresholds.
   Jesli progi nie sa podane dla danego kryterium, uzywany jest typ
   III (liniowy bez progu indyferencji, q=0) ze skala roznic
   danego kryterium jako p.
2. Macierz preferencji parami P_j(a,b) dla kazdego kryterium.
3. Zagregowany indeks preferencji pi(a,b) = sum_j w_j * P_j(a,b).
4. Przeplywy: Phi+(a) (preferencja wychodzaca), Phi-(a) (wchodzaca),
   Phi(a) = Phi+(a) - Phi-(a) -- PROMETHEE II (ranking pelny).
5. PROMETHEE I (ranking czesciowy, z mozliwa nieporownywalnoscia)
   dostepny jako dodatkowy wynik w polu intermediate["promethee_i"].

Referencja: Behzadian i in. "PROMETHEE: A comprehensive literature
review...", 2010 -- pozycja z bibliografii pracy.
"""

from __future__ import annotations

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.base import ICalculationStrategy, RankingResult


def _linear_preference(diff: np.ndarray, q: float, p: float) -> np.ndarray:
    """
    Funkcja preferencji typu V (liniowa z progami q i p).

    diff : d(a,b) = g(a) - g(b) dla kryterium typu 'zysk' (juz
           przeksztalconego tak, by wieksza wartosc byla lepsza).

    P(a,b) = 0                          gdy d <= q
           = (d - q) / (p - q)          gdy q < d < p
           = 1                          gdy d >= p
    """
    p = max(p, q + 1e-12)  # unik dzielenia przez 0 gdy p == q
    pref = np.zeros_like(diff)
    linear_zone = (diff > q) & (diff < p)
    pref[linear_zone] = (diff[linear_zone] - q) / (p - q)
    pref[diff >= p] = 1.0
    pref[diff <= q] = 0.0
    return pref


class PrometheeStrategy(ICalculationStrategy):

    name = "PROMETHEE"

    def __init__(self, default_preference_fraction: float = 0.5):
        """
        Parameters
        ----------
        default_preference_fraction : float
            Jesli problem.thresholds nie definiuje progu p dla danego
            kryterium, prog p jest szacowany jako
            `default_preference_fraction * (max - min)` roznic
            wartosci tego kryterium w danych (typowa heurystyka przy
            braku wiedzy eksperckiej o progach).
        """
        self.default_preference_fraction = default_preference_fraction

    def calculate_ranking(self, problem: DecisionProblem) -> RankingResult:
        X = problem.active_matrix
        w = problem.weights
        directions = problem.directions
        m, n = X.shape
        thresholds = problem.thresholds or {}

        # Macierz zagregowanych indeksow preferencji pi(a,b)
        pi = np.zeros((m, m))

        for j in range(n):
            pref_matrix = self._criterion_preference_matrix(X, directions, j, thresholds)
            pi += w[j] * pref_matrix

        np.fill_diagonal(pi, 0.0)

        # Przeplywy PROMETHEE
        phi_plus = pi.sum(axis=1) / (m - 1)
        phi_minus = pi.sum(axis=0) / (m - 1)
        phi_net = phi_plus - phi_minus

        ranking = list(np.argsort(-phi_net))  # PROMETHEE II -- ranking pelny

        promethee_i = self._partial_ranking(phi_plus, phi_minus)

        return RankingResult(
            method_name=self.name,
            scores=phi_net,
            ranking=ranking,
            alternative_names=problem.active_names,
            intermediate={
                "preference_index_matrix": pi,
                "phi_plus": phi_plus,
                "phi_minus": phi_minus,
                "phi_net": phi_net,
                "promethee_i": promethee_i,
            },
        )

    def unicriterion_net_flows(self, problem: DecisionProblem) -> np.ndarray:
        """
        Zwraca macierz (m x n) jednokryterialnych przeplywow netto:
        phi_j(a) = (1/(m-1)) * sum_b [P_j(a,b) - P_j(b,a)] -- przeplyw
        netto policzony OSOBNO dla kazdego kryterium, bez wazenia.

        To jest dokladnie macierz wejsciowa do geometrycznej analizy
        GAIA (Brans, Mareschal): kazdy wiersz to "profil" alternatywy
        w przestrzeni jednokryterialnych przeplywow, kazda kolumna to
        os odpowiadajaca jednemu kryterium. Rzutowana przez PCA do 2D
        (patrz mcdm/visualization/gaia.py) pozwala zobaczyc konflikty
        miedzy kryteriami i pozycje alternatyw wzgledem nich.
        """
        X = problem.active_matrix
        directions = problem.directions
        m, n = X.shape
        thresholds = problem.thresholds or {}

        flows = np.zeros((m, n))
        for j in range(n):
            pref_matrix = self._criterion_preference_matrix(X, directions, j, thresholds)
            np.fill_diagonal(pref_matrix, 0.0)
            flows[:, j] = (pref_matrix.sum(axis=1) - pref_matrix.sum(axis=0)) / (m - 1)
        return flows

    def _criterion_preference_matrix(
        self,
        X: np.ndarray,
        directions: list[str],
        j: int,
        thresholds: dict,
    ) -> np.ndarray:
        """
        Wylicza macierz preferencji P_j(a,b) dla pojedynczego
        kryterium j, uwzgledniajac jego kierunek optymalizacji oraz
        progi q/p (wspoldzielone przez calculate_ranking() i
        unicriterion_net_flows(), zeby nie duplikowac logiki progow).
        """
        col = X[:, j]
        values = col if directions[j] == "max" else -col

        # Macierz roznic d(a,b) = g(a) - g(b), ksztalt (m, m)
        diff = values.reshape(-1, 1) - values.reshape(1, -1)

        q, p = self._resolve_thresholds(values, j, thresholds)
        return _linear_preference(diff, q=q, p=p)

    def _resolve_thresholds(
        self, values: np.ndarray, j: int, thresholds: dict
    ) -> tuple[float, float]:
        """Zwraca (q, p) dla kryterium j: z problem.thresholds jesli podane,
        w przeciwnym razie q=0 i p szacowane jako ulamek rozstepu wartosci."""
        q_list = thresholds.get("q") or [0.0] * len(values)
        p_list = thresholds.get("p") or [None] * len(values)

        q = q_list[j] if q_list[j] is not None else 0.0
        p = p_list[j]
        if p is None:
            span = values.max() - values.min()
            p = max(span * self.default_preference_fraction, q + 1e-9)
        return q, p

    @staticmethod
    def _partial_ranking(phi_plus: np.ndarray, phi_minus: np.ndarray) -> dict:
        """
        PROMETHEE I: relacja czesciowa.
        a przewyzsza b (a P b) gdy Phi+(a) >= Phi+(b) i Phi-(a) <= Phi-(b),
        z co najmniej jedna nierownoscia ostra.
        a jest nieporownywalne z b (a R b) w przeciwnym przypadku,
        gdy zadna z alternatyw nie przewyzsza drugiej.
        """
        m = len(phi_plus)
        outranks = np.zeros((m, m), dtype=bool)
        incomparable = np.zeros((m, m), dtype=bool)

        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                ge_plus = phi_plus[a] >= phi_plus[b]
                le_minus = phi_minus[a] <= phi_minus[b]
                strict = (phi_plus[a] > phi_plus[b]) or (phi_minus[a] < phi_minus[b])
                if ge_plus and le_minus and strict:
                    outranks[a, b] = True

        for a in range(m):
            for b in range(a + 1, m):
                if not outranks[a, b] and not outranks[b, a]:
                    incomparable[a, b] = incomparable[b, a] = True

        return {"outranks": outranks, "incomparable": incomparable}
