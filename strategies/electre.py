"""
Implementacja metody ELECTRE I (Roy, 1968 / 1990).

UWAGA: Swiadomie zaimplementowano wariant ELECTRE I, a NIE ELECTRE III.
Nie ma tu procedury destylacji (distillation procedure) ani
zmiennych progow preferencji/weta zaleznych od poziomu ocen --
uzywana jest klasyczna, "twarda" relacja przewyzszania oparta na
progu zgodnosci (concordance threshold) i progu niezgodnosci
(discordance / weto), zgodnie z pierwotnym sformulowaniem Roy'a.

Kroki (3.5 w spisie tresci, ograniczone do wariantu I):
1. Indeks zgodnosci c(a,b) -- suma wag kryteriow, na ktorych
   alternatywa a jest nie gorsza niz b (z uwzglednieniem progu
   indyferencji q, jesli podany).
2. Indeks niezgodnosci d(a,b) -- znormalizowana najwieksza roznica
   "na niekorzysc" a wsrod kryteriow, na ktorych b jest lepsze od a
   (z uwzglednieniem progu weta v, jesli podany -- przekroczenie v
   na ktoromkolwiek kryterium blokuje przewyzszanie niezaleznie od c).
3. Relacja przewyzszania: a S b (a przewyzsza b) gdy
   c(a,b) >= concordance_threshold ORAZ d(a,b) <= discordance_threshold
   (oraz brak przekroczenia progu weta na zadnym kryterium).
4. Ranking koncowy w tej implementacji wyznaczany jest przez tzw.
   "net outranking flow": liczba alternatyw przewyzszanych przez a
   minus liczba alternatyw przewyzszajacych a (prosta, przejrzysta
   heurystyka porzadkujaca relacje S w pelny ranking na potrzeby
   porownan z AHP/TOPSIS/PROMETHEE w rozdziale 4 -- oryginalne
   ELECTRE I nie generuje pelnego rankingu, tylko jadro (kernel),
   ktore jest rowniez zwracane w intermediate["kernel"]).

Referencja: Roy, B. "Wielokryterialne wspomaganie decyzji.", 1990;
Belton, V., Stewart, T. "Multiple criteria decision analysis...", 2012.
"""

from __future__ import annotations

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.base import ICalculationStrategy, RankingResult


class ElectreIStrategy(ICalculationStrategy):

    name = "ELECTRE I"

    def __init__(
        self,
        concordance_threshold: float = 0.65,
        discordance_threshold: float = 0.35,
    ):
        """
        Parameters
        ----------
        concordance_threshold : float
            Minimalny wymagany indeks zgodnosci c(a,b), by uznac,
            ze a przewyzsza b (typowo 0.6-0.75 w literaturze).
        discordance_threshold : float
            Maksymalny dopuszczalny indeks niezgodnosci d(a,b).
        """
        self.concordance_threshold = concordance_threshold
        self.discordance_threshold = discordance_threshold

    def calculate_ranking(self, problem: DecisionProblem) -> RankingResult:
        X = problem.active_matrix
        w = problem.weights
        directions = problem.directions
        m, n = X.shape

        thresholds = problem.thresholds or {}
        q_list = thresholds.get("q") or [0.0] * n
        v_list = thresholds.get("v") or [None] * n

        # Przeksztalcenie tak, by wieksza wartosc = lepsza dla kazdego kryterium
        G = np.zeros_like(X)
        for j, direction in enumerate(directions):
            G[:, j] = X[:, j] if direction == "max" else -X[:, j]

        # Rozpietosc kazdego kryterium -- do normalizacji niezgodnosci
        ranges = G.max(axis=0) - G.min(axis=0)
        ranges[ranges == 0] = 1e-12

        concordance = np.zeros((m, m))
        discordance = np.zeros((m, m))
        veto_triggered = np.zeros((m, m), dtype=bool)

        for a in range(m):
            for b in range(m):
                if a == b:
                    continue
                c_ab = 0.0
                worst_discordance = 0.0
                veto = False

                for j in range(n):
                    q = q_list[j] if q_list[j] is not None else 0.0
                    diff = G[a, j] - G[b, j]  # >0 gdy a lepsze niz b

                    if diff >= -q:
                        # a jest nie gorsze niz b (z tolerancja progu indyferencji)
                        c_ab += w[j]
                    else:
                        # b jest lepsze od a na tym kryterium -- wklad do niezgodnosci
                        local_discordance = (-diff) / ranges[j]
                        worst_discordance = max(worst_discordance, local_discordance)

                        v = v_list[j]
                        if v is not None and (-diff) > v:
                            veto = True

                concordance[a, b] = c_ab
                discordance[a, b] = worst_discordance
                veto_triggered[a, b] = veto

        outranking = (
            (concordance >= self.concordance_threshold)
            & (discordance <= self.discordance_threshold)
            & (~veto_triggered)
        )
        np.fill_diagonal(outranking, False)

        # Net outranking flow -- heurystyka porzadkujaca do pelnego rankingu
        outgoing = outranking.sum(axis=1)   # ile alternatyw a przewyzsza
        incoming = outranking.sum(axis=0)   # ile alternatyw przewyzsza a
        net_flow = outgoing - incoming

        ranking = list(np.argsort(-net_flow))

        kernel = self._compute_kernel(outranking)

        return RankingResult(
            method_name=self.name,
            scores=net_flow.astype(float),
            ranking=ranking,
            alternative_names=problem.active_names,
            intermediate={
                "concordance": concordance,
                "discordance": discordance,
                "veto_triggered": veto_triggered,
                "outranking": outranking,
                "outgoing_flow": outgoing,
                "incoming_flow": incoming,
                "kernel": kernel,
            },
        )

    @staticmethod
    def _compute_kernel(outranking: np.ndarray) -> list[int]:
        """
        Wyznacza jadro (kernel) relacji przewyzszania: podzbior
        alternatyw niezdominowanych, ktore razem "pokrywaja" wszystkie
        pozostale alternatywy relacja S. Uproszczona, zachlanna
        implementacja wystarczajaca dla typowych rozmiarow problemow
        MCDM w tej pracy (rzedu kilku-kilkunastu alternatyw).
        """
        m = outranking.shape[0]
        dominated = np.zeros(m, dtype=bool)
        for a in range(m):
            for b in range(m):
                if a != b and outranking[b, a]:
                    dominated[a] = True
        return [i for i in range(m) if not dominated[i]]
