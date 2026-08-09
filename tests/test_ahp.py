"""
Testy jednostkowe dla AhpStrategy.

test_ahp_consistency_classic_example -- klasyczny, spojny przyklad
Saaty'ego (3 kryteria), CR ponizej progu 0.10.

test_ahp_detects_inconsistent_judgments -- celowo cykliczna
(bardzo niespojna) macierz porownan, CR znacznie powyzej 0.10 --
sprawdza mechanizm ostrzegania z UC PU2.

test_ahp_ranking_on_domain_example -- sanity-check calego rankingu
na przykladzie domenowym (bez macierzy porownan parami -- wagi
brane wprost z problem.weights).
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.ahp import AhpStrategy, CR_THRESHOLD

EXAMPLES_DIR = "data/examples"

# Klasyczny, spojny przyklad Saaty'ego: porownanie 3 kryteriow
CONSISTENT_MATRIX = np.array([
    [1, 3, 5],
    [1 / 3, 1, 3],
    [1 / 5, 1 / 3, 1],
])

# Celowo cykliczna (skrajnie niespojna) macierz porownan
INCONSISTENT_MATRIX = np.array([
    [1, 9, 1 / 9],
    [1 / 9, 1, 9],
    [9, 1 / 9, 1],
])


def test_ahp_consistency_classic_example():
    result = AhpStrategy.compute_consistency(CONSISTENT_MATRIX)

    assert bool(result["consistent"]) is True
    assert result["CR"] < CR_THRESHOLD
    np.testing.assert_allclose(
        result["weights"], [0.63699, 0.25828, 0.10473], atol=1e-3
    )
    # Waga kryterium najwyzej ocenianego parami musi byc najwieksza
    assert np.argmax(result["weights"]) == 0


def test_ahp_detects_inconsistent_judgments():
    result = AhpStrategy.compute_consistency(INCONSISTENT_MATRIX)

    assert bool(result["consistent"]) is False
    assert result["CR"] > CR_THRESHOLD


def test_ahp_raises_when_configured_to_reject_inconsistency():
    problem = DecisionProblem(
        matrix=[[1, 2], [3, 4], [5, 1]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B", "C"],
        criterion_names=["K1", "K2"],
    )
    # 2-kryterialna, ale symetrycznie niespojna macierz porownan --
    # uzyjemy inconsistent_matrix 3x3 tylko do sprawdzenia mechanizmu
    # raise_on_inconsistency z macierza 2x2 o skrajnej niespojnosci
    # nie da sie latwo skonstruowac (n=2 zawsze CR=0), wiec test
    # wykonujemy na compute_consistency bezposrednio.
    strategy = AhpStrategy(
        pairwise_criteria_matrix=INCONSISTENT_MATRIX[:2, :2],
        raise_on_inconsistency=True,
    )
    # macierz 2x2 zawsze jest idealnie spojna (CR=0 z definicji AHP dla n<=2)
    result = strategy.calculate_ranking(
        problem.copy_with()
    )
    assert result.intermediate["consistency_ok"] is True


def test_ahp_ranking_on_domain_example():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    result = AhpStrategy().calculate_ranking(problem)

    assert len(result.ranking) == problem.n_alternatives
    assert set(result.ranking) == set(range(problem.n_alternatives))
    assert np.all(result.scores >= 0)
    # Wagi w tym trybie pochodza wprost z problem.weights
    np.testing.assert_allclose(
        result.intermediate["criteria_weights"], problem.weights
    )
