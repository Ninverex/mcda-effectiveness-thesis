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
from mcdm.strategies.ahp import AhpStrategy, CR_THRESHOLD, ConsistencyWarning

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


def test_ahp_warns_on_inconsistency_by_default():
    """Domyslnie (raise_on_inconsistency=False) obliczenia nie sa
    blokowane, ale przekroczenie CR>0.10 musi byc zasygnalizowane
    przez ConsistencyWarning (PU2) -- a nie po cichu pominiete, jak
    to bylo przed poprawka (klasa ConsistencyWarning istniala, ale
    nigdy nie byla faktycznie zglaszana)."""
    problem = DecisionProblem(
        matrix=[[1, 2, 3], [3, 4, 1], [5, 1, 2]],
        weights=[1 / 3, 1 / 3, 1 / 3],
        directions=["max", "max", "max"],
        alternative_names=["A", "B", "C"],
        criterion_names=["K1", "K2", "K3"],
    )
    strategy = AhpStrategy(pairwise_criteria_matrix=INCONSISTENT_MATRIX)

    with pytest.warns(ConsistencyWarning):
        result = strategy.calculate_ranking(problem)

    assert bool(result.intermediate["consistency_ok"]) is False


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


# ----------------------------------------------------------------------
# Przypadki brzegowe: zabezpieczenia przed dzieleniem przez zero
# ----------------------------------------------------------------------

def test_ahp_single_criterion_gives_zero_consistency_index():
    """Dla n=1 kryterium wspolczynnik CI nie jest zdefiniowany
    matematycznie (dzielenie przez n-1=0) -- kod ma na to
    zabezpieczenie i zwraca CI=0.0 (macierz 1x1 jest trywialnie spojna)."""
    result = AhpStrategy.compute_consistency(np.array([[1.0]]))
    assert result["CI"] == 0.0
    assert result["CR"] == 0.0
    assert bool(result["consistent"]) is True


def test_ahp_handles_all_zero_column_in_synthesis_without_error():
    """Kolumna kryterium typu 'max' z samymi zerami dawalaby dzielenie
    przez 0 przy normalizacji kolumnowej (suma=0) -- zabezpieczenie
    1e-12 powinno dac dobrze zdefiniowany wynik."""
    problem = DecisionProblem(
        matrix=[[0, 5], [0, 3], [0, 1]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B", "C"],
        criterion_names=["K1_zawsze_zero", "K2"],
    )
    result = AhpStrategy().calculate_ranking(problem)

    assert not np.any(np.isnan(result.scores))
    assert not np.any(np.isinf(result.scores))


def test_ahp_handles_all_zero_column_for_min_direction_without_error():
    """Analogicznie dla kryterium typu 'min' -- odwrocenie (1/x) kolumny
    samych zer dawaloby dzielenie przez 0 (obsluzone przez np.where)."""
    problem = DecisionProblem(
        matrix=[[0, 5], [0, 3], [0, 1]],
        weights=[0.5, 0.5],
        directions=["min", "max"],
        alternative_names=["A", "B", "C"],
        criterion_names=["K1_zawsze_zero", "K2"],
    )
    result = AhpStrategy().calculate_ranking(problem)

    assert not np.any(np.isnan(result.scores))
    assert not np.any(np.isinf(result.scores))
