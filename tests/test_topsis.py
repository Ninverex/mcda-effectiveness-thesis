"""
Testy jednostkowe dla TopsisStrategy.

test_topsis_hand_calculated_example -- porownuje wynik implementacji
z wartosciami wyliczonymi recznie (patrz komentarz w kodzie), co
stanowi niezalezna weryfikacje poprawnosci wzorow.

test_topsis_on_laptop_example -- sprawdza podstawowe wlasciwosci
(kompletnosc rankingu, zakres wskaznika C w [0,1]) na przykladzie
domenowym z data/examples.

test_topsis_rejects_invalid_weights -- sprawdza integracje z
warstwa walidacji (WN1).
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.topsis import TopsisStrategy
from mcdm.validation.validators import validate_decision_problem, ValidationError

EXAMPLES_DIR = "data/examples"


def make_hand_example() -> DecisionProblem:
    """
    3 alternatywy x 2 kryteria (oba typu 'zysk'), wagi rowne 0.5/0.5.

    Macierz:
        Alt1: [1, 7]
        Alt2: [2, 4]
        Alt3: [3, 1]

    Wartosci posrednie wyliczone recznie (zaokraglone do 5 miejsc):
        R (normalizacja wektorowa):
            Alt1: [0.26726, 0.86163]
            Alt2: [0.53452, 0.49236]
            Alt3: [0.80178, 0.12309]
        V = R * 0.5:
            Alt1: [0.13363, 0.43082]
            Alt2: [0.26726, 0.24618]
            Alt3: [0.40089, 0.06155]
        PIS = [0.40089, 0.43082]
        NIS = [0.13363, 0.06155]
        D+ = [0.26726, 0.22793, 0.36927]
        D- = [0.36927, 0.22791, 0.26726]
        C  = [0.58024, 0.49998, 0.41996]
    """
    return DecisionProblem(
        matrix=[[1, 7], [2, 4], [3, 1]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["Alt1", "Alt2", "Alt3"],
        criterion_names=["K1", "K2"],
    )


def test_topsis_hand_calculated_example():
    problem = make_hand_example()
    strategy = TopsisStrategy()
    result = strategy.calculate_ranking(problem)

    expected_closeness = np.array([0.58024, 0.49998, 0.41996])
    np.testing.assert_allclose(result.scores, expected_closeness, atol=1e-3)

    # Ranking oczekiwany: Alt1 > Alt2 > Alt3 (indeksy 0, 1, 2)
    assert result.ranking == [0, 1, 2]
    assert result.as_ordered_names() == ["Alt1", "Alt2", "Alt3"]


def test_topsis_closeness_in_valid_range():
    problem = make_hand_example()
    result = TopsisStrategy().calculate_ranking(problem)
    assert np.all(result.scores >= 0.0)
    assert np.all(result.scores <= 1.0)


def test_topsis_on_domain_example():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    validate_decision_problem(problem)
    result = TopsisStrategy().calculate_ranking(problem)

    assert len(result.ranking) == problem.n_alternatives
    assert set(result.ranking) == set(range(problem.n_alternatives))
    # Suma dlugosci rankingu zgadza sie z liczba alternatyw wejsciowych
    assert len(result.as_ordered_names()) == 5


def test_topsis_rejects_invalid_weights():
    problem = DecisionProblem(
        matrix=[[1, 2], [3, 4]],
        weights=[0.3, 0.3],  # suma != 1.0
        directions=["max", "max"],
        alternative_names=["A", "B"],
        criterion_names=["K1", "K2"],
    )
    with pytest.raises(ValidationError):
        validate_decision_problem(problem)


def test_topsis_rejects_missing_values():
    problem = DecisionProblem(
        matrix=[[1, float("nan")], [3, 4]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B"],
        criterion_names=["K1", "K2"],
    )
    with pytest.raises(ValidationError):
        validate_decision_problem(problem)
