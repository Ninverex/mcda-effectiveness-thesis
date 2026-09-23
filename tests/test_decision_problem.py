"""
Testy jednostkowe dla mcdm/models/decision_problem.py -- warstwy
Model z architektury opisanej w pracy.

Pokrywa: konwersje typow w __post_init__, wlasciwosci pomocnicze
(active_matrix, active_names, n_alternatives, n_criteria), walidacje
konstruktora (kierunek optymalizacji), serializacje (from_json,
from_dict, to_dict) oraz copy_with (uzywane m.in. przez rank_reversal
i sensitivity -- musi byc nie-mutujace).
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem

EXAMPLES_DIR = "data/examples"


def make_problem(**overrides):
    defaults = dict(
        matrix=[[1, 7], [2, 4], [3, 1]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["Alt1", "Alt2", "Alt3"],
        criterion_names=["K1", "K2"],
    )
    defaults.update(overrides)
    return DecisionProblem(**defaults)


# ----------------------------------------------------------------------
# __post_init__ -- konwersje typow i walidacja kierunkow
# ----------------------------------------------------------------------

def test_matrix_and_weights_are_converted_to_float_ndarray():
    problem = make_problem(matrix=[[1, 2], [3, 4]], weights=[1, 1])
    assert isinstance(problem.matrix, np.ndarray)
    assert problem.matrix.dtype == np.float64
    assert isinstance(problem.weights, np.ndarray)
    assert problem.weights.dtype == np.float64


def test_active_mask_defaults_to_all_true_when_not_provided():
    problem = make_problem()
    assert problem.active_mask.dtype == bool
    np.testing.assert_array_equal(problem.active_mask, [True, True, True])


def test_active_mask_accepts_explicit_list():
    problem = make_problem(active_mask=[True, False, True])
    np.testing.assert_array_equal(problem.active_mask, [True, False, True])


def test_invalid_direction_raises_value_error():
    with pytest.raises(ValueError, match="Kierunek optymalizacji"):
        make_problem(directions=["max", "increasing"])


def test_valid_directions_do_not_raise():
    # Nie powinno rzucic wyjatku -- obie wartosci sa dozwolone
    make_problem(directions=["max", "min"])


# ----------------------------------------------------------------------
# Wlasciwosci pomocnicze
# ----------------------------------------------------------------------

def test_active_matrix_filters_by_active_mask():
    problem = make_problem(active_mask=[True, False, True])
    np.testing.assert_array_equal(problem.active_matrix, [[1, 7], [3, 1]])


def test_active_names_filters_by_active_mask():
    problem = make_problem(active_mask=[True, False, True])
    assert problem.active_names == ["Alt1", "Alt3"]


def test_n_alternatives_counts_only_active():
    problem = make_problem(active_mask=[True, False, True])
    assert problem.n_alternatives == 2
    # dlugosc pelnej listy nazw pozostaje niezmieniona
    assert len(problem.alternative_names) == 3


def test_n_criteria_matches_matrix_columns():
    problem = make_problem()
    assert problem.n_criteria == 2


def test_active_matrix_with_all_inactive_is_empty():
    problem = make_problem(active_mask=[False, False, False])
    assert problem.active_matrix.shape == (0, 2)
    assert problem.active_names == []
    assert problem.n_alternatives == 0


# ----------------------------------------------------------------------
# Serializacja: from_json / from_dict / to_dict
# ----------------------------------------------------------------------

def test_from_json_loads_example_file():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    assert problem.n_alternatives == 5
    assert problem.n_criteria == 4
    assert problem.alternative_names[0] == "Wariant_1_Grawitacyjny"


def test_from_dict_with_missing_optional_fields_uses_defaults():
    payload = {
        "matrix": [[1, 2], [3, 4]],
        "weights": [0.5, 0.5],
        "directions": ["max", "max"],
        "alternative_names": ["A", "B"],
        "criterion_names": ["K1", "K2"],
        # brak active_mask i thresholds -- powinny przyjac wartosci domyslne
    }
    problem = DecisionProblem.from_dict(payload)
    np.testing.assert_array_equal(problem.active_mask, [True, True])
    assert problem.thresholds is None


def test_to_dict_round_trip_preserves_data():
    original = make_problem(
        active_mask=[True, False, True],
        thresholds={"q": [0.1, 0.2], "p": [0.5, 0.5], "v": None},
    )
    restored = DecisionProblem.from_dict(original.to_dict())

    np.testing.assert_array_equal(restored.matrix, original.matrix)
    np.testing.assert_array_equal(restored.weights, original.weights)
    assert restored.directions == original.directions
    assert restored.alternative_names == original.alternative_names
    assert restored.criterion_names == original.criterion_names
    np.testing.assert_array_equal(restored.active_mask, original.active_mask)
    assert restored.thresholds == original.thresholds


def test_to_dict_returns_plain_json_serializable_types():
    """to_dict() musi zwracac typy natywne (list/float), nie numpy,
    zeby json.dumps() nie rzucal TypeError."""
    import json

    problem = make_problem()
    payload = problem.to_dict()
    json.dumps(payload)  # nie powinno rzucic wyjatku

    assert isinstance(payload["matrix"], list)
    assert isinstance(payload["weights"], list)
    assert isinstance(payload["active_mask"], list)
    assert isinstance(payload["active_mask"][0], bool)


# ----------------------------------------------------------------------
# copy_with -- nie-mutujaca modyfikacja
# ----------------------------------------------------------------------

def test_copy_with_overrides_weights_without_mutating_original():
    original = make_problem()
    modified = original.copy_with(weights=[0.9, 0.1])

    np.testing.assert_array_equal(modified.weights, [0.9, 0.1])
    # oryginal pozostaje niezmieniony
    np.testing.assert_array_equal(original.weights, [0.5, 0.5])


def test_copy_with_overrides_active_mask_without_mutating_original():
    original = make_problem()
    modified = original.copy_with(active_mask=[True, False, True])

    np.testing.assert_array_equal(modified.active_mask, [True, False, True])
    np.testing.assert_array_equal(original.active_mask, [True, True, True])
    assert modified.n_alternatives == 2
    assert original.n_alternatives == 3


def test_copy_with_preserves_unspecified_fields():
    original = make_problem(thresholds={"q": [0, 0], "p": [1, 1], "v": None})
    modified = original.copy_with(weights=[0.3, 0.7])

    assert modified.thresholds == original.thresholds
    assert modified.alternative_names == original.alternative_names
    assert modified.criterion_names == original.criterion_names


# ----------------------------------------------------------------------
# __repr__
# ----------------------------------------------------------------------

def test_repr_shows_active_over_total_alternatives_and_criteria_count():
    problem = make_problem(active_mask=[True, False, True])
    representation = repr(problem)

    assert "2/3" in representation  # aktywne/wszystkie alternatywy
    assert "criteria=2" in representation
