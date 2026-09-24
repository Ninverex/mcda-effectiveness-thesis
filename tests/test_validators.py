"""
Testy jednostkowe dla mcdm/validation/validators.py -- pelne pokrycie
kazdej regoly walidacyjnej wymaganej przez WN1 ("System musi
weryfikowac poprawnosc danych wejsciowych i informowac o bledach").

Kazda galaz kodu w _validate_shapes, _validate_no_missing_values,
_validate_weights i _validate_thresholds ma tu dedykowany test --
istniejace testy w tests/test_topsis.py sprawdzaly walidacje jedynie
przy okazji (2 przypadki), co nie pokrywalo np. niezgodnosci
wymiarow nazw kryteriow, ujemnych wag czy progow ELECTRE/PROMETHEE.
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.validation.validators import ValidationError, validate_decision_problem


def make_problem(**overrides):
    defaults = dict(
        matrix=[[1, 2], [3, 4], [5, 6]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B", "C"],
        criterion_names=["K1", "K2"],
    )
    defaults.update(overrides)
    return DecisionProblem(**defaults)


def test_valid_problem_passes_without_error():
    problem = make_problem()
    validate_decision_problem(problem)  # nie powinno rzucic wyjatku


# ----------------------------------------------------------------------
# _validate_shapes
# ----------------------------------------------------------------------

def test_rejects_alternative_names_length_mismatch():
    problem = make_problem()
    problem.alternative_names = ["A", "B"]  # bylo 3, teraz 2
    with pytest.raises(ValidationError, match="nazw alternatyw"):
        validate_decision_problem(problem)


def test_rejects_criterion_names_length_mismatch():
    problem = make_problem()
    problem.criterion_names = ["K1"]  # bylo 2, teraz 1
    with pytest.raises(ValidationError, match="nazw kryteriow"):
        validate_decision_problem(problem)


def test_rejects_weights_length_mismatch():
    problem = make_problem()
    problem.weights = np.array([0.3, 0.3, 0.4])  # 3 wagi na 2 kryteria
    with pytest.raises(ValidationError, match="Liczba wag"):
        validate_decision_problem(problem)


def test_rejects_directions_length_mismatch():
    problem = make_problem()
    problem.directions = ["max", "max", "min"]  # 3 kierunki na 2 kryteria
    with pytest.raises(ValidationError, match="kierunkow optymalizacji"):
        validate_decision_problem(problem)


def test_rejects_active_mask_length_mismatch():
    problem = make_problem()
    problem.active_mask = np.array([True, True])  # 2 zamiast 3
    with pytest.raises(ValidationError, match="active_mask"):
        validate_decision_problem(problem)


def test_rejects_fewer_than_two_active_alternatives():
    problem = make_problem(active_mask=[True, False, False])
    with pytest.raises(ValidationError, match="co najmniej"):
        validate_decision_problem(problem)


def test_rejects_zero_active_alternatives():
    problem = make_problem(active_mask=[False, False, False])
    with pytest.raises(ValidationError, match="co najmniej"):
        validate_decision_problem(problem)


def test_accepts_exactly_two_active_alternatives():
    problem = make_problem(active_mask=[True, True, False])
    validate_decision_problem(problem)  # dokladnie 2 -- graniczny przypadek OK


# ----------------------------------------------------------------------
# _validate_no_missing_values
# ----------------------------------------------------------------------

def test_rejects_single_nan_value_with_cell_position():
    problem = make_problem(matrix=[[1, 2], [3, float("nan")], [5, 6]])
    with pytest.raises(ValidationError, match=r"wiersz=1, kolumna=1"):
        validate_decision_problem(problem)


def test_rejects_multiple_nan_values_listing_all_positions():
    problem = make_problem(
        matrix=[[float("nan"), 2], [3, float("nan")], [5, 6]]
    )
    with pytest.raises(ValidationError) as exc_info:
        validate_decision_problem(problem)
    message = str(exc_info.value)
    assert "wiersz=0, kolumna=0" in message
    assert "wiersz=1, kolumna=1" in message


def test_truncates_message_when_more_than_ten_missing_cells():
    # Macierz 6x2 = 12 komorek, wszystkie NaN -> przekracza limit 10
    matrix = [[float("nan")] * 2 for _ in range(6)]
    problem = make_problem(
        matrix=matrix,
        alternative_names=[f"Alt{i}" for i in range(6)],
        active_mask=[True] * 6,
    )
    with pytest.raises(ValidationError, match=r"i wiecej"):
        validate_decision_problem(problem)


def test_accepts_matrix_without_missing_values():
    problem = make_problem()
    validate_decision_problem(problem)  # brak NaN -- nie powinno rzucic


# ----------------------------------------------------------------------
# _validate_weights
# ----------------------------------------------------------------------

def test_rejects_negative_weight():
    problem = make_problem(weights=[-0.2, 1.2])
    with pytest.raises(ValidationError, match="ujemne"):
        validate_decision_problem(problem)


def test_rejects_weights_not_summing_to_one():
    problem = make_problem(weights=[0.3, 0.3])
    with pytest.raises(ValidationError, match=r"musi wynosic 1\.0"):
        validate_decision_problem(problem)


def test_error_message_reports_actual_weight_sum():
    problem = make_problem(weights=[0.2, 0.3])
    with pytest.raises(ValidationError, match=r"0\.500000"):
        validate_decision_problem(problem)


def test_accepts_weights_within_default_tolerance():
    # Suma = 1.0000005, mieszczy sie w domyslnej tolerancji 1e-6... a
    # raczej tuz poza nia -- sprawdzmy wartosc faktycznie w tolerancji.
    problem = make_problem(weights=[0.5, 0.5 + 1e-9])
    validate_decision_problem(problem)  # nie powinno rzucic


def test_custom_tolerance_is_respected():
    problem = make_problem(weights=[0.5, 0.52])  # suma = 1.02
    with pytest.raises(ValidationError):
        validate_decision_problem(problem, weight_tolerance=1e-6)
    validate_decision_problem(problem, weight_tolerance=0.05)  # teraz przechodzi


# ----------------------------------------------------------------------
# _validate_thresholds (tylko gdy require_thresholds=True)
# ----------------------------------------------------------------------

def test_thresholds_not_checked_by_default():
    problem = make_problem(thresholds=None)
    validate_decision_problem(problem)  # require_thresholds=False domyslnie


def test_rejects_missing_thresholds_when_required():
    problem = make_problem(thresholds=None)
    with pytest.raises(ValidationError, match="wymaga zdefiniowania"):
        validate_decision_problem(problem, require_thresholds=True)


def test_rejects_empty_thresholds_dict_when_required():
    problem = make_problem(thresholds={})
    with pytest.raises(ValidationError, match="wymaga zdefiniowania"):
        validate_decision_problem(problem, require_thresholds=True)


def test_rejects_threshold_list_with_wrong_length():
    problem = make_problem(
        thresholds={"q": [0.1], "p": [0.5, 0.5], "v": None}  # q ma 1 zamiast 2
    )
    with pytest.raises(ValidationError, match="Prog 'q'"):
        validate_decision_problem(problem, require_thresholds=True)


def test_accepts_complete_thresholds_with_correct_lengths():
    problem = make_problem(
        thresholds={"q": [0.1, 0.1], "p": [0.5, 0.5], "v": [1.0, 1.0]}
    )
    validate_decision_problem(problem, require_thresholds=True)


def test_accepts_none_value_for_individual_threshold():
    """v=None jest dopuszczalne (weto opcjonalne w niektorych wariantach ELECTRE)."""
    problem = make_problem(
        thresholds={"q": [0.1, 0.1], "p": [0.5, 0.5], "v": None}
    )
    validate_decision_problem(problem, require_thresholds=True)


def test_accepts_missing_threshold_key_entirely():
    """Brak klucza 'v' w slowniku (nie tylko None) rowniez jest dopuszczalny."""
    problem = make_problem(thresholds={"q": [0.1, 0.1], "p": [0.5, 0.5]})
    validate_decision_problem(problem, require_thresholds=True)
