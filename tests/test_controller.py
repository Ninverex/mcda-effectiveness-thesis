"""
Testy jednostkowe dla mcdm/controller.py -- warstwy Controller z
architektury MVC opisanej w pracy. Sprawdza rejestr strategii,
dispatch po nazwie metody oraz integracje z warstwa walidacji (WN1).
"""

import pytest

from mcdm.controller import MCDMController
from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.base import RankingResult
from mcdm.strategies.topsis import TopsisStrategy
from mcdm.strategies.ahp import AhpStrategy
from mcdm.validation.validators import ValidationError


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
# Rejestr strategii
# ----------------------------------------------------------------------

def test_new_controller_has_no_available_methods():
    controller = MCDMController()
    assert controller.available_methods() == []


def test_register_strategy_adds_it_to_available_methods():
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    assert controller.available_methods() == ["TOPSIS"]


def test_register_multiple_strategies():
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    controller.register_strategy(AhpStrategy())
    assert set(controller.available_methods()) == {"TOPSIS", "AHP"}


def test_registering_same_name_twice_overwrites_previous():
    controller = MCDMController()
    first = TopsisStrategy()
    second = TopsisStrategy()
    controller.register_strategy(first)
    controller.register_strategy(second)

    assert controller.available_methods() == ["TOPSIS"]  # nie zduplikowane
    assert controller.get_strategy("TOPSIS") is second  # druga rejestracja wygrywa


def test_two_controller_instances_do_not_share_registry():
    """Regresja: rejestr strategii nie moze byc dzielonym stanem
    klasowym -- kazda instancja kontrolera musi miec wlasny."""
    controller_a = MCDMController()
    controller_b = MCDMController()
    controller_a.register_strategy(TopsisStrategy())

    assert controller_a.available_methods() == ["TOPSIS"]
    assert controller_b.available_methods() == []


# ----------------------------------------------------------------------
# get_strategy
# ----------------------------------------------------------------------

def test_get_strategy_returns_registered_instance():
    controller = MCDMController()
    strategy = TopsisStrategy()
    controller.register_strategy(strategy)
    assert controller.get_strategy("TOPSIS") is strategy


def test_get_strategy_returns_none_for_unknown_method():
    controller = MCDMController()
    assert controller.get_strategy("NIEISTNIEJACA") is None


# ----------------------------------------------------------------------
# run()
# ----------------------------------------------------------------------

def test_run_unknown_method_raises_value_error_listing_available():
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    problem = make_problem()

    with pytest.raises(ValueError, match="Nieznana metoda") as exc_info:
        controller.run(problem, "PROMETHEE")
    assert "TOPSIS" in str(exc_info.value)


def test_run_returns_ranking_result_from_correct_strategy():
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    problem = make_problem()

    result = controller.run(problem, "TOPSIS")

    assert isinstance(result, RankingResult)
    assert result.method_name == "TOPSIS"


def test_run_validates_problem_before_delegating_to_strategy():
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    invalid_problem = make_problem(weights=[0.3, 0.3])  # suma != 1.0

    with pytest.raises(ValidationError):
        controller.run(invalid_problem, "TOPSIS")


def test_run_propagates_require_thresholds_to_validation():
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    problem = make_problem(thresholds=None)

    # Bez wymogu progow: przechodzi
    controller.run(problem, "TOPSIS", require_thresholds=False)
    # Z wymogiem progow, ktorych brak: rzuca blad
    with pytest.raises(ValidationError, match="wymaga zdefiniowania"):
        controller.run(problem, "TOPSIS", require_thresholds=True)


# ----------------------------------------------------------------------
# run_all()
# ----------------------------------------------------------------------

def test_run_all_invokes_every_registered_strategy():
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    controller.register_strategy(AhpStrategy())
    problem = make_problem()

    results = controller.run_all(problem)

    assert set(results.keys()) == {"TOPSIS", "AHP"}
    assert all(isinstance(r, RankingResult) for r in results.values())


def test_run_all_on_empty_controller_returns_empty_dict():
    controller = MCDMController()
    problem = make_problem()
    assert controller.run_all(problem) == {}


def test_run_all_validates_once_per_strategy_and_fails_consistently():
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    controller.register_strategy(AhpStrategy())
    invalid_problem = make_problem(weights=[0.3, 0.3])

    with pytest.raises(ValidationError):
        controller.run_all(invalid_problem)
