"""
Testy jednostkowe dla mcdm/strategies/base.py -- RankingResult
(ujednolicony wynik wszystkich strategii) oraz ICalculationStrategy
(interfejs wzorca Strategia).
"""

import numpy as np
import pytest

from mcdm.strategies.base import ICalculationStrategy, RankingResult


def make_result(**overrides):
    defaults = dict(
        method_name="TEST",
        scores=np.array([0.9, 0.5, 0.7]),
        ranking=[0, 2, 1],
        alternative_names=["Alt1", "Alt2", "Alt3"],
    )
    defaults.update(overrides)
    return RankingResult(**defaults)


# ----------------------------------------------------------------------
# RankingResult.as_ordered_names / top
# ----------------------------------------------------------------------

def test_as_ordered_names_follows_ranking_order():
    result = make_result()
    assert result.as_ordered_names() == ["Alt1", "Alt3", "Alt2"]


def test_top_default_returns_single_best():
    result = make_result()
    assert result.top() == ["Alt1"]


def test_top_with_k_returns_first_k():
    result = make_result()
    assert result.top(2) == ["Alt1", "Alt3"]


def test_top_with_k_larger_than_ranking_returns_whole_ranking():
    result = make_result()
    assert result.top(10) == ["Alt1", "Alt3", "Alt2"]


def test_top_with_zero_returns_empty_list():
    result = make_result()
    assert result.top(0) == []


# ----------------------------------------------------------------------
# intermediate -- domyslna wartosc (field default_factory)
# ----------------------------------------------------------------------

def test_intermediate_defaults_to_empty_dict_when_not_provided():
    result = make_result()
    assert result.intermediate == {}


def test_intermediate_default_is_independent_per_instance():
    """Regresja: default_factory=dict musi tworzyc NOWY slownik za
    kazdym razem -- gdyby uzyto zmiennego domyslnego argumentu wprost
    (intermediate: dict = {}), wszystkie instancje dzielilyby ten sam
    obiekt i mutacja jednej wplywalaby na inne."""
    result_a = make_result()
    result_b = make_result()

    result_a.intermediate["CR"] = 0.05
    assert "CR" not in result_b.intermediate


def test_intermediate_can_be_provided_explicitly():
    result = make_result(intermediate={"CI": 0.02, "CR": 0.03})
    assert result.intermediate == {"CI": 0.02, "CR": 0.03}


# ----------------------------------------------------------------------
# __repr__
# ----------------------------------------------------------------------

def test_repr_shows_method_name_and_ordered_names():
    result = make_result()
    representation = repr(result)
    assert "TEST" in representation
    assert "Alt1 > Alt3 > Alt2" in representation


# ----------------------------------------------------------------------
# ICalculationStrategy -- interfejs abstrakcyjny
# ----------------------------------------------------------------------

def test_cannot_instantiate_abstract_strategy_directly():
    with pytest.raises(TypeError):
        ICalculationStrategy()


def test_subclass_without_calculate_ranking_cannot_be_instantiated():
    class IncompleteStrategy(ICalculationStrategy):
        pass  # nie implementuje calculate_ranking()

    with pytest.raises(TypeError):
        IncompleteStrategy()


def test_subclass_implementing_calculate_ranking_can_be_instantiated():
    class DummyStrategy(ICalculationStrategy):
        name = "DUMMY"

        def calculate_ranking(self, problem):
            return make_result(method_name=self.name)

    strategy = DummyStrategy()
    assert strategy.name == "DUMMY"
    result = strategy.calculate_ranking(problem=None)
    assert result.method_name == "DUMMY"
