"""
Testy jednostkowe dla mcdm/visualization/radar.py.
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.visualization.radar import normalized_profiles, plot_radar_chart
from mcdm.visualization._utils import figure_to_base64

EXAMPLES_DIR = "data/examples"


def test_normalized_profiles_are_in_unit_range():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    profiles = normalized_profiles(problem)

    assert profiles.shape == (problem.n_alternatives, problem.n_criteria)
    assert np.all(profiles >= 0.0)
    assert np.all(profiles <= 1.0)


def test_normalized_profiles_best_value_gets_one():
    """
    Dla kryterium typu 'max' (Przepustowosc), alternatywa z najwyzsza
    wartoscia w danych powinna dostac znormalizowana wartosc 1.0.
    Dla kryterium typu 'min' (Koszt_inwestycji), alternatywa z
    najnizsza wartoscia powinna dostac 1.0.
    """
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    profiles = normalized_profiles(problem)

    cost_idx = problem.criterion_names.index("Koszt_inwestycji")
    capacity_idx = problem.criterion_names.index("Przepustowosc")

    cheapest_alt = np.argmin(problem.active_matrix[:, cost_idx])
    highest_capacity_alt = np.argmax(problem.active_matrix[:, capacity_idx])

    assert profiles[cheapest_alt, cost_idx] == pytest.approx(1.0)
    assert profiles[highest_capacity_alt, capacity_idx] == pytest.approx(1.0)


def test_normalized_profiles_handles_zero_span():
    """Gdy wszystkie alternatywy maja identyczna wartosc kryterium,
    normalizacja nie powinna dzielic przez zero -- neutralna 0.5."""
    problem = DecisionProblem(
        matrix=[[5, 1], [5, 2], [5, 3]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B", "C"],
        criterion_names=["Stale", "Zmienne"],
    )
    profiles = normalized_profiles(problem)
    np.testing.assert_allclose(profiles[:, 0], [0.5, 0.5, 0.5])


def test_plot_radar_chart_returns_valid_figure():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    fig = plot_radar_chart(problem)

    assert fig is not None
    assert len(fig.axes) == 1

    # Sprawdza, ze figure da sie zserializowac do PNG/base64 (smoke test
    # calej sciezki uzywanej pozniej przez GUI)
    encoded = figure_to_base64(fig)
    assert isinstance(encoded, str)
    assert len(encoded) > 100


def test_plot_radar_chart_with_subset_of_alternatives():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    subset = problem.active_names[:2]
    fig = plot_radar_chart(problem, alternative_names=subset)

    ax = fig.axes[0]
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert set(legend_labels) == set(subset)

    figure_to_base64(fig)  # nie powinno rzucic wyjatku
