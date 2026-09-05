"""
Testy jednostkowe dla mcdm/visualization/gaia.py.
"""

import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.promethee import PrometheeStrategy
from mcdm.visualization.gaia import plot_gaia_plane
from mcdm.visualization._utils import figure_to_base64

EXAMPLES_DIR = "data/examples"


def test_plot_gaia_plane_returns_valid_figure():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    fig = plot_gaia_plane(problem)

    assert fig is not None
    assert len(fig.axes) == 1

    encoded = figure_to_base64(fig)
    assert isinstance(encoded, str)
    assert len(encoded) > 100


def test_plot_gaia_plane_rejects_fewer_than_three_criteria():
    problem = DecisionProblem(
        matrix=[[1, 7], [2, 4], [3, 1]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["Alt1", "Alt2", "Alt3"],
        criterion_names=["K1", "K2"],
    )
    with pytest.raises(ValueError):
        plot_gaia_plane(problem)


def test_plot_gaia_plane_accepts_custom_promethee_strategy():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    strategy = PrometheeStrategy(default_preference_fraction=0.3)
    fig = plot_gaia_plane(problem, promethee_strategy=strategy)
    assert fig is not None
    figure_to_base64(fig)  # nie powinno rzucic wyjatku


def test_plot_gaia_plane_places_all_alternatives():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    fig = plot_gaia_plane(problem)
    ax = fig.axes[0]

    # Jedna adnotacja tekstowa per alternatywa + jedna per kryterium
    annotation_texts = {a.get_text() for a in ax.texts}
    for name in problem.active_names:
        assert name in annotation_texts
    for name in problem.criterion_names:
        assert name in annotation_texts
