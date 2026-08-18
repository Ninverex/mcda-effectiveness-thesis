"""
Testy jednostkowe dla mcdm/evaluation/.
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies import TopsisStrategy, AhpStrategy
from mcdm.evaluation.rank_correlation import compare_rankings, compare_all_pairs
from mcdm.evaluation.rank_reversal import simulate_rank_reversal
from mcdm.evaluation.sensitivity import weight_sensitivity_analysis

EXAMPLES_DIR = "data/examples"


# ----------------------------------------------------------------------
# rank_correlation
# ----------------------------------------------------------------------

def test_compare_rankings_identical_gives_perfect_correlation():
    ranking = [2, 0, 1, 3]
    result = compare_rankings(ranking, ranking)
    assert result["kendall_tau"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)


def test_compare_rankings_reversed_gives_negative_correlation():
    ranking_a = [0, 1, 2, 3]
    ranking_b = [3, 2, 1, 0]
    result = compare_rankings(ranking_a, ranking_b)
    assert result["kendall_tau"] == pytest.approx(-1.0)
    assert result["spearman_rho"] == pytest.approx(-1.0)


def test_compare_rankings_rejects_mismatched_alternative_sets():
    with pytest.raises(ValueError):
        compare_rankings([0, 1, 2], [0, 1, 3])


def test_compare_all_pairs_on_domain_example():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    results = {
        "TOPSIS": TopsisStrategy().calculate_ranking(problem),
        "AHP": AhpStrategy().calculate_ranking(problem),
    }
    pairs = compare_all_pairs(results)
    assert ("TOPSIS", "AHP") in pairs
    assert -1.0 <= pairs[("TOPSIS", "AHP")]["kendall_tau"] <= 1.0


# ----------------------------------------------------------------------
# rank_reversal
# ----------------------------------------------------------------------

def test_rank_reversal_report_structure():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    strategy = TopsisStrategy()
    report = simulate_rank_reversal(
        problem, strategy, alternative_name="Wariant_2_Modulowy", top_k=3
    )

    assert report.method_name == "TOPSIS"
    assert report.removed_alternative == "Wariant_2_Modulowy"
    assert "Wariant_2_Modulowy" not in report.modified_order
    assert len(report.modified_order) == len(report.baseline_order) - 1
    # Usunieta alternatywa musi miec pozycje "po" = None
    assert report.position_changes["Wariant_2_Modulowy"][1] is None
    assert isinstance(report.reversal_detected, bool)


def test_rank_reversal_unknown_alternative_raises():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    with pytest.raises(ValueError):
        simulate_rank_reversal(problem, TopsisStrategy(), "Nieistniejacy_Wariant")


def test_rank_reversal_does_not_mutate_original_problem():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    original_mask = problem.active_mask.copy()
    simulate_rank_reversal(problem, TopsisStrategy(), "Wariant_2_Modulowy")
    np.testing.assert_array_equal(problem.active_mask, original_mask)


# ----------------------------------------------------------------------
# sensitivity
# ----------------------------------------------------------------------

def test_sensitivity_weights_always_sum_to_one():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    report = weight_sensitivity_analysis(
        problem, TopsisStrategy(), criterion_name="Koszt_inwestycji"
    )
    for point in report.points:
        assert point.weights.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(point.weights >= 0)


def test_sensitivity_zero_delta_matches_baseline_ranking():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    strategy = TopsisStrategy()
    baseline = strategy.calculate_ranking(problem)

    report = weight_sensitivity_analysis(
        problem, strategy, criterion_name="Koszt_inwestycji",
        delta_range=np.array([0.0]),
    )
    assert report.points[0].ranking_names == baseline.as_ordered_names()


def test_sensitivity_unknown_criterion_raises():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    with pytest.raises(ValueError):
        weight_sensitivity_analysis(problem, TopsisStrategy(), "Nieistniejace_Kryterium")


def test_sensitivity_report_stability_check():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    report = weight_sensitivity_analysis(
        problem, TopsisStrategy(), criterion_name="Koszt_inwestycji",
        delta_range=np.array([0.0, 0.0, 0.0]),
    )
    # Przy braku faktycznej zmiany delty lider musi byc stabilny
    assert report.is_stable() is True
