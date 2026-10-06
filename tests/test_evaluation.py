"""
Testy jednostkowe dla mcdm/evaluation/.
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies import TopsisStrategy, AhpStrategy
from mcdm.evaluation.rank_correlation import compare_rankings, compare_all_pairs
from mcdm.evaluation.rank_reversal import simulate_rank_reversal
from mcdm.evaluation.sensitivity import (
    weight_sensitivity_analysis,
    _perturb_and_renormalize,
)

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


def test_compare_rankings_partial_agreement_gives_intermediate_correlation():
    """Przypadek posredni (nie identyczny, nie w pelni odwrocony) --
    rankingi zgadzaja sie na pozycjach 1-2, a rozjezdzaja na 3-4
    (jedna zamieniona para). Wartosci policzone recznie:
    4 pary -> 6 porownan, 5 zgodnych i 1 niezgodna
    (konkordantne - niekonkordantne) / wszystkie = (5-1)/6 = 2/3.
    Spearman rho dla tej samej zamiany jednej sasiadujacej pary = 0.8.
    """
    ranking_a = [0, 1, 2, 3]
    ranking_b = [0, 1, 3, 2]
    result = compare_rankings(ranking_a, ranking_b)
    assert result["kendall_tau"] == pytest.approx(2 / 3)
    assert result["spearman_rho"] == pytest.approx(0.8)
    # Posrednia zgodnosc musi wypadac strywalnie pomiedzy identycznym
    # a w pelni odwrocony przypadkiem, nigdy na krawedziach [-1, 1].
    assert -1.0 < result["kendall_tau"] < 1.0
    assert -1.0 < result["spearman_rho"] < 1.0


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


def test_rank_reversal_genuinely_detects_reversal_for_ahp():
    """
    Dotychczasowy test (`test_rank_reversal_report_structure`) sprawdza
    tylko, ze `reversal_detected` jest typu bool -- nigdy przypadku, w
    ktorym faktycznie wynosi True. Ta macierz zostala znaleziona
    (przeszukiwaniem losowych wariantow) tak, by usuniecie slabszej
    alternatywy B realnie zmienilo wzajemna kolejnosc A i C w czolowce
    rankingu AHP -- to jest klasyczny, podrecznikowy przyklad rank
    reversal (Belton & Gear 1983 / Saaty'ego krytyka tego zjawiska).

    Baseline (wszystkie 4 alternatywy): D > A > C > B
    Po usunieciu B (top_k=2, A i C licza sie do czolowki):    D > C > A
    -- A i C zamienily sie miejscami, mimo ze B nie nalezala do
    scislej czolowki przed usunieciem.
    """
    problem = DecisionProblem(
        matrix=[[4.74, 8.01], [16.60, 2.84], [16.92, 2.83], [19.55, 9.90]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B", "C", "D"],
        criterion_names=["K1", "K2"],
    )
    strategy = AhpStrategy()

    report = simulate_rank_reversal(problem, strategy, alternative_name="B", top_k=2)

    assert report.baseline_order == ["D", "A", "C", "B"]
    assert report.modified_order == ["D", "C", "A"]
    assert report.reversal_detected is True


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


def test_perturb_and_renormalize_redistributes_evenly_when_others_are_zero():
    """
    Skrajny przypadek koncentracji wagi: jedno kryterium ma wage 1.0,
    a wszystkie pozostale 0.0 (others_sum <= 1e-12 w kodzie). Zwykle
    proporcjonalne przeskalowanie (scale = ...) nie zadziala, bo
    dzielenie przez others_sum=0 dalaby NaN -- kod ma dedykowana
    sciezke rownomiernego rozdzielenia nadwyzki/deficytu.

    Zmniejszenie wagi kryterium 0 o delta=-0.3 (z 1.0 do 0.7) musi
    rozdzielic uwolnione 0.3 po rowno na 2 pozostale kryteria (po 0.15),
    a nie zostawic ich na zero lub zglosic blad dzielenia przez zero.
    """
    weights = np.array([1.0, 0.0, 0.0])
    result = _perturb_and_renormalize(weights, criterion_idx=0, delta=-0.3)

    assert not np.any(np.isnan(result))
    np.testing.assert_allclose(result, [0.7, 0.15, 0.15], atol=1e-9)
    assert result.sum() == pytest.approx(1.0)


def test_perturb_and_renormalize_never_produces_negative_weights():
    """Nawet przy skrajnej, niemal pelnej koncentracji wagi na jednym
    kryterium (0.999999 / 0.0000005 / 0.0000005), duza ujemna delta
    (-0.5) nie moze sprowadzic zadnej wagi pod zero -- clip(..., 0.0, None)
    musi to zagwarantowac niezaleznie od tego, ktora sciezka
    (proporcjonalna czy rownomierna) zostala uzyta."""
    weights = np.array([0.999999, 0.0000005, 0.0000005])
    result = _perturb_and_renormalize(weights, criterion_idx=0, delta=-0.5)

    assert np.all(result >= 0.0)
    assert result.sum() == pytest.approx(1.0)
