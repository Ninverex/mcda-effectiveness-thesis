"""
Testy jednostkowe dla PrometheeStrategy.

test_promethee_symmetric_case -- przy rownych wagach i skrajnie
"lustrzanych" ocenach (kazda alternatywa wygrywa dokladnie jedno
kryterium) przeplywy netto powinny sie zerowac -- dobra weryfikacja
poprawnosci macierzy preferencji.

test_promethee_hand_calculated_example -- przyklad z niesymetrycznymi
wagami, gdzie wynik dominacji jest jednoznaczny i policzalny recznie
(patrz komentarz w kodzie).

test_promethee_i_partial_ranking -- sprawdza, ze PROMETHEE I poprawnie
wykrywa nieporownywalnosc w przypadku symetrycznym.
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.promethee import PrometheeStrategy

EXAMPLES_DIR = "data/examples"


def make_example(weights):
    return DecisionProblem(
        matrix=[[1, 7], [2, 4], [3, 1]],
        weights=weights,
        directions=["max", "max"],
        alternative_names=["Alt1", "Alt2", "Alt3"],
        criterion_names=["K1", "K2"],
        thresholds={"q": [0, 0], "p": [None, None]},
    )


def test_promethee_symmetric_case():
    """
    Wagi rowne 0.5/0.5. Kazda para alternatyw wygrywa dokladnie
    jedno z dwoch kryteriow wzgledem drugiej -> przeplywy netto = 0
    dla kazdej alternatywy.
    """
    problem = make_example([0.5, 0.5])
    result = PrometheeStrategy().calculate_ranking(problem)

    np.testing.assert_allclose(result.scores, [0.0, 0.0, 0.0], atol=1e-9)
    # Phi+ musi rownowazyc Phi- w tym symetrycznym przypadku
    np.testing.assert_allclose(
        result.intermediate["phi_plus"], result.intermediate["phi_minus"]
    )


def test_promethee_hand_calculated_example():
    """
    Wagi [0.7, 0.3] (K1 dominuje). Wartosci obliczone recznie:
        Phi+ = [0.3, 0.5, 0.7]
        Phi- = [0.7, 0.5, 0.3]
        Phi_net = [-0.4, 0.0, 0.4]
    Oczekiwany ranking: Alt3 > Alt2 > Alt1 (bo Alt3 dominuje w K1,
    ktore ma wieksza wage).
    """
    problem = make_example([0.7, 0.3])
    result = PrometheeStrategy().calculate_ranking(problem)

    np.testing.assert_allclose(
        result.intermediate["phi_plus"], [0.3, 0.5, 0.7], atol=1e-9
    )
    np.testing.assert_allclose(
        result.intermediate["phi_minus"], [0.7, 0.5, 0.3], atol=1e-9
    )
    np.testing.assert_allclose(result.scores, [-0.4, 0.0, 0.4], atol=1e-9)
    assert result.as_ordered_names() == ["Alt3", "Alt2", "Alt1"]


def test_promethee_i_partial_ranking_detects_incomparability():
    problem = make_example([0.5, 0.5])
    result = PrometheeStrategy().calculate_ranking(problem)

    incomparable = result.intermediate["promethee_i"]["incomparable"]
    # W przypadku symetrycznym kazda para powinna byc nieporownywalna
    # (zaden z Phi+/Phi- warunkow outrankingu nie jest spelniony)
    off_diagonal = incomparable[~np.eye(3, dtype=bool)]
    assert np.all(off_diagonal)


def test_promethee_on_domain_example_returns_full_ranking():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    result = PrometheeStrategy().calculate_ranking(problem)

    assert len(result.ranking) == problem.n_alternatives
    assert set(result.ranking) == set(range(problem.n_alternatives))
    # phi_plus - phi_minus musi odpowiadac scores (Phi_net)
    np.testing.assert_allclose(
        result.intermediate["phi_plus"] - result.intermediate["phi_minus"],
        result.scores,
    )


def test_promethee_handles_zero_variance_criterion_without_error():
    """Kryterium, na ktorym wszystkie alternatywy maja identyczna
    wartosc (span=0, wiec domyslny prog p byloby zerowe) -- kod ma
    zabezpieczenie (p = max(..., q + 1e-9)). Takie kryterium nie
    powinno roznicowac alternatyw."""
    problem = DecisionProblem(
        matrix=[[5, 1], [5, 4], [5, 9]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B", "C"],
        criterion_names=["K1_stale", "K2"],
    )
    result = PrometheeStrategy().calculate_ranking(problem)
    assert not np.any(np.isnan(result.scores))
    assert not np.any(np.isinf(result.scores))
    assert result.as_ordered_names()[0] == "C"


def test_promethee_minimal_two_alternatives():
    """Przypadek brzegowy m=2 (minimalna liczba alternatyw wg WN1) --
    dzielenie przez (m-1)=1 w przeplywach nie powinno sprawiac problemow."""
    problem = DecisionProblem(
        matrix=[[1, 5], [3, 2]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B"],
        criterion_names=["K1", "K2"],
    )
    result = PrometheeStrategy().calculate_ranking(problem)
    assert len(result.ranking) == 2
    assert not np.any(np.isnan(result.scores))
    assert result.scores[0] == pytest.approx(-result.scores[1])


def test_unicriterion_net_flows_has_correct_shape():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    strategy = PrometheeStrategy()
    flows = strategy.unicriterion_net_flows(problem)
    assert flows.shape == (problem.n_alternatives, problem.n_criteria)
    assert not np.any(np.isnan(flows))


def test_unicriterion_net_flows_values_are_bounded():
    """Kazdy jednokryterialny przeplyw netto musi miescic sie w [-1, 1]
    (to srednia roznic preferencji, ktore same sa w [-1, 1])."""
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    flows = PrometheeStrategy().unicriterion_net_flows(problem)
    assert np.all(flows >= -1.0)
    assert np.all(flows <= 1.0)


def test_unicriterion_net_flows_weighted_sum_matches_phi_net():
    """Zagregowany Phi_net z calculate_ranking() powinien odpowiadac
    wazonej sumie jednokryterialnych przeplywow z unicriterion_net_flows()
    -- to dokladnie definiuje relacje miedzy PROMETHEE a GAIA."""
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    strategy = PrometheeStrategy()
    result = strategy.calculate_ranking(problem)
    flows = strategy.unicriterion_net_flows(problem)
    reconstructed_phi_net = flows @ problem.weights
    np.testing.assert_allclose(reconstructed_phi_net, result.scores, atol=1e-9)
