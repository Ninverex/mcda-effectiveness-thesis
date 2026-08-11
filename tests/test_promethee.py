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
