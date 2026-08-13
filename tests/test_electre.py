"""
Testy jednostkowe dla ElectreIStrategy.

Uwaga: testowany jest wariant ELECTRE I (bez procedury destylacji
znanej z ELECTRE III) -- zgodnie z zakresem przyjetym w tej pracy.

test_electre_hand_calculated_example -- 3 alternatywy x 2 kryteria,
wartosci concordance/discordance wyliczone i zweryfikowane recznie
(patrz komentarz w kodzie).

test_electre_veto_blocks_outranking -- sprawdza, ze przekroczenie
progu weta blokuje przewyzszanie nawet przy wysokiej zgodnosci.

test_electre_kernel_is_subset_of_alternatives -- podstawowa
wlasciwosc jadra relacji przewyzszania.
"""

import numpy as np

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.electre import ElectreIStrategy

EXAMPLES_DIR = "data/examples"


def make_example():
    return DecisionProblem(
        matrix=[[5, 1], [4, 4], [1, 5]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["Alt1", "Alt2", "Alt3"],
        criterion_names=["K1", "K2"],
        thresholds={"q": [0, 0], "v": [None, None]},
    )


def test_electre_hand_calculated_example():
    """
    Recznie zweryfikowane wartosci:
        concordance(Alt1,Alt2) = 0.5  (Alt1 wygrywa K1, przegrywa K2)
        discordance(Alt1,Alt2) = 0.75 (roznica na K2: |1-4|/range(4) = 3/4)
    Alternatywa zrownowazona (Alt2) powinna przewyzszac obie skrajne
    (Alt1, Alt3), poniewaz jej niezgodnosc wzgledem nich jest niska
    (0.25), podczas gdy skrajne alternatywy maja wobec siebie
    dyskordancje =1.0 (calkowita przewaga jednego kryterium).
    """
    problem = make_example()
    strategy = ElectreIStrategy(concordance_threshold=0.5, discordance_threshold=0.5)
    result = strategy.calculate_ranking(problem)

    concordance = result.intermediate["concordance"]
    discordance = result.intermediate["discordance"]

    np.testing.assert_allclose(concordance[0, 1], 0.5, atol=1e-9)
    np.testing.assert_allclose(discordance[0, 1], 0.75, atol=1e-9)
    np.testing.assert_allclose(discordance[1, 0], 0.25, atol=1e-9)
    np.testing.assert_allclose(discordance[0, 2], 1.0, atol=1e-9)

    outranking = result.intermediate["outranking"]
    # Alt2 przewyzsza obie skrajne alternatywy
    assert outranking[1, 0] == True
    assert outranking[1, 2] == True
    # Skrajne alternatywy nie przewyzszaja niczego przy tych progach
    assert not outranking[0, :].any()
    assert not outranking[2, :].any()

    assert result.as_ordered_names()[0] == "Alt2"
    assert result.intermediate["kernel"] == [1]


def test_electre_veto_blocks_outranking():
    """
    Nawet przy wysokiej zgodnosci, przekroczenie progu weta na
    pojedynczym kryterium musi zablokowac przewyzszanie.
    """
    problem = DecisionProblem(
        matrix=[[10, 1], [9, 9]],
        weights=[0.9, 0.1],
        directions=["max", "max"],
        alternative_names=["A", "B"],
        criterion_names=["K1", "K2"],
        thresholds={"q": [0, 0], "v": [None, 5]},  # weto na K2: roznica > 5
    )
    strategy = ElectreIStrategy(concordance_threshold=0.5, discordance_threshold=1.0)
    result = strategy.calculate_ranking(problem)

    # A ma bardzo wysoka zgodnosc wzgledem B (wygrywa K1 z waga 0.9),
    # ale roznica na K2 (9-1=8) przekracza prog weta v=5 -> A nie
    # moze przewyzszac B mimo wysokiej zgodnosci.
    assert result.intermediate["veto_triggered"][0, 1] == True
    assert result.intermediate["outranking"][0, 1] == False


def test_electre_kernel_is_subset_of_alternatives():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    result = ElectreIStrategy().calculate_ranking(problem)

    kernel = result.intermediate["kernel"]
    assert len(kernel) >= 1
    assert set(kernel).issubset(set(range(problem.n_alternatives)))
    assert len(result.ranking) == problem.n_alternatives
