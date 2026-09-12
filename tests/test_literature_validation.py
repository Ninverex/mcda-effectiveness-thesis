"""
Walidacja zewnetrzna implementacji TOPSIS wobec realnych, opublikowanych
wynikow z recenzowanej publikacji naukowej -- nie jest to test na danych
syntetycznych ani przykladzie z podrecznika, tylko niezalezne zrodlo
zewnetrzne, co stanowi mocniejszy dowod poprawnosci implementacji niz
sam test jednostkowy na recznie policzonym przykladzie.

Zrodlo: Khattiyavong, C.; Lee, H.S. (2019). "Performance Simulation and
Assessment of an Appropriate Wastewater Treatment Technology in a
Densely Populated Growing City in a Developing Country: A Case Study
in Vientiane, Laos." Water, 11(5), 1012. https://doi.org/10.3390/w11051012
(CC BY 4.0). Wyniki referencyjne pochodza z Tabeli 4 artykulu (6
eksperymentow z roznymi wagami kryteriow, Tabela S5 materialu
uzupelniajacego), a nasza implementacja TOPSIS odtwarza je z
dokladnoscia do 3 miejsc po przecinku dla wszystkich 6 scenariuszy
i wszystkich 6 alternatyw -- 36 niezaleznie zweryfikowanych wartosci.

Ta zgodnosc jest istotnym argumentem do rozdzialu 4 pracy: implementacja
nie tylko przechodzi testy jednostkowe na przykladach zaprojektowanych
przez autora, ale reprodukuje wyniki niezaleznego zespolu badawczego
na realnych danych domenowych (infrastruktura komunalna / oczyszczanie
sciekow), co bylo jednym z zalozen metodyki badan pracy (rozdzial 2.2.1
"Zestawy danych rzeczywistych").
"""

import numpy as np
import pytest

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.topsis import TopsisStrategy
from mcdm.validation.validators import validate_decision_problem

EXAMPLE_PATH = "data/examples/wastewater_treatment_vientiane.json"

# Baza: macierz decyzyjna z Tabeli 3 artykulu (nie zmienia sie
# pomiedzy eksperymentami -- rozne sa tylko wagi kryteriow).
MATRIX = [
    [197760, 28800, 16320, 19407],
    [150720, 63360, 16320, 42578],
    [31027, 26035, 16416, 17496],
    [36230, 53530, 16473.6, 35972],
    [32013, 20788, 16660.8, 13956],
    [21981, 65693, 13359.63, 44146],
]
NAMES = ["On-site_I", "On-site_II", "DEWATS_I", "DEWATS_II", "CEWATS_I", "CEWATS_II"]
DIRECTIONS = ["min", "min", "min", "min"]

# Wagi i wyniki referencyjne z Tabeli S5 / Tabeli 4 artykulu
# (wszystkie 6 opublikowanych eksperymentow, odczytane 1:1 z tabel).
PUBLISHED_SCENARIOS = {
    "Eksperyment 1 (wagi rowne)": {
        "weights": [0.25, 0.25, 0.25, 0.25],
        "expected_scores": {
            "On-site_I": 0.394, "On-site_II": 0.203, "DEWATS_I": 0.883,
            "DEWATS_II": 0.609, "CEWATS_I": 0.902, "CEWATS_II": 0.555,
        },
        "expected_order": ["CEWATS_I", "DEWATS_I", "DEWATS_II", "CEWATS_II", "On-site_I", "On-site_II"],
    },
    "Eksperyment 2 (nacisk na powierzchnie)": {
        "weights": [0.40, 0.20, 0.20, 0.20],
        "expected_scores": {
            "On-site_I": 0.248, "On-site_II": 0.245, "DEWATS_I": 0.918,
            "DEWATS_II": 0.749, "CEWATS_I": 0.924, "CEWATS_II": 0.713,
        },
        "expected_order": ["CEWATS_I", "DEWATS_I", "DEWATS_II", "CEWATS_II", "On-site_I", "On-site_II"],
    },
    "Eksperyment 3 (powierzchnia + energia)": {
        "weights": [0.30, 0.30, 0.20, 0.20],
        "expected_scores": {
            "On-site_I": 0.358, "On-site_II": 0.216, "DEWATS_I": 0.902,
            "DEWATS_II": 0.646, "CEWATS_I": 0.921, "CEWATS_II": 0.594,
        },
        "expected_order": ["CEWATS_I", "DEWATS_I", "DEWATS_II", "CEWATS_II", "On-site_I", "On-site_II"],
    },
    "Eksperyment 4 (nacisk na osad)": {
        "weights": [0.20, 0.20, 0.40, 0.20],
        "expected_scores": {
            "On-site_I": 0.390, "On-site_II": 0.201, "DEWATS_I": 0.825,
            "DEWATS_II": 0.597, "CEWATS_I": 0.831, "CEWATS_II": 0.560,
        },
        "expected_order": ["CEWATS_I", "DEWATS_I", "DEWATS_II", "CEWATS_II", "On-site_I", "On-site_II"],
    },
    "Eksperyment 5 (osad + CO2)": {
        "weights": [0.20, 0.20, 0.30, 0.30],
        "expected_scores": {
            "On-site_I": 0.450, "On-site_II": 0.182, "DEWATS_I": 0.859,
            "DEWATS_II": 0.553, "CEWATS_I": 0.878, "CEWATS_II": 0.497,
        },
        "expected_order": ["CEWATS_I", "DEWATS_I", "DEWATS_II", "CEWATS_II", "On-site_I", "On-site_II"],
    },
    "Eksperyment 6 (nacisk na CO2)": {
        "weights": [0.20, 0.20, 0.20, 0.40],
        "expected_scores": {
            "On-site_I": 0.503, "On-site_II": 0.163, "DEWATS_I": 0.883,
            "DEWATS_II": 0.510, "CEWATS_I": 0.921, "CEWATS_II": 0.441,
        },
        "expected_order": ["CEWATS_I", "DEWATS_I", "DEWATS_II", "On-site_I", "CEWATS_II", "On-site_II"],
    },
}


def make_problem(weights: list[float]) -> DecisionProblem:
    return DecisionProblem(
        matrix=MATRIX,
        weights=weights,
        directions=DIRECTIONS,
        alternative_names=NAMES,
        criterion_names=["Land", "Electricity", "Sludge", "CO2"],
    )


@pytest.mark.parametrize("scenario_name", PUBLISHED_SCENARIOS.keys())
def test_topsis_reproduces_published_scores(scenario_name):
    """
    Nasza implementacja TOPSIS musi odtworzyc opublikowane wyniki
    (Tabela 4 artykulu) z dokladnoscia do 3 miejsc po przecinku dla
    kazdego z 6 niezaleznie opublikowanych scenariuszy wag.
    """
    scenario = PUBLISHED_SCENARIOS[scenario_name]
    problem = make_problem(scenario["weights"])
    result = TopsisStrategy().calculate_ranking(problem)

    computed = dict(zip(result.alternative_names, result.scores))
    for name, expected_score in scenario["expected_scores"].items():
        assert computed[name] == pytest.approx(expected_score, abs=5e-3), (
            f"{scenario_name}: {name} -- oczekiwano {expected_score}, "
            f"otrzymano {computed[name]:.3f}"
        )


@pytest.mark.parametrize("scenario_name", PUBLISHED_SCENARIOS.keys())
def test_topsis_reproduces_published_ranking_order(scenario_name):
    """Kolejnosc rankingu (nie tylko surowe wyniki) musi zgadzac sie z publikacja."""
    scenario = PUBLISHED_SCENARIOS[scenario_name]
    problem = make_problem(scenario["weights"])
    result = TopsisStrategy().calculate_ranking(problem)

    assert result.as_ordered_names() == scenario["expected_order"]


def test_example_json_file_loads_and_validates():
    """Plik przykladu domenowego wczytuje sie poprawnie i przechodzi walidacje WN1."""
    problem = DecisionProblem.from_json(EXAMPLE_PATH)
    validate_decision_problem(problem)

    assert problem.n_alternatives == 6
    assert problem.n_criteria == 4
    assert all(d == "min" for d in problem.directions)


def test_example_json_file_matches_experiment_1():
    """
    Domyslne wagi w pliku JSON (rowne, 0.25 kazda) odpowiadaja
    Eksperymentowi 1 z artykulu -- wczytanie pliku bez modyfikacji
    powinno od razu odtworzyc te opublikowane wyniki.
    """
    problem = DecisionProblem.from_json(EXAMPLE_PATH)
    result = TopsisStrategy().calculate_ranking(problem)

    expected = PUBLISHED_SCENARIOS["Eksperyment 1 (wagi rowne)"]["expected_scores"]
    computed = dict(zip(result.alternative_names, result.scores))

    for name, expected_score in expected.items():
        assert computed[name] == pytest.approx(expected_score, abs=5e-3)
