"""
Testy dla widoku /compare (porownanie side-by-side dwoch niezaleznych
problemow decyzyjnych).
"""

import io
import json

import pytest

from webapp import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


def test_compare_page_loads(client):
    response = client.get("/compare")
    assert response.status_code == 200
    assert "Zbior A".encode("utf-8") in response.data
    assert "Zbior B".encode("utf-8") in response.data


def test_compare_two_builtin_examples(client):
    response = client.post(
        "/compare",
        data={
            "example_a": "sewage_network_variants",
            "example_b": "wastewater_treatment_vientiane",
        },
    )
    assert response.status_code == 200
    assert "Wariant_1_Grawitacyjny".encode("utf-8") in response.data
    assert "CEWATS_I".encode("utf-8") in response.data
    for method in [b"TOPSIS", b"AHP", b"PROMETHEE", b"ELECTRE I"]:
        assert response.data.count(method) >= 2


def test_compare_does_not_affect_main_session_problem(client):
    """Porownanie jest niezalezne od 'biezacego problemu' w sesji (PU1)."""
    client.post("/load-example/laptop_selection", follow_redirects=True)

    client.post(
        "/compare",
        data={
            "example_a": "sewage_network_variants",
            "example_b": "wastewater_treatment_vientiane",
        },
    )

    response = client.get("/problem")
    assert response.status_code == 200
    # "Biezacy problem" wciaz powinien byc laptop_selection, nie zaden
    # z problemow uzytych w porownaniu.
    assert "Laptop A".encode("utf-8") in response.data


def test_compare_with_custom_uploaded_file(client):
    custom_payload = {
        "matrix": [[1, 7], [2, 4], [3, 1]],
        "weights": [0.5, 0.5],
        "directions": ["max", "max"],
        "alternative_names": ["X", "Y", "Z"],
        "criterion_names": ["K1", "K2"],
    }
    data = {
        "example_a": "laptop_selection",
        "file_b": (
            io.BytesIO(json.dumps(custom_payload).encode("utf-8")),
            "custom.json",
        ),
    }
    response = client.post("/compare", data=data, content_type="multipart/form-data")
    assert response.status_code == 200
    assert "custom.json".encode("utf-8") in response.data


def test_compare_missing_selection_shows_error(client):
    response = client.post(
        "/compare",
        data={"example_a": "laptop_selection"},  # brak example_b
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Blad wczytywania danych".encode("utf-8") in response.data


def test_compare_invalid_uploaded_json_shows_error(client):
    bad_payload = {
        "matrix": [[1, 2], [3, 4]],
        "weights": [0.3, 0.3],  # suma != 1.0
        "directions": ["max", "max"],
        "alternative_names": ["A", "B"],
        "criterion_names": ["K1", "K2"],
    }
    data = {
        "example_a": "laptop_selection",
        "file_b": (
            io.BytesIO(json.dumps(bad_payload).encode("utf-8")),
            "bad.json",
        ),
    }
    response = client.post(
        "/compare", data=data, content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Blad wczytywania danych".encode("utf-8") in response.data
