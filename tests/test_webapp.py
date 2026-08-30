"""
Testy integracyjne warstwy GUI (Flask), przy uzyciu wbudowanego
test clienta -- bez odpalania prawdziwego serwera HTTP.

Pokrywaja glowne sciezki przypadkow uzycia:
- PU1: wczytanie przykladu, upload wlasnego pliku, podglad problemu,
  reset sesji.
- PU2: obliczenie wag AHP z macierzy porownan parami, w tym blokada
  przy CR > 0.10.
- PU4: wyniki wszystkich metod, symulacja rank reversal, analiza
  wrazliwosci wag.
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


def _load_example(client, name="sewage_network_variants"):
    return client.post(f"/load-example/{name}", follow_redirects=True)


# ----------------------------------------------------------------------
# PU1
# ----------------------------------------------------------------------

def test_index_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "MCDM Toolkit".encode() in response.data


def test_load_example_and_view_problem(client):
    response = _load_example(client)
    assert response.status_code == 200
    assert "Wariant_1_Grawitacyjny".encode("utf-8") in response.data


def test_problem_page_redirects_without_loaded_problem(client):
    response = client.get("/problem", follow_redirects=True)
    assert response.status_code == 200
    # Powinno przekierowac na strone glowna (brak problemu w sesji)
    assert "Wybierz przykladowy problem".encode("utf-8") in response.data


def test_upload_invalid_json_shows_validation_error(client):
    bad_payload = {
        "matrix": [[1, 2], [3, 4]],
        "weights": [0.3, 0.3],  # suma != 1.0 -- powinno zostac odrzucone
        "directions": ["max", "max"],
        "alternative_names": ["A", "B"],
        "criterion_names": ["K1", "K2"],
    }
    data = {
        "problem_file": (
            io.BytesIO(json.dumps(bad_payload).encode("utf-8")),
            "bad.json",
        )
    }
    response = client.post(
        "/upload", data=data, content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "Blad walidacji".encode("utf-8") in response.data


def test_upload_valid_json_succeeds(client):
    good_payload = {
        "matrix": [[1, 2], [3, 4]],
        "weights": [0.5, 0.5],
        "directions": ["max", "max"],
        "alternative_names": ["A", "B"],
        "criterion_names": ["K1", "K2"],
    }
    data = {
        "problem_file": (
            io.BytesIO(json.dumps(good_payload).encode("utf-8")),
            "good.json",
        )
    }
    response = client.post(
        "/upload", data=data, content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "good.json".encode("utf-8") in response.data


def test_reset_clears_session(client):
    _load_example(client)
    response = client.post("/reset", follow_redirects=True)
    assert response.status_code == 200
    # Po resecie strona glowna nie powinna pokazywac aktywnego problemu
    assert "Aktualnie wczytany problem".encode("utf-8") not in response.data


# ----------------------------------------------------------------------
# PU2 -- AHP porownania parami
# ----------------------------------------------------------------------

def test_ahp_pairwise_page_loads(client):
    _load_example(client)
    response = client.get("/problem/ahp-pairwise")
    assert response.status_code == 200
    assert "porownan parami".encode("utf-8") in response.data


def test_ahp_pairwise_consistent_allows_using_weights(client):
    _load_example(client)
    # sewage_network_variants ma 4 kryteria -> 6 par
    form_data = {
        "pair_0_1": "2",
        "pair_0_2": "1",
        "pair_0_3": "1",
        "pair_1_2": "1/2",
        "pair_1_3": "1/2",
        "pair_2_3": "1",
    }
    response = client.post("/problem/ahp-pairwise", data=form_data)
    assert response.status_code == 200
    assert "spojne".encode("utf-8") in response.data or "CR".encode("utf-8") in response.data


def test_ahp_pairwise_inconsistent_blocks_weight_usage(client):
    _load_example(client)
    # Skrajnie cykliczne porownania -> wysokie CR
    form_data = {
        "pair_0_1": "9",
        "pair_0_2": "1/9",
        "pair_0_3": "1",
        "pair_1_2": "9",
        "pair_1_3": "1/9",
        "pair_2_3": "9",
        "use_weights": "1",
    }
    response = client.post(
        "/problem/ahp-pairwise", data=form_data, follow_redirects=True
    )
    assert response.status_code == 200
    assert "przekracza prog 0.10".encode("utf-8") in response.data


# ----------------------------------------------------------------------
# PU4
# ----------------------------------------------------------------------

def test_results_page_shows_all_methods(client):
    _load_example(client)
    response = client.get("/problem/results")
    assert response.status_code == 200
    for method in ["TOPSIS", "AHP", "PROMETHEE", "ELECTRE I"]:
        assert method.encode("utf-8") in response.data


def test_rank_reversal_simulation(client):
    _load_example(client)
    response = client.post(
        "/problem/rank-reversal",
        data={"method": "TOPSIS", "alternative": "Wariant_2_Modulowy"},
    )
    assert response.status_code == 200
    assert "Wariant_2_Modulowy".encode("utf-8") in response.data


def test_sensitivity_analysis(client):
    _load_example(client)
    response = client.post(
        "/problem/sensitivity",
        data={"method": "TOPSIS", "criterion": "Koszt_inwestycji"},
    )
    assert response.status_code == 200
    assert "Lider rankingu".encode("utf-8") in response.data
