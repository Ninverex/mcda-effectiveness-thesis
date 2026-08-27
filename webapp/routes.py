"""
Warstwa Controller (Flask Blueprint) -- spina Model (mcdm/) z View
(templates/). Kazda funkcja widoku odpowiada jednemu z przypadkow
uzycia PU1/PU2/PU4 opisanych w pracy.

Biezacy problem decyzyjny trzymany jest w sesji Flask (signed cookie)
jako slownik zwracany przez DecisionProblem.to_dict() -- prototyp nie
wymaga bazy danych ani serwera stanu.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flask import Blueprint, flash, redirect, render_template, request, session, url_for

from mcdm.controller import MCDMController
from mcdm.evaluation.rank_correlation import compare_all_pairs
from mcdm.evaluation.rank_reversal import simulate_rank_reversal
from mcdm.evaluation.sensitivity import weight_sensitivity_analysis
from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies import AhpStrategy, ElectreIStrategy, PrometheeStrategy, TopsisStrategy
from mcdm.validation.validators import ValidationError, validate_decision_problem

bp = Blueprint("main", __name__)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "examples"

EXAMPLE_LABELS = {
    "laptop_selection": "Wybor laptopa (przyklad dydaktyczny)",
    "sewage_network_variants": "Warianty sieci kanalizacyjnej (przyklad domenowy)",
}


def _build_controller() -> MCDMController:
    """Rejestruje wszystkie 4 zaimplementowane metody. Tanie w konstrukcji,
    wiec tworzone na kazde zadanie -- brak potrzeby trzymania stanu."""
    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    controller.register_strategy(AhpStrategy())
    controller.register_strategy(PrometheeStrategy())
    controller.register_strategy(ElectreIStrategy())
    return controller


def _get_current_problem() -> DecisionProblem | None:
    data = session.get("problem")
    if data is None:
        return None
    return DecisionProblem.from_dict(data)


def _store_problem(problem: DecisionProblem, source_label: str) -> None:
    session["problem"] = problem.to_dict()
    session["problem_source"] = source_label


def _require_problem():
    """Zwraca (problem, None) albo (None, redirect) gdy brak problemu w sesji."""
    problem = _get_current_problem()
    if problem is None:
        flash("Najpierw wczytaj problem decyzyjny.", "error")
        return None, redirect(url_for("main.index"))
    return problem, None


# ----------------------------------------------------------------------
# PU1: Zarzadzanie problemem decyzyjnym i walidacja danych
# ----------------------------------------------------------------------


@bp.route("/")
def index():
    return render_template(
        "index.html",
        examples=EXAMPLE_LABELS,
        problem_source=session.get("problem_source"),
    )


@bp.route("/load-example/<name>", methods=["POST"])
def load_example(name):
    if name not in EXAMPLE_LABELS:
        flash(f"Nieznany przyklad: {name}", "error")
        return redirect(url_for("main.index"))

    path = EXAMPLES_DIR / f"{name}.json"
    try:
        problem = DecisionProblem.from_json(path)
        validate_decision_problem(problem)
    except (ValidationError, FileNotFoundError, json.JSONDecodeError) as exc:
        flash(f"Blad wczytywania przykladu: {exc}", "error")
        return redirect(url_for("main.index"))

    _store_problem(problem, EXAMPLE_LABELS[name])
    flash(f"Wczytano problem: {EXAMPLE_LABELS[name]}", "success")
    return redirect(url_for("main.problem_detail"))


@bp.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("problem_file")
    if not file or file.filename == "":
        flash("Nie wybrano pliku.", "error")
        return redirect(url_for("main.index"))

    try:
        payload = json.load(file.stream)
        problem = DecisionProblem.from_dict(payload)
        # PU1, przeplyw alternatywny: precyzyjny komunikat bledu
        # ze wskazaniem konkretnej komorki macierzy (patrz validators.py).
        validate_decision_problem(problem)
    except (ValidationError, json.JSONDecodeError, KeyError, ValueError) as exc:
        flash(f"Blad walidacji pliku '{file.filename}': {exc}", "error")
        return redirect(url_for("main.index"))

    _store_problem(problem, file.filename)
    flash(f"Wczytano i zwalidowano plik: {file.filename}", "success")
    return redirect(url_for("main.problem_detail"))


@bp.route("/problem")
def problem_detail():
    problem, err = _require_problem()
    if err:
        return err

    rows = list(
        zip(
            problem.alternative_names,
            problem.matrix.tolist(),
            problem.active_mask.tolist(),
        )
    )

    return render_template(
        "problem_detail.html",
        problem=problem,
        source=session.get("problem_source"),
        rows=rows,
    )


@bp.route("/reset", methods=["POST"])
def reset():
    session.pop("problem", None)
    session.pop("problem_source", None)
    flash("Wyczyszczono biezacy problem decyzyjny.", "success")
    return redirect(url_for("main.index"))


# ----------------------------------------------------------------------
# PU2: Definicja preferencji eksperckich -- AHP, porownania parami
# ----------------------------------------------------------------------


@bp.route("/problem/ahp-pairwise", methods=["GET", "POST"])
def ahp_pairwise():
    problem, err = _require_problem()
    if err:
        return err

    n = problem.n_criteria
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    result = None
    submitted_values = {f"pair_{i}_{j}": "1" for i, j in pairs}

    if request.method == "POST":
        matrix = np.ones((n, n))
        try:
            for i, j in pairs:
                raw = request.form.get(f"pair_{i}_{j}", "1")
                submitted_values[f"pair_{i}_{j}"] = raw
                value = _parse_saaty_value(raw)
                matrix[i, j] = value
                matrix[j, i] = 1.0 / value
        except (ValueError, ZeroDivisionError) as exc:
            flash(f"Nieprawidlowa wartosc porownania: {exc}", "error")
            return render_template(
                "ahp_pairwise.html",
                problem=problem,
                pairs=pairs,
                result=None,
                values=submitted_values,
            )

        result = AhpStrategy.compute_consistency(matrix)

        if "use_weights" in request.form:
            if not result["consistent"]:
                # PU2, rozszerzenie: CR > 0.10 blokuje przejscie do
                # obliczen koncowych, dopoki uzytkownik nie zrewiduje
                # najbardziej niespojnych porownan.
                flash(
                    f"CR = {result['CR']:.3f} przekracza prog 0.10 -- "
                    "zrewiduj porownania przed uzyciem tych wag.",
                    "error",
                )
            else:
                data = problem.to_dict()
                data["weights"] = result["weights"].tolist()
                session["problem"] = data
                flash("Zastosowano wagi wyliczone z macierzy porownan AHP.", "success")
                return redirect(url_for("main.problem_detail"))

    return render_template(
        "ahp_pairwise.html",
        problem=problem,
        pairs=pairs,
        result=result,
        values=submitted_values,
    )


def _parse_saaty_value(raw: str) -> float:
    """Parsuje wartosc ze skali Saaty'ego: '5' lub '1/5'."""
    raw = raw.strip()
    if "/" in raw:
        num, den = raw.split("/", 1)
        return float(num) / float(den)
    return float(raw)


# ----------------------------------------------------------------------
# PU4: Analiza wynikow i symulacja odwrocenia rankingu
# ----------------------------------------------------------------------


@bp.route("/problem/results")
def results():
    problem, err = _require_problem()
    if err:
        return err

    controller = _build_controller()
    try:
        all_results = controller.run_all(problem)
    except ValidationError as exc:
        flash(f"Blad walidacji: {exc}", "error")
        return redirect(url_for("main.problem_detail"))

    correlations = compare_all_pairs(all_results)

    return render_template(
        "results.html",
        problem=problem,
        results=all_results,
        correlations=correlations,
    )


@bp.route("/problem/rank-reversal", methods=["GET", "POST"])
def rank_reversal():
    problem, err = _require_problem()
    if err:
        return err

    controller = _build_controller()
    report = None

    if request.method == "POST":
        method_name = request.form.get("method")
        alternative_name = request.form.get("alternative")
        strategy = controller.get_strategy(method_name)
        if strategy is None:
            flash("Nieznana metoda.", "error")
        else:
            try:
                report = simulate_rank_reversal(problem, strategy, alternative_name)
            except ValueError as exc:
                flash(str(exc), "error")

    return render_template(
        "rank_reversal.html",
        problem=problem,
        methods=controller.available_methods(),
        report=report,
    )


@bp.route("/problem/sensitivity", methods=["GET", "POST"])
def sensitivity():
    problem, err = _require_problem()
    if err:
        return err

    controller = _build_controller()
    report = None

    if request.method == "POST":
        method_name = request.form.get("method")
        criterion_name = request.form.get("criterion")
        strategy = controller.get_strategy(method_name)
        if strategy is None:
            flash("Nieznana metoda.", "error")
        else:
            try:
                report = weight_sensitivity_analysis(problem, strategy, criterion_name)
            except ValueError as exc:
                flash(str(exc), "error")

    return render_template(
        "sensitivity.html",
        problem=problem,
        methods=controller.available_methods(),
        report=report,
    )
