"""Testy: rozstrzyganie remisow i zmiany pozycji po usunieciu alternatywy."""

from models.decision_problem import DecisionProblem
from strategies import AhpStrategy, TopsisStrategy, PrometheeStrategy, ElectreIStrategy
from evaluation.rank_reversal import simulate_rank_reversal, _compare_positions


def _problem_with_duplicates():
    # alternatywy A i C maja identyczne oceny, B jest wyraznie gorsza
    return DecisionProblem(
        matrix=[[5, 5, 5], [1, 1, 1], [5, 5, 5]],
        weights=[1 / 3, 1 / 3, 1 / 3],
        directions=["max", "max", "max"],
        alternative_names=["A", "B", "C"],
        criterion_names=["K1", "K2", "K3"],
    )


def test_ties_keep_input_order_for_all_methods():
    problem = _problem_with_duplicates()
    for strategy in (AhpStrategy(), TopsisStrategy(), PrometheeStrategy(), ElectreIStrategy()):
        ranking = strategy.calculate_ranking(problem).ranking
        assert ranking.index(0) < ranking.index(2), strategy.name  # A przed C
        assert ranking[-1] == 1, strategy.name  # B ostatnia


def test_compare_positions_ignores_shift_after_removing_leader():
    changes = _compare_positions(["A", "B", "C"], ["B", "C"], "A")
    assert changes["A"] == (1, None)
    assert changes["B"] == (1, 1)
    assert changes["C"] == (2, 2)


def test_removing_leader_with_unchanged_order_reports_no_changes():
    problem = DecisionProblem(
        matrix=[[9, 9], [6, 6], [4, 4], [2, 2]],
        weights=[0.5, 0.5],
        directions=["max", "max"],
        alternative_names=["A", "B", "C", "D"],
        criterion_names=["K1", "K2"],
    )
    report = simulate_rank_reversal(problem, TopsisStrategy(), "A")
    assert not report.reversal_detected
    assert all(b == a for n, (b, a) in report.position_changes.items() if a is not None)
