"""
Testy jednostkowe dla mcdm/visualization/diff_report.py.
"""

from mcdm.models.decision_problem import DecisionProblem
from mcdm.strategies.topsis import TopsisStrategy
from mcdm.evaluation.rank_reversal import simulate_rank_reversal
from mcdm.visualization.diff_report import (
    build_diff_html_table,
    plot_rank_reversal_diff,
)
from mcdm.visualization._utils import figure_to_base64

EXAMPLES_DIR = "data/examples"


def _make_report():
    problem = DecisionProblem.from_json(
        f"{EXAMPLES_DIR}/sewage_network_variants.json"
    )
    return simulate_rank_reversal(
        problem, TopsisStrategy(), alternative_name="Wariant_2_Modulowy"
    )


def test_build_diff_html_table_contains_all_alternatives():
    report = _make_report()
    html = build_diff_html_table(report)

    assert "<table" in html
    for name in report.baseline_order:
        assert name in html


def test_build_diff_html_table_marks_removed_alternative():
    report = _make_report()
    html = build_diff_html_table(report)

    assert "diff-row-removed" in html
    assert "usunieta" in html


def test_build_diff_html_table_marks_unchanged_and_changed_rows():
    report = _make_report()
    html = build_diff_html_table(report)

    # Przynajmniej jedna z klas powinna wystapic (zalezy od danych,
    # ale w tym przykladzie oczekujemy przynajmniej "unchanged")
    assert "diff-row-unchanged" in html or "diff-row-changed" in html


def test_plot_rank_reversal_diff_returns_valid_figure():
    report = _make_report()
    fig = plot_rank_reversal_diff(report)

    assert fig is not None
    encoded = figure_to_base64(fig)
    assert isinstance(encoded, str)
    assert len(encoded) > 100


def test_plot_rank_reversal_diff_excludes_removed_alternative_from_lines():
    report = _make_report()
    fig = plot_rank_reversal_diff(report)
    ax = fig.axes[0]

    annotation_texts = {a.get_text() for a in ax.texts}
    assert report.removed_alternative not in annotation_texts
