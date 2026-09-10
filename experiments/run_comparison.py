"""
Skrypt demonstracyjny: wczytanie problemu decyzyjnego, walidacja,
uruchomienie wszystkich 4 metod przez kontroler, ocena zgodnosci
rankingow oraz zapis wizualizacji (radar, GAIA, rank reversal) do
plikow PNG w experiments/output/ -- gotowy material do rozdzialu 4
pracy (bez recznego odpalania GUI).

Uruchomienie:
    cd mcdm-toolkit
    python experiments/run_comparison.py
"""

from pathlib import Path
import sys

# Umozliwia uruchomienie skryptu bezposrednio z katalogu experiments/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcdm.models.decision_problem import DecisionProblem
from mcdm.controller import MCDMController
from mcdm.strategies import (
    TopsisStrategy,
    AhpStrategy,
    PrometheeStrategy,
    ElectreIStrategy,
)
from mcdm.evaluation.rank_correlation import compare_rankings
from mcdm.evaluation.rank_reversal import simulate_rank_reversal
from mcdm.visualization._utils import save_figure
from mcdm.visualization.diff_report import plot_rank_reversal_diff
from mcdm.visualization.gaia import plot_gaia_plane
from mcdm.visualization.radar import plot_radar_chart

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def main():
    problem = DecisionProblem.from_json(
        Path(__file__).resolve().parent.parent
        / "data/examples/sewage_network_variants.json"
    )

    controller = MCDMController()
    controller.register_strategy(TopsisStrategy())
    controller.register_strategy(AhpStrategy())
    controller.register_strategy(PrometheeStrategy())
    controller.register_strategy(ElectreIStrategy())

    print(f"Problem: {problem}")
    print(f"Metody dostepne: {controller.available_methods()}\n")

    results = controller.run_all(problem)

    for name, result in results.items():
        print(result)
        for pos, idx in enumerate(result.ranking, start=1):
            alt_name = result.alternative_names[idx]
            score = result.scores[idx]
            print(f"  {pos}. {alt_name:<28} score = {score:.4f}")
        print()

    print("=== Zgodnosc rankingow (Kendall tau / Spearman rho) ===")
    method_names = list(results.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            a, b = method_names[i], method_names[j]
            corr = compare_rankings(results[a].ranking, results[b].ranking)
            print(
                f"  {a:<12} vs {b:<12}: "
                f"tau={corr['kendall_tau']:.3f}, rho={corr['spearman_rho']:.3f}"
            )

    print(f"\n=== Zapisywanie wizualizacji do {OUTPUT_DIR} ===")

    radar_path = save_figure(
        plot_radar_chart(problem), OUTPUT_DIR / "radar_profiles.png"
    )
    print(f"  Zapisano: {radar_path}")

    gaia_path = save_figure(plot_gaia_plane(problem), OUTPUT_DIR / "gaia_plane.png")
    print(f"  Zapisano: {gaia_path}")

    # Rank reversal: usuwamy najslabsza alternatywe wg TOPSIS i patrzymy,
    # czy zmienila sie kolejnosc w czolowce (badanie stabilnosci z 4.4)
    topsis_result = results["TOPSIS"]
    weakest_alt = topsis_result.as_ordered_names()[-1]
    report = simulate_rank_reversal(
        problem, controller.get_strategy("TOPSIS"), alternative_name=weakest_alt
    )
    diff_path = save_figure(
        plot_rank_reversal_diff(report), OUTPUT_DIR / "rank_reversal_diff.png"
    )
    print(f"  Zapisano: {diff_path}")
    print(
        f"  ({'WYKRYTO' if report.reversal_detected else 'brak'} odwrocenia "
        f"rankingu po usunieciu '{weakest_alt}')"
    )


if __name__ == "__main__":
    main()
