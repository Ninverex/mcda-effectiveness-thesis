"""
Skrypt demonstracyjny: wczytanie problemu decyzyjnego, walidacja,
uruchomienie TOPSIS przez kontroler.

W miare dopisywania kolejnych strategii (AHP, PROMETHEE, ELECTRE)
ten skrypt bedzie rozbudowywany o controller.run_all(...) oraz
wywolania modulu evaluation (Kendall/Spearman, rank reversal,
sensitivity) -- docelowo to on wygeneruje dane do rozdzialu 4 pracy.

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


if __name__ == "__main__":
    main()
