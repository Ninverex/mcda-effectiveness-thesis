"""
Eksperymenty E1--E3 z rozdzialu 2 pracy, uruchamiane na dwoch zbiorach
rzeczywistych (Vientiane, Turcja) i czterech metodach (TOPSIS, AHP,
PROMETHEE II, ELECTRE I). Wyniki zapisywane sa jako pliki CSV w
experiments/output/results/ (do tabel w rozdziale 4).

  E1  ranking bazowy, tau/rho dla par metod, zgodnosc pozycji skrajnych
      i srodkowych
  E2  zmiana jednej wagi (delta od -0.20 do +0.20, 9 punktow)
      oraz scenariusze wag z publikacji (tylko Vientiane)
  E3  usuniecie kazdej alternatywy po kolei (rank reversal, top-3)

Uruchomienie z katalogu glownego repozytorium:
    python experiments/run_experiments.py
"""

from __future__ import annotations

import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# katalog główny projektu = ten, w którym leży folder "models" (skrypt może leżeć w korzeniu lub w experiments/)
ROOT = _HERE if (_HERE / "models").is_dir() else _HERE.parent
sys.path.insert(0, str(ROOT))

from models.decision_problem import DecisionProblem
from strategies import (
    AhpStrategy,
    ElectreIStrategy,
    PrometheeStrategy,
    TopsisStrategy,
)
from evaluation.rank_correlation import compare_rankings
from evaluation.rank_reversal import simulate_rank_reversal
from evaluation.sensitivity import weight_sensitivity_analysis

OUT = ROOT / "experiments" / "output" / "results"

DATASETS = {
    "vientiane": ROOT / "data/examples/wastewater_treatment_vientiane.json",
    "turcja": ROOT / "data/examples/energy_turkey_sahin.json",
}

# Zestawy wag z publikacji Khattiyavong i Lee (2019), tabela S5.
VIENTIANE_SCENARIOS = {
    "E1 rowne": [0.25, 0.25, 0.25, 0.25],
    "E2 powierzchnia": [0.40, 0.20, 0.20, 0.20],
    "E3 powierzchnia+energia": [0.30, 0.30, 0.20, 0.20],
    "E4 osad": [0.20, 0.20, 0.40, 0.20],
    "E5 osad+CO2": [0.20, 0.20, 0.30, 0.30],
    "E6 CO2": [0.20, 0.20, 0.20, 0.40],
}


def strategies():
    return [TopsisStrategy(), AhpStrategy(), PrometheeStrategy(), ElectreIStrategy()]


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(header)
        w.writerows(rows)
    print(f"  zapisano {path.relative_to(ROOT)}")


def positions(ranking: list[int], m: int) -> list[int]:
    pos = [0] * m
    for p, idx in enumerate(ranking, start=1):
        pos[idx] = p
    return pos


def normalized_scores(problem: DecisionProblem) -> np.ndarray:
    """Oceny znormalizowane min-max z uwzglednieniem kierunku (wiecej = lepiej)."""
    X = problem.active_matrix
    out = np.zeros_like(X, dtype=float)
    for j, d in enumerate(problem.directions):
        lo, hi = X[:, j].min(), X[:, j].max()
        span = hi - lo
        if span == 0:
            out[:, j] = 0.5
        elif d == "max":
            out[:, j] = (X[:, j] - lo) / span
        else:
            out[:, j] = (hi - X[:, j]) / span
    return out


def ties_in(scores) -> bool:
    s = np.round(np.asarray(scores, dtype=float), 12)
    return len(set(s)) < len(s)


# ----------------------------------------------------------------------
def experiment_e1(tag: str, problem: DecisionProblem) -> dict:
    m = len(problem.active_names)
    results = {s.name: s.calculate_ranking(problem) for s in strategies()}

    rows = []
    for name, r in results.items():
        pos = positions(list(r.ranking), m)
        for i, alt in enumerate(r.alternative_names):
            rows.append([name, alt, pos[i], f"{r.scores[i]:.6f}"])
    write_csv(f"e1_rankingi_{tag}.csv", ["metoda", "alternatywa", "pozycja", "wynik"], rows)

    ties = {name: ties_in(r.scores) for name, r in results.items()}
    rows = []
    for a, b in combinations(results, 2):
        ra, rb = list(results[a].ranking), list(results[b].ranking)
        corr = compare_rankings(ra, rb)
        pa, pb = positions(ra, m), positions(rb, m)
        best_same = ra[0] == rb[0]
        worst_same = ra[-1] == rb[-1]
        middle = [i for i in range(m) if pa[i] not in (1, m) and pb[i] not in (1, m)]
        mad = float(np.mean([abs(pa[i] - pb[i]) for i in middle])) if middle else float("nan")
        rows.append([
            a, b,
            f"{corr['kendall_tau']:.3f}", f"{corr['kendall_p_value']:.3f}",
            f"{corr['spearman_rho']:.3f}", f"{corr['spearman_p_value']:.3f}",
            int(best_same), int(worst_same), f"{mad:.2f}", len(middle),
        ])
    write_csv(
        f"e1_zgodnosc_{tag}.csv",
        ["metoda_A", "metoda_B", "tau", "p_tau", "rho", "p_rho",
         "najlepsza_ta_sama", "najgorsza_ta_sama", "sr_roznica_pozycji_srodkowych",
         "liczba_alt_srodkowych"],
        rows,
    )
    return {"results": results, "ties": ties}


# ----------------------------------------------------------------------
def experiment_e2_single(tag: str, problem: DecisionProblem, base: dict) -> None:
    m = len(problem.active_names)
    norm_std = normalized_scores(problem).std(axis=0)
    rows, summary = [], []
    for strat in strategies():
        base_rank = list(base["results"][strat.name].ranking)
        for j, crit in enumerate(problem.criterion_names):
            rep = weight_sensitivity_analysis(problem, strat, crit)
            taus, leaders = [], []
            for pt in rep.points:
                rank_idx = [problem.alternative_index(n) for n in pt.ranking_names]
                tau = compare_rankings(base_rank, rank_idx)["kendall_tau"]
                taus.append(tau)
                leaders.append(pt.ranking_names[0])
                rows.append([
                    strat.name, crit, f"{pt.delta:+.2f}",
                    ";".join(f"{w:.4f}" for w in pt.weights),
                    pt.ranking_names[0], f"{tau:.3f}",
                    ">".join(pt.ranking_names),
                ])
            summary.append([
                strat.name, crit, f"{norm_std[j]:.3f}",
                int(rep.is_stable()), len(set(leaders)),
                f"{min(taus):.3f}", f"{taus[0]:.3f}", f"{taus[-1]:.3f}",
            ])
    write_csv(
        f"e2_wrazliwosc_{tag}.csv",
        ["metoda", "kryterium", "delta", "wagi", "lider", "tau_vs_bazowy", "ranking"],
        rows,
    )
    write_csv(
        f"e2_podsumowanie_{tag}.csv",
        ["metoda", "kryterium", "std_ocen_znormalizowanych", "lider_stabilny",
         "liczba_roznych_liderow", "min_tau", "tau_delta_-0.20", "tau_delta_+0.20"],
        summary,
    )


def experiment_e2_scenarios(problem: DecisionProblem) -> None:
    m = len(problem.active_names)
    rows, taus_rows = [], []
    for strat in strategies():
        rankings = {}
        for scen, w in VIENTIANE_SCENARIOS.items():
            r = strat.calculate_ranking(problem.copy_with(weights=w))
            rankings[scen] = list(r.ranking)
            rows.append([
                strat.name, scen,
                ">".join(r.alternative_names[i] for i in r.ranking),
                int(ties_in(r.scores)),
            ])
        base = rankings["E1 rowne"]
        for scen, rk in rankings.items():
            if scen == "E1 rowne":
                continue
            taus_rows.append([
                strat.name, scen,
                f"{compare_rankings(base, rk)['kendall_tau']:.3f}",
                int(base[0] != rk[0]),
            ])
    write_csv("e2_scenariusze_vientiane.csv",
              ["metoda", "scenariusz", "ranking", "remis"], rows)
    write_csv("e2_scenariusze_vs_rowne_vientiane.csv",
              ["metoda", "scenariusz", "tau_vs_rowne", "zmiana_lidera"], taus_rows)


# ----------------------------------------------------------------------
def experiment_e3(tag: str, problem: DecisionProblem) -> None:
    rows = []
    for strat in strategies():
        for alt in problem.alternative_names:
            rep = simulate_rank_reversal(problem, strat, alt)
            changed = sum(
                1 for n, (b, a) in rep.position_changes.items()
                if a is not None and b != a
            )
            base_top = [n for n in rep.baseline_order if n != alt][:3]
            mod_top = [n for n in rep.modified_order if n != alt][:3]
            rows.append([
                strat.name, alt, int(rep.reversal_detected), changed,
                ">".join(base_top), ">".join(mod_top),
            ])
    write_csv(
        f"e3_odwrocenie_{tag}.csv",
        ["metoda", "usunieta", "odwrocenie_top3", "liczba_zmian_pozycji",
         "top3_bazowy", "top3_po_usunieciu"],
        rows,
    )


# ----------------------------------------------------------------------
def main() -> None:
    for tag, path in DATASETS.items():
        problem = DecisionProblem.from_json(path)
        print(f"\n=== {tag}: {problem} ===")
        base = experiment_e1(tag, problem)
        print("  remisy w rankingach bazowych:",
              {k: v for k, v in base["ties"].items() if v} or "brak")
        experiment_e2_single(tag, problem, base)
        if tag == "vientiane":
            experiment_e2_scenarios(problem)
        experiment_e3(tag, problem)
    print("\nGotowe.")


if __name__ == "__main__":
    main()
