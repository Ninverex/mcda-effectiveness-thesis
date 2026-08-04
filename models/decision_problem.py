"""
Model danych: Problem_Decyzyjny.

Odwzorowuje relację 1:N (Problem_Decyzyjny -> Kryterium/Alternatywa)
oraz encję asocjacyjną Macierz_Decyzyjna_Ocena opisaną w projekcie
bazy danych pracy inżynierskiej. Zamiast osobnych tabel SQL,
na potrzeby prototypu obliczeniowego wszystko trzyma się w jednej
strukturze w pamięci (numpy.ndarray), a persystencja odbywa się
przez pliki JSON (patrz: data/schema.json).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np


VALID_DIRECTIONS = {"max", "min"}  # max = kryterium typu "zysk", min = "koszt"


@dataclass
class DecisionProblem:
    """
    Reprezentuje pojedynczy problem decyzyjny MCDM.

    Attributes
    ----------
    matrix : np.ndarray, shape (m, n)
        Macierz decyzyjna X. Wiersze = alternatywy, kolumny = kryteria.
    weights : np.ndarray, shape (n,)
        Wektor wag kryteriów. Powinien sumować się do 1.0.
    directions : list[str]
        Kierunek optymalizacji per kryterium: "max" (zysk) lub "min" (koszt).
    alternative_names : list[str]
    criterion_names : list[str]
    active_mask : np.ndarray[bool], shape (m,)
        Flaga "Status_Aktywnosci" z projektu bazy danych. Pozwala
        "wyłączyć" alternatywę z obliczeń bez fizycznego usuwania
        jej z danych -- kluczowe przy symulacji rank reversal.
    thresholds : dict | None
        Parametry specyficzne dla ELECTRE/PROMETHEE, np.
        {"q": [...], "p": [...], "v": [...]} -- progi indyferencji,
        preferencji i weta, per kryterium. Mogą być NULL/None jeśli
        używane są tylko metody AHP/TOPSIS.
    """

    matrix: np.ndarray
    weights: np.ndarray
    directions: list[str]
    alternative_names: list[str]
    criterion_names: list[str]
    active_mask: np.ndarray = field(default=None)
    thresholds: dict | None = None

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)

        if self.active_mask is None:
            self.active_mask = np.ones(self.matrix.shape[0], dtype=bool)
        else:
            self.active_mask = np.asarray(self.active_mask, dtype=bool)

        if any(d not in VALID_DIRECTIONS for d in self.directions):
            raise ValueError(
                f"Kierunek optymalizacji musi byc jednym z {VALID_DIRECTIONS}, "
                f"otrzymano: {self.directions}"
            )

    # ------------------------------------------------------------------
    # Wlasciwosci pomocnicze
    # ------------------------------------------------------------------

    @property
    def active_matrix(self) -> np.ndarray:
        """Macierz decyzyjna ograniczona do aktywnych alternatyw (rank reversal)."""
        return self.matrix[self.active_mask]

    @property
    def active_names(self) -> list[str]:
        return [
            name for name, active in zip(self.alternative_names, self.active_mask)
            if active
        ]

    @property
    def n_alternatives(self) -> int:
        return int(self.active_mask.sum())

    @property
    def n_criteria(self) -> int:
        return self.matrix.shape[1]

    # ------------------------------------------------------------------
    # Konstruktory / serializacja
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "DecisionProblem":
        """Wczytuje problem decyzyjny z pliku JSON (patrz data/examples/)."""
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict) -> "DecisionProblem":
        return cls(
            matrix=payload["matrix"],
            weights=payload["weights"],
            directions=payload["directions"],
            alternative_names=payload["alternative_names"],
            criterion_names=payload["criterion_names"],
            active_mask=payload.get("active_mask"),
            thresholds=payload.get("thresholds"),
        )

    def to_dict(self) -> dict:
        return {
            "matrix": self.matrix.tolist(),
            "weights": self.weights.tolist(),
            "directions": self.directions,
            "alternative_names": self.alternative_names,
            "criterion_names": self.criterion_names,
            "active_mask": self.active_mask.tolist(),
            "thresholds": self.thresholds,
        }

    def copy_with(self, **overrides) -> "DecisionProblem":
        """
        Zwraca nowa instancje z podmienionymi polami (np. nowe wagi
        przy analizie wrazliwosci, albo zmieniona active_mask przy
        symulacji rank reversal). Nie mutuje oryginalu.
        """
        data = self.to_dict()
        data.update(overrides)
        return DecisionProblem.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"DecisionProblem(alternatives={self.n_alternatives}/"
            f"{len(self.alternative_names)}, criteria={self.n_criteria})"
        )
