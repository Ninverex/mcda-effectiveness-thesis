"""
Wspolne narzedzia numeryczne uzywane przez kilka strategii MCDM.

Kazda z metod (TOPSIS, ELECTRE, a czesciowo tez AHP i PROMETHEE) musi
w pewnym momencie dzielic przez wartosc, ktora teoretycznie moze
wyniesc zero (np. rozpietosc kryterium, na ktorym wszystkie
alternatywy maja identyczna ocene, albo suma odleglosci D+ i D-,
gdy wszystkie alternatywy sa identyczne). Dotychczas kazda strategia
powielala ten sam idiom (`tablica[tablica == 0] = 1e-12`) niezaleznie
-- ten modul zbiera go w jednym, przetestowanym miejscu.
"""

from __future__ import annotations

import numpy as np

#: Domyslny epsilon uzywany do zastepowania zerowych mianownikow.
#: Dobrany tak, by byc pomijalnie maly wzgledem typowych wartosci
#: w macierzach decyzyjnych, a jednoczesnie nie generowac inf/NaN.
DEFAULT_EPSILON = 1e-12


def safe_denominator(values: np.ndarray, eps: float = DEFAULT_EPSILON) -> np.ndarray:
    """
    Zwraca kopie `values`, w ktorej kazda wartosc rowna dokladnie 0
    zostala zastapiona przez `eps`. Nie modyfikuje tablicy wejsciowej.

    Uzyteczne, gdy mianownik jest potrzebny osobno (np. do dalszych
    obliczen posrednich, jak rozpietosc kryterium w ELECTRE), a nie
    tylko do jednego dzielenia.
    """
    result = np.array(values, dtype=float, copy=True)
    result[result == 0] = eps
    return result


def safe_divide(
    numerator: np.ndarray, denominator: np.ndarray, eps: float = DEFAULT_EPSILON
) -> np.ndarray:
    """
    Dzieli `numerator / denominator` element po elemencie, zastepujac
    zerowe mianowniki przez `eps` zamiast dopuszczac do dzielenia
    przez zero (co w numpy dalo by inf/NaN zamiast czytelnego bledu
    lub -- gorzej -- cichego zepsucia dalszych obliczen rankingu).
    """
    return numerator / safe_denominator(denominator, eps)
