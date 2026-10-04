"""
Testy jednostkowe dla mcdm/visualization/_utils.py -- wspolnych
narzedzi warstwy wizualizacji (konwersja figure -> base64/bajty/plik).

Do tej pory te funkcje byly cwiczone wylacznie posrednio, w ramach
testow radar.py/gaia.py/diff_report.py -- ten plik testuje je wprost,
w izolacji od konkretnych wykresow.
"""

import base64
import io

import matplotlib.pyplot as plt
import pytest
from PIL import Image

from mcdm.visualization._utils import (
    figure_to_base64,
    figure_to_png_bytes,
    save_figure,
)


def make_simple_figure():
    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 4])
    return fig


# ----------------------------------------------------------------------
# figure_to_base64
# ----------------------------------------------------------------------


def test_figure_to_base64_returns_decodable_png():
    encoded = figure_to_base64(make_simple_figure())
    assert isinstance(encoded, str)
    decoded = base64.b64decode(encoded)
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_figure_to_base64_closes_the_figure():
    fig = make_simple_figure()
    fig_num = fig.number
    figure_to_base64(fig)
    assert not plt.fignum_exists(fig_num)


def test_figure_to_base64_respects_dpi_parameter():
    low_dpi = base64.b64decode(figure_to_base64(make_simple_figure(), dpi=50))
    high_dpi = base64.b64decode(figure_to_base64(make_simple_figure(), dpi=200))
    img_low = Image.open(io.BytesIO(low_dpi))
    img_high = Image.open(io.BytesIO(high_dpi))
    assert img_high.size[0] > img_low.size[0]


# ----------------------------------------------------------------------
# figure_to_png_bytes
# ----------------------------------------------------------------------


def test_figure_to_png_bytes_returns_valid_png():
    data = figure_to_png_bytes(make_simple_figure())
    assert isinstance(data, bytes)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_figure_to_png_bytes_closes_the_figure():
    fig = make_simple_figure()
    fig_num = fig.number
    figure_to_png_bytes(fig)
    assert not plt.fignum_exists(fig_num)


def test_figure_to_png_bytes_is_loadable_by_pillow():
    data = figure_to_png_bytes(make_simple_figure())
    img = Image.open(io.BytesIO(data))
    img.verify()


# ----------------------------------------------------------------------
# save_figure
# ----------------------------------------------------------------------


def test_save_figure_creates_file_at_given_path(tmp_path):
    target = tmp_path / "chart.png"
    returned_path = save_figure(make_simple_figure(), target)
    assert returned_path == target
    assert target.exists()
    assert target.stat().st_size > 0


def test_save_figure_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "nested" / "deeper" / "chart.png"
    save_figure(make_simple_figure(), target)
    assert target.exists()


def test_save_figure_accepts_string_path(tmp_path):
    target = str(tmp_path / "chart_str.png")
    returned_path = save_figure(make_simple_figure(), target)
    assert returned_path.exists()


def test_save_figure_closes_the_figure(tmp_path):
    fig = make_simple_figure()
    fig_num = fig.number
    save_figure(fig, tmp_path / "chart.png")
    assert not plt.fignum_exists(fig_num)


def test_save_figure_output_is_valid_png(tmp_path):
    target = tmp_path / "chart.png"
    save_figure(make_simple_figure(), target)
    with Image.open(target) as img:
        img.verify()
