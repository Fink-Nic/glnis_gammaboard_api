"""Slice plots retain sampler-specific values and tolerate undefined ratios."""

from types import SimpleNamespace
import warnings

import numpy as np
import pytest
from matplotlib.figure import Figure

from glnis.scripts.slice_rendering import plot_slices


def slice_data(dimension=1, *, labels=("small flow", "large/flow")):
    grid = [SimpleNamespace(start=0.0, stop=1.0, num=5)]
    if dimension == 2:
        grid.append(SimpleNamespace(start=-1.0, stop=1.0, num=3))
    size = int(np.prod([axis.num for axis in grid]))
    function = np.linspace(-1, 1, size)
    return SimpleNamespace(
        name="Sine slice", origin=np.zeros(2), discrete=[], dirs=list(np.eye(2)[:dimension]), grid=grid,
        func_val=function, func_vals={label: function * (index + 1) for index, label in enumerate(labels)},
        prob={label: np.full(size, index + 1.0) for index, label in enumerate(labels)},
    )


def data(*slices, normalized=False, **plotting):
    return SimpleNamespace(
        slices1d=[slice_ for slice_ in slices if len(slice_.grid) == 1],
        slices2d=[slice_ for slice_ in slices if len(slice_.grid) == 2],
        settings={"slice_plot": {"plotting": {"dpi": 35, **plotting}}},
        itg=2.0 if normalized else None,
        output_file=None,
    )


def test_1d_uses_model_function_arrays_without_double_normalization(tmp_path, monkeypatch):
    slice_ = slice_data()
    captured = {}
    original = Figure.savefig

    def capture(figure, path, **kwargs):
        if path.name.endswith("comparison.png"):
            captured["lines"] = [(line.get_label(), line.get_ydata().copy()) for line in figure.axes[0].lines]
            captured["ratios"] = [line.get_ydata().copy() for line in figure.axes[1].lines]
        return original(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = plot_slices(data(slice_, normalized=True), tmp_path)
    assert [path.name for path in paths] == [
        "slices_slice1d_000_sampler_000.png", "slices_slice1d_000_sampler_001.png",
        "slices_slice1d_000_comparison.png",
    ]
    np.testing.assert_array_equal(captured["lines"][0][1], np.abs(slice_.func_vals["small flow"]))
    np.testing.assert_array_equal(captured["lines"][2][1], np.abs(slice_.func_vals["large/flow"]))
    assert "I_abs" in captured["lines"][0][0]
    np.testing.assert_array_equal(captured["ratios"][0], captured["ratios"][1])
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in paths)
    assert all(path.parent == tmp_path for path in paths)


def test_2d_grid_orientation_and_shared_comparison_scales(tmp_path, monkeypatch):
    slice_ = slice_data(2)
    captured = {}
    original = Figure.savefig

    def capture(figure, path, **kwargs):
        captured["images"] = [axis.images[0].get_array().copy() for axis in figure.axes if axis.images]
        captured["extents"] = [axis.images[0].get_extent() for axis in figure.axes if axis.images]
        captured["norms"] = [axis.images[0].norm for axis in figure.axes if axis.images]
        captured["titles"] = [axis.get_title() for axis in figure.axes[:4]]
        return original(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = plot_slices(data(slice_, plot_singles=False), tmp_path)
    assert len(paths) == 1
    assert paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(captured["images"][0], np.abs(slice_.func_val).reshape(3, 5))
    np.testing.assert_array_equal(captured["images"][4], 2 * np.abs(slice_.func_val).reshape(3, 5))
    assert captured["extents"][0] == [0.0, 1.0, -1.0, 1.0]
    assert captured["norms"][0] is captured["norms"][4]
    assert captured["titles"][0] == "|f|"


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("value", [0.0, np.nan])
def test_zero_or_masked_values_and_missing_target_render_without_warnings(tmp_path, dimension, value):
    slice_ = slice_data(dimension, labels=("one run",))
    slice_.func_val[:] = value
    slice_.func_vals["one run"][:] = value
    slice_.prob["one run"][:] = value
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        paths = plot_slices(data(slice_, plot_comparison=False), tmp_path)
    assert len(paths) == 1
    assert paths[0].is_file()


def test_skips_absent_or_incompatible_densities_and_accepts_missing_function(tmp_path):
    missing = slice_data()
    missing.prob = None
    incompatible = slice_data()
    incompatible.prob = {"wrong grid": np.ones(2)}
    valid = slice_data(labels=("arbitrary../label",))
    valid.func_val = None
    valid.func_vals = {}
    paths = plot_slices(data(missing, incompatible, valid, plot_comparison=False), tmp_path)
    assert len(paths) == 1
    assert "_002_sampler_001.png" in paths[0].name


def test_disabled_plots_and_unsupported_input(tmp_path):
    assert plot_slices(data(slice_data(), plot_singles=False, plot_comparison=False), tmp_path) == []
    with pytest.raises(TypeError, match="SlicePlotData"):
        plot_slices(object(), tmp_path)
