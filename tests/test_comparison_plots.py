"""Plots retain every named run, including zeros and missing loss values."""

from types import SimpleNamespace

import numpy as np
from matplotlib.figure import Figure

from glnis.scripts.comparison_plots import plot_sampler_comp
from glnis.utils.types import Result


def progress(samples=(), losses=(), *, rsds=(), abs_rsds=(), tvars=()):
    return SimpleNamespace(
        nspl_losses=list(samples), losses=list(losses),
        nspl_snapshot=list(samples), steps_losses=list(range(1, len(samples) + 1)),
        steps_snapshot=list(range(1, len(samples) + 1)),
        rsds=list(rsds), abs_rsds=list(abs_rsds), tvars=list(tvars),
    )


def comparison(histories, results, target=None):
    return SimpleNamespace(
        training_progress=histories, result=results, target=target,
        settings={"comparison": {"run_name": "Two batch sizes"}},
        run_configs={label: {"sampler": {"seed": index}, "integrand": {"kind": "sine"},
                             "aggregator": {"training_phase": "real"}}
                     for index, label in enumerate(results)},
    )


def test_named_training_curves_and_zero_error_results_are_preserved(tmp_path, monkeypatch):
    data = comparison(
        {"Small batch": progress([4, 8], [1, None], rsds=[0, 0], abs_rsds=[0, 0], tvars=[0, 0]),
         "Large batch": progress([7, 14], [0, -1], rsds=[2, 1], abs_rsds=[1, 0.5], tvars=[0.4, 0.2])},
        {"Small batch": Result.from_kwargs(n_points=20, total_time=1, real_mean=0),
         "Large batch": Result.from_kwargs(n_points=30, total_time=2, real_mean=1, real_error=0.1)},
        target=Result.from_kwargs(real_mean=0),
    )
    captured = {}
    original = Figure.savefig

    def capture(figure, filename, **kwargs):
        title_bounds = figure.texts[0].get_window_extent(figure.canvas.get_renderer())
        assert figure.bbox.contains(title_bounds.x0, title_bounds.y0)
        assert figure.bbox.contains(title_bounds.x1, title_bounds.y1)
        if filename.name.endswith("training_prog.png"):
            captured["loss"] = {line.get_label(): (line.get_xdata().copy(), line.get_ydata().copy(), line.get_color())
                                for line in figure.axes[0].lines}
            captured["loss_scale"] = figure.axes[0].get_yscale()
            captured["training_axes"] = len(figure.axes)
        else:
            captured["means"] = {container.get_label(): (container.lines[0].get_ydata(), container.lines[0].get_color())
                                 for container in figure.axes[0].containers}
            captured["reference"] = [line.get_ydata() for line in figure.axes[0].lines
                                     if line.get_label() == "Reference"]
        return original(figure, filename, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    artifacts = plot_sampler_comp(tmp_path / "comparison.pkl", data, "Review run", {"dpi": 45})
    assert [path.name for path in artifacts] == [
        "comparison_summary.txt", "comparison_training_prog.png", "comparison_integration_result.png",
    ]
    for path in artifacts[1:]:
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    np.testing.assert_array_equal(captured["loss"]["Small batch"][0], [4, 8])
    np.testing.assert_array_equal(captured["loss"]["Large batch"][0], [7, 14])
    assert np.isnan(captured["loss"]["Small batch"][1][1])
    assert captured["loss_scale"] == "linear"
    assert captured["training_axes"] == 4  # No ambiguous shared training-step axis.
    assert captured["means"]["Small batch"][0] == [0]
    assert captured["reference"] == [[0, 0]]
    for label in data.result:
        assert captured["means"][label][1] == captured["loss"][label][2]
    summary = artifacts[0].read_text()
    assert "Review run" in summary
    assert "Training samples: 8" in summary
    assert "Training samples: 14" in summary
    assert "Inference samples: 20" in summary
    assert "'seed': 1" in summary


def test_inference_only_single_run_without_reference(tmp_path):
    data = comparison({"Untrained": progress()}, {"Untrained": Result.from_kwargs(n_points=2, real_mean=0)})
    artifacts = plot_sampler_comp(tmp_path / "infer.pkl", data, plotting_settings={"dpi": 35})
    assert [path.name for path in artifacts] == ["infer_summary.txt", "infer_integration_result.png"]
    assert all(path.is_file() for path in artifacts)
    assert "Reference:" not in artifacts[0].read_text()


def test_training_only_does_not_plot_an_empty_inference_result(tmp_path):
    data = comparison(
        {"Train only": progress([4], [None], rsds=[0], abs_rsds=[0], tvars=[0])},
        {"Train only": Result()},
    )
    artifacts = plot_sampler_comp(tmp_path / "train.pkl", data, plotting_settings={"dpi": 35})
    assert [path.name for path in artifacts] == ["train_summary.txt", "train_training_prog.png"]
    assert "Inference samples: 0" in artifacts[0].read_text()
