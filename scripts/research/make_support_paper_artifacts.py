"""Generate paper figures, tables, and compact source-data extracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEXT = "#111111"
BACKGROUND = "#FFFFFF"
GRID = "#D1D5DB"
CORRECTED = "#0072B2"
NAIVE = "#D55E00"
PARTIAL = "#009E73"
NU_COLORS = ["#332288", "#117733", "#44AA99", "#DDCC77", "#CC6677", "#882255"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "legend.labelcolor": TEXT,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def asymptotic_scale(bandwidth: np.ndarray, smoothness: float) -> np.ndarray:
    if smoothness < 1.0:
        return bandwidth ** (2.0 * smoothness)
    if smoothness == 1.0:
        return bandwidth**2 * np.log(1.0 / bandwidth)
    return bandwidth**2


def save_figure(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    figure.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(figure)


def phase_figure(data: pd.DataFrame, output: Path) -> None:
    required = {"leading_shift", "coefficient_ratio"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"phase data are missing theorem-comparison columns: {sorted(missing)}")
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    shift_axis, ratio_axis = axes
    for color, (smoothness, group) in zip(
        NU_COLORS,
        data.sort_values("smoothness").groupby("smoothness", sort=True),
    ):
        group = group.sort_values("bandwidth")
        bandwidth = group["bandwidth"].to_numpy()
        shift = group["decay_shift"].to_numpy()
        label = rf"$\nu={smoothness:g}$"
        shift_axis.plot(
            bandwidth,
            shift,
            marker="o",
            markersize=3.8,
            linewidth=1.7,
            color=color,
            label=label,
        )
        shift_axis.plot(
            bandwidth,
            group["leading_shift"].to_numpy(),
            linestyle="--",
            linewidth=1.0,
            color=color,
            alpha=0.85,
        )
        ratio_axis.plot(
            bandwidth,
            group["coefficient_ratio"].to_numpy(),
            marker="o",
            markersize=3.2,
            linewidth=1.4,
            color=color,
            label=label,
        )
    shift_axis.set_xscale("log")
    shift_axis.set_yscale("log")
    shift_axis.set_xlabel(r"Bandwidth $h$")
    shift_axis.set_ylabel(r"Decay shift $\alpha-\alpha_h^\dagger$")
    shift_axis.set_title("Exact shift and leading term")
    shift_axis.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.75)
    ratio_axis.axhline(1.0, color=TEXT, linestyle=":", linewidth=1.0)
    ratio_axis.set_xscale("log")
    ratio_axis.set_xlabel(r"Bandwidth $h$")
    ratio_axis.set_ylabel("Exact shift / leading term")
    ratio_axis.set_title("Coefficient convergence")
    ratio_axis.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.75)
    handles, labels = ratio_axis.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=6,
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
    )
    legend.get_frame().set_alpha(1.0)
    figure.suptitle("Two-dimensional Matérn support-bias phase law", color=TEXT)
    figure.subplots_adjust(bottom=0.22, top=0.84, wspace=0.32)
    save_figure(figure, output / "phase_law")


def transition_figure(data: pd.DataFrame, output: Path) -> None:
    required = {
        "smoothness",
        "bandwidth",
        "exact_to_leading_ratio",
        "transition_relative_error",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"transition data are missing columns: {sorted(missing)}")
    bandwidths = sorted(data["bandwidth"].unique())
    if len(bandwidths) > len(NU_COLORS):
        raise ValueError("transition figure has more bandwidths than declared colors")
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.55))
    leading_axis, transition_axis = axes
    for color, bandwidth in zip(NU_COLORS, bandwidths, strict=False):
        group = data[data["bandwidth"] == bandwidth].sort_values("smoothness")
        label = rf"$h={bandwidth:g}$"
        for index, segment in enumerate(
            (
                group[group["smoothness"] < 1.0],
                group[group["smoothness"] > 1.0],
            )
        ):
            leading_axis.plot(
                segment["smoothness"],
                segment["exact_to_leading_ratio"],
                color=color,
                marker="o",
                markersize=3.0,
                linewidth=1.5,
                label=label if index == 0 else None,
            )
        threshold = group[group["smoothness"] == 1.0]
        leading_axis.scatter(
            threshold["smoothness"],
            threshold["exact_to_leading_ratio"],
            color=color,
            marker="D",
            s=23,
            zorder=3,
        )
        transition_error_percent = 100.0 * group[
            "transition_relative_error"
        ].to_numpy()
        if np.any(transition_error_percent <= 0):
            raise ValueError("transition relative errors must be strictly positive")
        transition_axis.plot(
            group["smoothness"],
            transition_error_percent,
            color=color,
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            label=label,
        )
    leading_axis.axhline(1.0, color=TEXT, linestyle=":", linewidth=1.0)
    leading_axis.axvline(1.0, color=GRID, linestyle="--", linewidth=1.0)
    leading_axis.set_xlabel(r"Matérn smoothness $\nu$")
    leading_axis.set_ylabel("Exact shift / one-term shift")
    leading_axis.set_title("One-term law is nonuniform")
    leading_axis.grid(True, color=GRID, linewidth=0.6, alpha=0.75)
    transition_axis.axvline(1.0, color=GRID, linestyle="--", linewidth=1.0)
    transition_axis.set_yscale("log")
    transition_axis.set_xlabel(r"Matérn smoothness $\nu$")
    transition_axis.set_ylabel("Two-term relative error (%)")
    transition_axis.set_title("Cancellation-aware approximation")
    transition_axis.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.75)
    handles, labels = transition_axis.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
    )
    legend.get_frame().set_alpha(1.0)
    figure.suptitle("Finite-bandwidth stress audit around smoothness one")
    figure.subplots_adjust(bottom=0.23, top=0.82, wspace=0.34)
    save_figure(figure, output / "transition_stress")


def finite_figure(summary: pd.DataFrame, output: Path) -> None:
    core = summary[
        summary["config_id"].str.match(r"d2_nu(05|10|15|25)_h(00|03|05|07)")
    ].copy()
    smoothness_values = sorted(core["smoothness"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(7.4, 6.0), sharex=True, sharey=True)
    for axis, smoothness in zip(axes.ravel(), smoothness_values):
        panel = core[core["smoothness"] == smoothness]
        for model, color, marker in [
            ("corrected", CORRECTED, "o"),
            ("naive", NAIVE, "s"),
        ]:
            values = panel[panel["model"] == model].sort_values("bandwidth")
            bandwidth = values["bandwidth"].to_numpy()
            target = values["population_target"].to_numpy()
            mean = values["mean_estimate"].to_numpy()
            error = 2.0 * values["monte_carlo_standard_error_mean"].to_numpy()
            axis.plot(
                bandwidth,
                target,
                color=color,
                linewidth=1.8,
                label=f"{model.capitalize()} target",
            )
            axis.errorbar(
                bandwidth,
                mean,
                yerr=error,
                color=color,
                marker=marker,
                markersize=4.2,
                linestyle="none",
                capsize=2.5,
                label=f"{model.capitalize()} mean",
            )
        axis.axhline(1.0, color=TEXT, linestyle=":", linewidth=1.0)
        axis.set_title(rf"Smoothness $\nu={smoothness:g}$")
        axis.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
        axis.set_ylim(bottom=0.0)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"Bandwidth $h$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"Decay parameter $\alpha$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=4,
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
    )
    legend.get_frame().set_alpha(1.0)
    figure.suptitle("Finite-grid full-likelihood targets and estimates", color=TEXT)
    figure.subplots_adjust(bottom=0.14, top=0.91, hspace=0.22, wspace=0.16)
    save_figure(figure, output / "finite_targets")


def convergence_figure(summary: pd.DataFrame, output: Path) -> None:
    data = summary[summary["config_id"].str.startswith("d1_n")].copy()
    data["number_of_inputs"] = pd.to_numeric(data["number_of_inputs"])
    figure, axis = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for model, color, marker in [
        ("corrected", CORRECTED, "o"),
        ("naive", NAIVE, "s"),
    ]:
        values = data[data["model"] == model].sort_values("number_of_inputs")
        n = values["number_of_inputs"].to_numpy()
        target = values["population_target"].to_numpy()
        mean = values["mean_estimate"].to_numpy()
        error = 2.0 * values["monte_carlo_standard_error_mean"].to_numpy()
        axis.plot(n, target, color=color, linewidth=1.8, label=f"{model.capitalize()} target")
        axis.errorbar(
            n,
            mean,
            yerr=error,
            color=color,
            marker=marker,
            linestyle="none",
            capsize=2.5,
            label=f"{model.capitalize()} mean",
        )
    axis.axhline(1.0, color=TEXT, linestyle=":", linewidth=1.0)
    axis.set_xlabel("Number of raw lattice sites")
    axis.set_ylabel(r"Decay parameter $\alpha$")
    axis.set_title(r"Increasing-domain check ($\nu=0.5$, $h=0.4$)")
    axis.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    legend = axis.legend(frameon=True, facecolor=BACKGROUND, edgecolor=GRID, ncol=2)
    legend.get_frame().set_alpha(1.0)
    save_figure(figure, output / "finite_convergence")


def anisotropy_figure(data: pd.DataFrame, output: Path) -> None:
    required = {
        "aspect_ratio",
        "angle_degrees",
        "smoothness",
        "bandwidth",
        "decay_shift",
        "implied_range_ratio",
        "directional_contrast_coefficient_to_minor",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"anisotropy data are missing columns: {sorted(missing)}")
    aspect_ratio = float(data["aspect_ratio"].max())
    selected = data[data["aspect_ratio"] == aspect_ratio].copy()
    maximum_bandwidth = float(selected["bandwidth"].max())
    figure, axes = plt.subplots(1, 2, figsize=(7.6, 3.8))
    angle_axis, contrast_axis = axes
    for color, (smoothness, group) in zip(
        NU_COLORS,
        selected.sort_values("smoothness").groupby("smoothness", sort=True),
    ):
        angular = group[group["bandwidth"] == maximum_bandwidth].sort_values(
            "angle_degrees"
        )
        label = rf"$\nu={smoothness:g}$"
        angle_axis.plot(
            angular["angle_degrees"],
            100.0 * (angular["implied_range_ratio"] - 1.0),
            color=color,
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            label=label,
        )
        major = group[group["angle_degrees"] == 0.0].sort_values("bandwidth")
        minor = group[group["angle_degrees"] == 90.0].sort_values("bandwidth")
        if major.empty or minor.empty:
            raise ValueError("anisotropy data must contain zero- and 90-degree axes")
        np.testing.assert_allclose(major["bandwidth"], minor["bandwidth"])
        coefficient = major[
            "directional_contrast_coefficient_to_minor"
        ].to_numpy()
        contrast = major["decay_shift"].to_numpy() - minor["decay_shift"].to_numpy()
        ratio = contrast / (coefficient * major["bandwidth"].to_numpy() ** 2)
        contrast_axis.plot(
            major["bandwidth"],
            ratio,
            color=color,
            marker="o",
            markersize=3.0,
            linewidth=1.5,
            label=label,
        )
    angle_axis.set_xlabel(r"Lag angle (degrees)")
    angle_axis.set_ylabel("Inferred range inflation (%)")
    angle_axis.set_title(
        rf"Aspect $\varrho={aspect_ratio:g}$ at $h={maximum_bandwidth:g}$"
    )
    angle_axis.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    contrast_axis.axhline(1.0, color=TEXT, linestyle=":", linewidth=1.0)
    contrast_axis.set_xscale("log")
    contrast_axis.set_xlabel(r"Bandwidth $h$")
    contrast_axis.set_ylabel(r"Exact contrast / $(D_\nu h^2)$")
    contrast_axis.set_title("Directional coefficient check")
    contrast_axis.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.8)
    handles, labels = contrast_axis.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=len(labels),
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
    )
    legend.get_frame().set_alpha(1.0)
    figure.suptitle("Directional range shift from elongated observation support")
    figure.subplots_adjust(bottom=0.22, top=0.84, wspace=0.32)
    save_figure(figure, output / "anisotropic_support")


def likelihood_projection_figure(
    multilag: pd.DataFrame,
    full_likelihood: pd.DataFrame,
    output: Path,
) -> None:
    """Show lag sensitivity and convergence for the two new likelihood results."""
    multilag_required = {
        "smoothness",
        "bandwidth",
        "lag",
        "pair_shift_coefficient",
        "composite_shift_coefficient",
        "composite_shift_ratio",
        "minimum_kl_ratio",
    }
    full_required = {
        "smoothness",
        "bandwidth",
        "decay_shift_ratio",
        "minimum_kl_ratio",
    }
    if missing := multilag_required.difference(multilag.columns):
        raise ValueError(f"multi-lag data are missing columns: {sorted(missing)}")
    if missing := full_required.difference(full_likelihood.columns):
        raise ValueError(f"full-likelihood data are missing columns: {sorted(missing)}")

    smoothness_values = sorted(multilag["smoothness"].unique())
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.45))
    lag_axis, composite_axis, full_axis = axes
    for color, smoothness in zip(NU_COLORS, smoothness_values, strict=False):
        group = multilag[multilag["smoothness"] == smoothness]
        coefficients = group.drop_duplicates("lag").sort_values("lag")
        label = rf"$\nu={smoothness:g}$"
        lag_axis.plot(
            coefficients["lag"],
            coefficients["pair_shift_coefficient"],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            label=label,
        )
        composite_coefficient = float(group["composite_shift_coefficient"].iloc[0])
        lag_axis.hlines(
            composite_coefficient,
            float(coefficients["lag"].min()),
            float(coefficients["lag"].max()),
            color=color,
            linestyle=":",
            linewidth=1.0,
        )
        composite_cells = group.drop_duplicates("bandwidth").sort_values("bandwidth")
        composite_axis.plot(
            composite_cells["bandwidth"],
            composite_cells["composite_shift_ratio"],
            color=color,
            marker="o",
            markersize=3.2,
            linewidth=1.4,
        )
        composite_axis.plot(
            composite_cells["bandwidth"],
            composite_cells["minimum_kl_ratio"],
            color=color,
            marker="s",
            markersize=3.0,
            linestyle="--",
            linewidth=1.1,
        )
        full_cells = full_likelihood[
            full_likelihood["smoothness"] == smoothness
        ].sort_values("bandwidth")
        full_axis.plot(
            full_cells["bandwidth"],
            full_cells["decay_shift_ratio"],
            color=color,
            marker="o",
            markersize=3.2,
            linewidth=1.4,
        )
        full_axis.plot(
            full_cells["bandwidth"],
            full_cells["minimum_kl_ratio"],
            color=color,
            marker="s",
            markersize=3.0,
            linestyle="--",
            linewidth=1.1,
        )

    lag_axis.set_xlabel(r"Dimensionless lag $\alpha R$")
    lag_axis.set_ylabel(r"Pair coefficient $C_\nu(R)$")
    lag_axis.set_title("Lag-specific shifts")
    lag_axis.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    lag_axis.legend(
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
        fontsize=7.5,
        ncol=2,
    )
    for axis, title in (
        (composite_axis, "Multi-lag composite"),
        (full_axis, "Full Gaussian likelihood"),
    ):
        axis.axhline(1.0, color=TEXT, linestyle=":", linewidth=1.0)
        axis.set_xscale("log")
        axis.set_xlabel(r"Bandwidth $h$")
        axis.set_ylabel("Exact / leading term")
        axis.set_title(title)
        axis.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.8)
        axis.plot([], [], color=TEXT, marker="o", linewidth=1.4, label="Decay shift")
        axis.plot(
            [], [], color=TEXT, marker="s", linestyle="--", linewidth=1.1,
            label="Minimum KL",
        )
        legend = axis.legend(
            frameon=True,
            facecolor=BACKGROUND,
            edgecolor=GRID,
            fontsize=7.5,
        )
        legend.get_frame().set_alpha(1.0)
    figure.suptitle("Genuine likelihood misspecification has the predicted local geometry")
    figure.subplots_adjust(bottom=0.18, top=0.82, wspace=0.36)
    save_figure(figure, output / "likelihood_projection")


def joint_smoothness_figure(summary: pd.DataFrame, output: Path) -> None:
    """Plot population targets and Monte Carlo medians in the joint library."""
    required = {
        "bandwidth",
        "smoothness_true",
        "model",
        "population_smoothness_target",
        "population_decay_target",
        "median_smoothness_estimate",
        "median_decay_estimate",
    }
    if missing := required.difference(summary.columns):
        raise ValueError(f"joint smoothness summary is missing columns: {sorted(missing)}")
    bandwidths = sorted(summary["bandwidth"].unique())
    if len(bandwidths) != 2:
        raise ValueError("the joint-smoothness paper figure requires exactly two bandwidths")
    styles = {
        "support_aware": (CORRECTED, "Support-aware"),
        "partial_support": (PARTIAL, "75% bandwidth"),
        "point_support": (NAIVE, "Point-support"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.7, 3.85), sharex=True, sharey=True)
    for axis, bandwidth in zip(axes, bandwidths, strict=True):
        panel = summary[np.isclose(summary["bandwidth"], bandwidth)]
        for model, (color, _) in styles.items():
            group = panel[panel["model"] == model].sort_values("smoothness_true")
            for row in group.itertuples(index=False):
                axis.plot(
                    [row.smoothness_true, row.population_smoothness_target],
                    [1.0, row.population_decay_target],
                    color=color,
                    linewidth=1.0,
                    alpha=0.65,
                )
            axis.scatter(
                group["population_smoothness_target"],
                group["population_decay_target"],
                color=color,
                marker="o",
                s=27,
                zorder=3,
            )
            axis.scatter(
                group["median_smoothness_estimate"],
                group["median_decay_estimate"],
                color=color,
                marker="x",
                s=29,
                linewidths=1.2,
                zorder=4,
            )
        truths = sorted(panel["smoothness_true"].unique())
        axis.scatter(
            truths,
            np.ones(len(truths)),
            color=TEXT,
            marker="*",
            s=42,
            zorder=5,
        )
        axis.axhline(1.0, color=GRID, linestyle=":", linewidth=0.9)
        axis.set_title(rf"True bandwidth $h={bandwidth:g}$")
        axis.set_xlabel(r"Fitted smoothness $\nu$")
        axis.grid(True, color=GRID, linewidth=0.6, alpha=0.65)
    axes[0].set_ylabel(r"Fitted inverse range $\alpha$")
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=color, marker="o", linewidth=1.1, label=label)
        for color, label in styles.values()
    ]
    handles.extend(
        [
            Line2D([0], [0], color=TEXT, marker="*", linestyle="none", label="Truth"),
            Line2D([0], [0], color=TEXT, marker="x", linestyle="none", label="MC median"),
        ]
    )
    legend = figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=5,
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
        fontsize=7.5,
    )
    legend.get_frame().set_alpha(1.0)
    figure.suptitle("Ignored support moves both Matérn smoothness and range")
    figure.subplots_adjust(bottom=0.22, top=0.82, wspace=0.12)
    save_figure(figure, output / "joint_smoothness_targets")


def raw_support_figure(data: pd.DataFrame, output: Path) -> None:
    required = {"field_stage", "replicate", "x", "y", "value", "bandwidth"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"raw example data are missing columns: {sorted(missing)}")
    replicate = int(data["replicate"].min())
    selected = data[data["replicate"] == replicate]
    stages = ["latent_input", "averaged_output"]
    if set(stages).difference(selected["field_stage"].unique()):
        raise ValueError("raw example must contain latent and averaged field stages")
    magnitude = float(np.max(np.abs(selected["value"])))
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), constrained_layout=True)
    image = None
    for axis, stage, title in zip(
        axes,
        stages,
        ("Latent Matérn field", "Recorded local averages"),
    ):
        values = selected[selected["field_stage"] == stage]
        pivot = values.pivot(index="y", columns="x", values="value").sort_index()
        image = axis.imshow(
            pivot.to_numpy(),
            origin="lower",
            cmap="coolwarm",
            vmin=-magnitude,
            vmax=magnitude,
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel(r"Coordinate $s_1$")
        axis.set_ylabel(r"Coordinate $s_2$")
        axis.set_facecolor(BACKGROUND)
    if image is None:
        raise RuntimeError("raw support figure did not create an image")
    colorbar = figure.colorbar(image, ax=axes, shrink=0.88, pad=0.025)
    colorbar.set_label("Field value")
    bandwidth = float(selected["bandwidth"].iloc[0])
    figure.suptitle(
        rf"One SupportShift realization before and after averaging ($h={bandwidth:g}$)"
    )
    save_figure(figure, output / "supportshift_raw_example")


def highdim_figure(data: pd.DataFrame, output: Path) -> pd.DataFrame:
    required = {
        "config_id",
        "model",
        "sample_size",
        "dimension_p",
        "smoothness",
        "decay_estimate",
        "decay_true",
        "decay_error_to_grid_target",
        "population_grid_decay_target",
        "max_abs_criterion_deviation",
        "uniform_likelihood_bound",
        "uniform_bound_holds",
        "simultaneous_candidatewise_bound_holds",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"high-dimensional data are missing columns: {sorted(missing)}")
    summary_records: list[dict[str, float | int | str]] = []
    group_columns = [
        "config_id",
        "model",
        "sample_size",
        "dimension_p",
        "smoothness",
    ]
    for keys, group in data.groupby(group_columns, sort=True):
        errors = group["decay_error_to_grid_target"].to_numpy()
        summary_records.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "effective_information": int(
                    group["sample_size"].iloc[0] * group["dimension_p"].iloc[0]
                ),
                "criterion_deviation_q95": float(
                    group["max_abs_criterion_deviation"].quantile(0.95)
                ),
                "uniform_likelihood_bound": float(
                    group["uniform_likelihood_bound"].iloc[0]
                ),
                "worst_envelope_coverage": float(
                    group["uniform_bound_holds"].mean()
                ),
                "candidatewise_simultaneous_coverage": float(
                    group["simultaneous_candidatewise_bound_holds"].mean()
                ),
                "decay_rmse_to_grid_target": float(np.sqrt(np.mean(errors**2))),
                "decay_mean": float(group["decay_estimate"].mean()),
                "decay_q10": float(group["decay_estimate"].quantile(0.10)),
                "decay_q90": float(group["decay_estimate"].quantile(0.90)),
                "decay_true": float(group["decay_true"].iloc[0]),
                "population_grid_decay_target": float(
                    group["population_grid_decay_target"].iloc[0]
                ),
            }
        )
    summary = pd.DataFrame.from_records(summary_records)
    figure, axes = plt.subplots(1, 3, figsize=(10.4, 3.55))
    certificate_axis, rmse_axis, target_axis = axes
    model_styles = {
        "corrected": (CORRECTED, "o", "Support-aware"),
        "naive": (NAIVE, "s", "Point-support"),
    }
    for model, (color, marker, label) in model_styles.items():
        values = summary[summary["model"] == model]
        certificate_axis.scatter(
            values["uniform_likelihood_bound"],
            values["criterion_deviation_q95"],
            color=color,
            marker=marker,
            s=25,
            alpha=0.82,
            label=label,
        )
        for (_, _, _), group in values.groupby(
            ["dimension_p", "smoothness", "model"], sort=True
        ):
            group = group.sort_values("effective_information")
            rmse_axis.plot(
                group["effective_information"],
                group["decay_rmse_to_grid_target"],
                color=color,
                marker=marker,
                markersize=3.0,
                linewidth=0.9,
                alpha=0.45,
            )
    certificate_min = float(
        min(summary["criterion_deviation_q95"].min(), summary["uniform_likelihood_bound"].min())
    )
    certificate_max = float(
        max(summary["criterion_deviation_q95"].max(), summary["uniform_likelihood_bound"].max())
    )
    certificate_axis.plot(
        [certificate_min, certificate_max],
        [certificate_min, certificate_max],
        color=TEXT,
        linestyle=":",
        linewidth=1.0,
        label="Equality",
    )
    certificate_axis.set_xscale("log")
    certificate_axis.set_yscale("log")
    certificate_axis.set_xlabel("Theorem radius")
    certificate_axis.set_ylabel("95th percentile max deviation")
    certificate_axis.set_title("Worst-radius envelope")
    certificate_axis.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.75)

    rmse_axis.set_xscale("log")
    rmse_axis.set_yscale("log")
    rmse_axis.set_xlabel(r"Effective information $Np$")
    rmse_axis.set_ylabel("Decay RMSE to KL grid target")
    rmse_axis.set_title("Stochastic error contracts")
    rmse_axis.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.75)

    maximum_dimension = int(summary["dimension_p"].max())
    selected_smoothness = float(summary["smoothness"].max())
    selected = summary[
        (summary["dimension_p"] == maximum_dimension)
        & (summary["smoothness"] == selected_smoothness)
    ]
    for model, (color, marker, label) in model_styles.items():
        values = selected[selected["model"] == model].sort_values("sample_size")
        target_axis.fill_between(
            values["sample_size"],
            values["decay_q10"],
            values["decay_q90"],
            color=color,
            alpha=0.15,
            linewidth=0.0,
        )
        target_axis.plot(
            values["sample_size"],
            values["decay_mean"],
            color=color,
            marker=marker,
            markersize=4.0,
            linewidth=1.6,
            label=f"{label} mean",
        )
        target = float(values["population_grid_decay_target"].iloc[0])
        target_axis.axhline(
            target,
            color=color,
            linestyle="--",
            linewidth=1.2,
            label=f"{label} KL target",
        )
    target_axis.axhline(
        float(selected["decay_true"].iloc[0]),
        color=TEXT,
        linestyle=":",
        linewidth=1.2,
        label="Physical decay",
    )
    target_axis.set_xscale("log", base=2)
    target_axis.set_xlabel(r"Independent fields $N$")
    target_axis.set_ylabel(r"Inverse range $\alpha$")
    target_axis.set_title(
        rf"More precise, wrong target ($p={maximum_dimension}$, $\nu={selected_smoothness:g}$)"
    )
    target_axis.grid(True, which="both", color=GRID, linewidth=0.6, alpha=0.75)
    handles, labels = target_axis.get_legend_handles_labels()
    legend = figure.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=3,
        frameon=True,
        facecolor=BACKGROUND,
        edgecolor=GRID,
    )
    legend.get_frame().set_alpha(1.0)
    figure.suptitle("High-dimensional SupportShift likelihood experiment")
    figure.subplots_adjust(bottom=0.27, top=0.82, wspace=0.38)
    save_figure(figure, output / "supportshift_highdim")
    return summary


def finite_table(summary: pd.DataFrame, path: Path) -> None:
    selected = summary[
        summary["config_id"].str.match(r"d2_nu(05|10|15|25)_h07")
    ].sort_values(["smoothness", "model"])
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"$\nu$ & Model & Target & Mean & MCSE & Bound hits\\",
        r"\midrule",
    ]
    for row in selected.itertuples(index=False):
        model = "Corrected" if row.model == "corrected" else "Naive"
        lines.append(
            f"{row.smoothness:g} & {model} & {row.population_target:.3f} & "
            f"{row.mean_estimate:.3f} & {row.monte_carlo_standard_error_mean:.3f} & "
            f"{int(row.boundary_fits)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def dimension_kernel_table(data: pd.DataFrame, path: Path) -> None:
    required = {
        "dimension",
        "kernel_family",
        "smoothness",
        "bandwidth",
        "decay_shift",
        "coefficient_ratio",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(
            f"dimension-kernel data are missing columns: {sorted(missing)}"
        )
    smallest_bandwidth = float(data["bandwidth"].min())
    selected = data[np.isclose(data["bandwidth"], smallest_bandwidth)]
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Kernel & $d$ & Minimum ratio & Maximum ratio\\",
        r"\midrule",
    ]
    for (kernel_family, dimension), group in selected.groupby(
        ["kernel_family", "dimension"], sort=True
    ):
        label = str(kernel_family).capitalize()
        lines.append(
            f"{label} & {int(dimension)} & "
            f"{group['coefficient_ratio'].min():.3f} & "
            f"{group['coefficient_ratio'].max():.3f}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def matched_boundary_table(data: pd.DataFrame, path: Path) -> None:
    """Write the matched-size boundary and intermediate-method comparison."""
    required = {
        "region", "output_dimension", "bandwidth", "smoothness", "model",
        "decay_target", "minimum_kl",
    }
    if missing := required.difference(data.columns):
        raise ValueError(f"matched boundary data are missing columns: {sorted(missing)}")
    selected = data[np.isclose(data["bandwidth"], data["bandwidth"].max())]
    if selected["output_dimension"].nunique() != 1:
        raise ValueError("matched boundary table requires a fixed output dimension")
    lookup = {
        (float(row.smoothness), str(row.region), str(row.model)): row
        for row in selected.itertuples(index=False)
    }
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\nu$ & $\alpha^*_{\rm point,I}$ & $\alpha^*_{\rm point,B}$ & "
        r"KL$_{.75h}$/KL$_{\rm point,I}$ & KL$_{.75h}$/KL$_{\rm point,B}$ & $p$\\",
        r"\midrule",
    ]
    for smoothness in sorted(selected["smoothness"].unique()):
        interior_point = lookup[(smoothness, "interior", "point_support")]
        boundary_point = lookup[(smoothness, "boundary", "point_support")]
        interior_partial = lookup[(smoothness, "interior", "partial_support")]
        boundary_partial = lookup[(smoothness, "boundary", "partial_support")]
        lines.append(
            f"{smoothness:g} & {interior_point.decay_target:.3f} & "
            f"{boundary_point.decay_target:.3f} & "
            f"{interior_partial.minimum_kl / interior_point.minimum_kl:.3f} & "
            f"{boundary_partial.minimum_kl / boundary_point.minimum_kl:.3f} & "
            f"{int(interior_point.output_dimension)}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_extract(
    source_path: Path,
    destination_path: Path,
) -> None:
    """Copy an immutable source table unless it already is the destination."""
    if source_path.resolve() == destination_path.resolve():
        return
    shutil.copyfile(source_path, destination_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--phase-metadata", type=Path, required=True)
    parser.add_argument("--transition-stress", type=Path)
    parser.add_argument("--transition-stress-metadata", type=Path)
    parser.add_argument("--dimension-kernel-robustness", type=Path)
    parser.add_argument("--dimension-kernel-robustness-metadata", type=Path)
    parser.add_argument("--finite-summary", type=Path, required=True)
    parser.add_argument("--finite-results", type=Path, required=True)
    parser.add_argument("--finite-audit", type=Path, required=True)
    parser.add_argument("--finite-manifest", type=Path, required=True)
    parser.add_argument("--anisotropy", type=Path)
    parser.add_argument("--anisotropy-metadata", type=Path)
    parser.add_argument("--raw-example", type=Path)
    parser.add_argument("--highdim", type=Path)
    parser.add_argument("--highdim-metadata", type=Path)
    parser.add_argument("--multilag", type=Path)
    parser.add_argument("--multilag-metadata", type=Path)
    parser.add_argument("--full-likelihood", type=Path)
    parser.add_argument("--full-likelihood-metadata", type=Path)
    parser.add_argument("--joint-smoothness", type=Path)
    parser.add_argument("--joint-smoothness-metadata", type=Path)
    parser.add_argument("--matched-boundary", type=Path)
    parser.add_argument("--matched-boundary-metadata", type=Path)
    parser.add_argument("--paper-directory", type=Path, required=True)
    args = parser.parse_args()
    input_paths = {
        name: path
        for name, path in {
            "phase": args.phase,
            "phase_metadata": args.phase_metadata,
            "transition_stress": args.transition_stress,
            "transition_stress_metadata": args.transition_stress_metadata,
            "dimension_kernel_robustness": args.dimension_kernel_robustness,
            "dimension_kernel_robustness_metadata": args.dimension_kernel_robustness_metadata,
            "finite_summary": args.finite_summary,
            "finite_results": args.finite_results,
            "finite_audit": args.finite_audit,
            "finite_manifest": args.finite_manifest,
            "anisotropy": args.anisotropy,
            "anisotropy_metadata": args.anisotropy_metadata,
            "raw_example": args.raw_example,
            "highdim": args.highdim,
            "highdim_metadata": args.highdim_metadata,
            "multilag": args.multilag,
            "multilag_metadata": args.multilag_metadata,
            "full_likelihood": args.full_likelihood,
            "full_likelihood_metadata": args.full_likelihood_metadata,
            "joint_smoothness": args.joint_smoothness,
            "joint_smoothness_metadata": args.joint_smoothness_metadata,
            "matched_boundary": args.matched_boundary,
            "matched_boundary_metadata": args.matched_boundary_metadata,
        }.items()
        if path is not None
    }
    paired_inputs = (
        ("transition_stress", "transition_stress_metadata"),
        ("dimension_kernel_robustness", "dimension_kernel_robustness_metadata"),
        ("anisotropy", "anisotropy_metadata"),
        ("highdim", "highdim_metadata"),
        ("multilag", "multilag_metadata"),
        ("full_likelihood", "full_likelihood_metadata"),
        ("joint_smoothness", "joint_smoothness_metadata"),
        ("matched_boundary", "matched_boundary_metadata"),
    )
    for data_name, metadata_name in paired_inputs:
        if (data_name in input_paths) != (metadata_name in input_paths):
            raise ValueError(
                f"{data_name} and {metadata_name} must be supplied together"
            )
    input_manifest = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in input_paths.items()
    }
    configure_matplotlib()
    figures = args.paper_directory / "figures"
    tables = args.paper_directory / "tables"
    data_directory = args.paper_directory / "data"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    data_directory.mkdir(parents=True, exist_ok=True)
    phase = pd.read_csv(args.phase)
    summary = pd.read_csv(args.finite_summary)
    phase_figure(phase, figures)
    if args.transition_stress is not None:
        transition_stress = pd.read_csv(args.transition_stress)
        transition_figure(transition_stress, figures)
        write_source_extract(
            args.transition_stress,
            data_directory / "transition_stress.csv",
        )
    if args.dimension_kernel_robustness is not None:
        dimension_kernel = pd.read_csv(args.dimension_kernel_robustness)
        dimension_kernel_table(
            dimension_kernel,
            tables / "dimension_kernel_robustness.tex",
        )
        write_source_extract(
            args.dimension_kernel_robustness,
            data_directory / "dimension_kernel_robustness.csv",
        )
    finite_figure(summary, figures)
    convergence_figure(summary, figures)
    finite_table(summary, tables / "finite_summary.tex")
    if args.anisotropy is not None:
        anisotropy = pd.read_csv(args.anisotropy)
        anisotropy_figure(anisotropy, figures)
        write_source_extract(
            args.anisotropy,
            data_directory / "anisotropic_phase.csv",
        )
    if args.raw_example is not None:
        raw_example = pd.read_csv(args.raw_example)
        raw_support_figure(raw_example, figures)
        write_source_extract(
            args.raw_example,
            data_directory / "supportshift_raw_example.csv",
        )
    if args.highdim is not None:
        highdim = pd.read_csv(args.highdim)
        highdim_summary = highdim_figure(highdim, figures)
        highdim_summary.to_csv(
            data_directory / "supportshift_highdim_summary.csv",
            index=False,
        )
    if args.multilag is not None and args.full_likelihood is not None:
        likelihood_projection_figure(
            pd.read_csv(args.multilag),
            pd.read_csv(args.full_likelihood),
            figures,
        )
        write_source_extract(
            args.multilag, data_directory / "multilag_composite.csv"
        )
        write_source_extract(
            args.full_likelihood, data_directory / "full_likelihood_phase.csv"
        )
    elif (args.multilag is None) != (args.full_likelihood is None):
        raise ValueError("multi-lag and full-likelihood inputs must be supplied together")
    if args.joint_smoothness is not None:
        joint_summary_path = args.joint_smoothness.with_name(
            f"{args.joint_smoothness.stem}.summary.csv"
        )
        joint_smoothness_figure(pd.read_csv(joint_summary_path), figures)
        write_source_extract(
            args.joint_smoothness, data_directory / "joint_smoothness.csv"
        )
        write_source_extract(
            joint_summary_path, data_directory / "joint_smoothness_summary.csv"
        )
    if args.matched_boundary is not None:
        matched_boundary = pd.read_csv(args.matched_boundary)
        matched_boundary_table(
            matched_boundary, tables / "matched_boundary.tex"
        )
        write_source_extract(
            args.matched_boundary, data_directory / "matched_boundary.csv"
        )
    write_source_extract(
        args.phase,
        data_directory / "phase_oracle_d2.csv",
    )
    write_source_extract(
        args.finite_summary,
        data_directory / "finite_summary.csv",
    )
    artifact_paths = [
        figures / "phase_law.pdf",
        figures / "phase_law.png",
        figures / "finite_targets.pdf",
        figures / "finite_targets.png",
        figures / "finite_convergence.pdf",
        figures / "finite_convergence.png",
        tables / "finite_summary.tex",
        data_directory / "phase_oracle_d2.csv",
        data_directory / "finite_summary.csv",
    ]
    if args.transition_stress is not None:
        artifact_paths.extend(
            [
                figures / "transition_stress.pdf",
                figures / "transition_stress.png",
                data_directory / "transition_stress.csv",
            ]
        )
    if args.dimension_kernel_robustness is not None:
        artifact_paths.extend(
            [
                tables / "dimension_kernel_robustness.tex",
                data_directory / "dimension_kernel_robustness.csv",
            ]
        )
    if args.anisotropy is not None:
        artifact_paths.extend(
            [
                figures / "anisotropic_support.pdf",
                figures / "anisotropic_support.png",
                data_directory / "anisotropic_phase.csv",
            ]
        )
    if args.raw_example is not None:
        artifact_paths.extend(
            [
                figures / "supportshift_raw_example.pdf",
                figures / "supportshift_raw_example.png",
                data_directory / "supportshift_raw_example.csv",
            ]
        )
    if args.highdim is not None:
        artifact_paths.extend(
            [
                figures / "supportshift_highdim.pdf",
                figures / "supportshift_highdim.png",
                data_directory / "supportshift_highdim_summary.csv",
            ]
        )
    if args.multilag is not None:
        artifact_paths.extend(
            [
                figures / "likelihood_projection.pdf",
                figures / "likelihood_projection.png",
                data_directory / "multilag_composite.csv",
                data_directory / "full_likelihood_phase.csv",
            ]
        )
    if args.joint_smoothness is not None:
        artifact_paths.extend(
            [
                figures / "joint_smoothness_targets.pdf",
                figures / "joint_smoothness_targets.png",
                data_directory / "joint_smoothness.csv",
                data_directory / "joint_smoothness_summary.csv",
            ]
        )
    if args.matched_boundary is not None:
        artifact_paths.extend(
            [
                tables / "matched_boundary.tex",
                data_directory / "matched_boundary.csv",
            ]
        )
    missing_artifacts = [str(path) for path in artifact_paths if not path.is_file()]
    if missing_artifacts:
        raise RuntimeError(f"artifact generation omitted expected files: {missing_artifacts}")
    output_aliases = {
        name: str(destination.relative_to(args.paper_directory))
        for name, destination in {
            "phase": data_directory / "phase_oracle_d2.csv",
            "transition_stress": data_directory / "transition_stress.csv",
            "dimension_kernel_robustness": data_directory
            / "dimension_kernel_robustness.csv",
            "finite_summary": data_directory / "finite_summary.csv",
            "anisotropy": data_directory / "anisotropic_phase.csv",
            "raw_example": data_directory / "supportshift_raw_example.csv",
            "multilag": data_directory / "multilag_composite.csv",
            "full_likelihood": data_directory / "full_likelihood_phase.csv",
            "joint_smoothness": data_directory / "joint_smoothness.csv",
            "matched_boundary": data_directory / "matched_boundary.csv",
        }.items()
        if name in input_paths
        and input_paths[name].resolve() == destination.resolve()
    }
    manifest = {
        "schema_version": "1.4",
        "inputs": input_manifest,
        "input_output_aliases": output_aliases,
        "outputs": {
            str(path.relative_to(args.paper_directory)): sha256_file(path)
            for path in artifact_paths
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
        },
    }
    manifest_path = data_directory / "supportshift_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Generated figures, table, and source-data extracts under {args.paper_directory}")


if __name__ == "__main__":
    main()
