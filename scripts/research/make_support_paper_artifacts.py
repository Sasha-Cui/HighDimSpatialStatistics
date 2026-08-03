"""Generate paper figures, tables, and compact source-data extracts."""
from __future__ import annotations

import argparse
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
NU_COLORS = ["#332288", "#117733", "#44AA99", "#DDCC77", "#CC6677", "#882255"]


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
                "uniform_bound_coverage": float(group["uniform_bound_holds"].mean()),
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
    certificate_axis.set_title("Finite-library certificate")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--finite-summary", type=Path, required=True)
    parser.add_argument("--anisotropy", type=Path)
    parser.add_argument("--raw-example", type=Path)
    parser.add_argument("--highdim", type=Path)
    parser.add_argument("--paper-directory", type=Path, required=True)
    args = parser.parse_args()
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
    finite_figure(summary, figures)
    convergence_figure(summary, figures)
    finite_table(summary, tables / "finite_summary.tex")
    if args.anisotropy is not None:
        anisotropy = pd.read_csv(args.anisotropy)
        anisotropy_figure(anisotropy, figures)
        anisotropy.to_csv(data_directory / "anisotropic_phase.csv", index=False)
    if args.raw_example is not None:
        raw_example = pd.read_csv(args.raw_example)
        raw_support_figure(raw_example, figures)
        raw_example.to_csv(data_directory / "supportshift_raw_example.csv", index=False)
    if args.highdim is not None:
        highdim = pd.read_csv(args.highdim)
        highdim_summary = highdim_figure(highdim, figures)
        highdim_summary.to_csv(
            data_directory / "supportshift_highdim_summary.csv",
            index=False,
        )
    phase.to_csv(data_directory / "phase_oracle_d2.csv", index=False)
    summary.to_csv(data_directory / "finite_summary.csv", index=False)
    print(f"Generated figures, table, and source-data extracts under {args.paper_directory}")


if __name__ == "__main__":
    main()
