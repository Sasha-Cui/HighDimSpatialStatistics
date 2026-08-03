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
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
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
    phase.to_csv(data_directory / "phase_oracle_d2.csv", index=False)
    summary.to_csv(data_directory / "finite_summary.csv", index=False)
    print(f"Generated figures, table, and source-data extracts under {args.paper_directory}")


if __name__ == "__main__":
    main()
