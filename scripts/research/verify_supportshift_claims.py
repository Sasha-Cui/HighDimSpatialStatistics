"""Mechanically audit numerical claims reported in the SupportShift papers."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClaimCheck:
    """One paper-facing numerical assertion and its audit result."""

    claim: str
    observed: Any
    requirement: str
    passed: bool


class ClaimAudit:
    """Collect exact, rounded, and inequality checks without stopping early."""

    def __init__(self) -> None:
        self.checks: list[ClaimCheck] = []

    def exact(self, claim: str, observed: Any, expected: Any) -> None:
        observed = _jsonable(observed)
        expected = _jsonable(expected)
        self.checks.append(
            ClaimCheck(claim, observed, f"equals {expected!r}", observed == expected)
        )

    def close(self, claim: str, observed: float, reported: float, decimals: int) -> None:
        tolerance = 0.5 * 10.0 ** (-decimals) + 1e-12
        observed = float(observed)
        self.checks.append(
            ClaimCheck(
                claim,
                observed,
                f"rounds to {reported:.{decimals}f}",
                abs(observed - reported) <= tolerance,
            )
        )

    def upper(self, claim: str, observed: float, bound: float) -> None:
        observed = float(observed)
        self.checks.append(
            ClaimCheck(claim, observed, f"at most {bound:.12g}", observed <= bound)
        )

    def lower(self, claim: str, observed: float, bound: float) -> None:
        observed = float(observed)
        self.checks.append(
            ClaimCheck(claim, observed, f"at least {bound:.12g}", observed >= bound)
        )

    def truth(self, claim: str, observed: bool, requirement: str) -> None:
        self.checks.append(ClaimCheck(claim, bool(observed), requirement, bool(observed)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_row(data: pd.DataFrame, **conditions: float | int | str) -> pd.Series:
    selected = data
    for column, value in conditions.items():
        if isinstance(value, float):
            selected = selected[np.isclose(selected[column], value)]
        else:
            selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one row for {conditions}, found {len(selected)}")
    return selected.iloc[0]


def _rmse(values: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(values.to_numpy(dtype=float)))))


def _tables_match(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    relative_tolerance: float = 5e-15,
    absolute_tolerance: float = 5e-18,
) -> bool:
    """Compare promoted and paper tables across pandas CSV parsers.

    The paper-facing CSV is decimal-rounded from the promoted source.  Exact
    float equality can therefore depend on the parser version at the final bit.
    Non-floating columns remain exact, while floating columns must agree to a
    tolerance far below the numerical precision reported in either paper.
    """
    if left.shape != right.shape or not left.columns.equals(right.columns):
        return False
    for column in left.columns:
        left_values = left[column]
        right_values = right[column]
        if pd.api.types.is_float_dtype(left_values.dtype) and pd.api.types.is_float_dtype(
            right_values.dtype
        ):
            if not np.allclose(
                left_values.to_numpy(dtype=float),
                right_values.to_numpy(dtype=float),
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                equal_nan=True,
            ):
                return False
        elif not left_values.equals(right_values):
            return False
    return True


def audit_claims(repository_root: Path, paper_directory: Path) -> dict[str, Any]:
    """Return a complete numerical claim ledger for both paper versions."""
    audit = ClaimAudit()
    data_directory = paper_directory / "data"
    output_directory = repository_root / "outputs" / "smoothing_bias"

    phase = pd.read_csv(data_directory / "phase_oracle_d2.csv")
    promoted_phase = pd.read_csv(output_directory / "phase_oracle_d2_v2.csv")
    phase_metadata = _load_json(output_directory / "phase_oracle_d2_v2.metadata.json")
    audit.truth(
        "paper phase table matches promoted source",
        _tables_match(phase, promoted_phase),
        "schemas and discrete values are exact; floats agree within 5e-15 "
        "relative and 5e-18 absolute tolerance",
    )
    audit.exact("phase rows", len(phase), 108)
    audit.exact("phase dimensions", sorted(phase["dimension"].unique().tolist()), [2])
    audit.exact("phase quadrature order", sorted(phase["quadrature_order"].unique()), [96])
    audit.exact("phase bandwidth count", phase["bandwidth"].nunique(), 18)
    audit.close("phase minimum bandwidth", phase["bandwidth"].min(), 0.003, 3)
    audit.close("phase maximum bandwidth", phase["bandwidth"].max(), 0.300, 3)
    audit.truth("all phase shifts positive", bool((phase["decay_shift"] > 0).all()), "all positive")
    reported_slopes = {0.25: 0.515, 0.5: 0.999, 0.75: 1.468, 1.0: 1.817, 1.5: 1.994, 2.5: 2.000}
    reported_ratios = {0.25: 1.021, 0.5: 1.000, 0.75: 0.958, 1.0: 1.059, 1.5: 0.997, 2.5: 1.000}
    for smoothness, reported in reported_slopes.items():
        smallest = phase[np.isclose(phase["smoothness"], smoothness)].nsmallest(
            6, "bandwidth"
        )
        slope = np.polyfit(
            np.log(smallest["bandwidth"]), np.log(smallest["decay_shift"]), 1
        )[0]
        audit.close(f"phase slope nu={smoothness:g}", slope, reported, 3)
        audit.close(
            f"smallest-bandwidth coefficient ratio nu={smoothness:g}",
            smallest.iloc[0]["coefficient_ratio"],
            reported_ratios[smoothness],
            3,
        )
    phase_refinement = phase_metadata["quadrature_refinement"]
    audit.upper(
        "phase order-64 absolute refinement",
        phase_refinement["64"]["max_abs_pseudo_decay_difference"],
        4.4e-8,
    )
    audit.upper(
        "phase order-128 absolute refinement",
        phase_refinement["128"]["max_abs_pseudo_decay_difference"],
        5.2e-9,
    )

    transition = pd.read_csv(data_directory / "transition_stress.csv")
    transition_metadata = _load_json(
        output_directory / "supportshift_transition_stress_20260804.metadata.json"
    )
    audit.exact("transition rows", len(transition), 111)
    audit.exact("transition smoothness count", transition["smoothness"].nunique(), 37)
    audit.close("transition smoothness minimum", transition["smoothness"].min(), 0.55, 2)
    audit.close("transition smoothness maximum", transition["smoothness"].max(), 1.45, 2)
    audit.exact(
        "transition bandwidths",
        sorted(transition["bandwidth"].unique().tolist()),
        [0.01, 0.02, 0.05],
    )
    audit.close(
        "minimum exact-to-leading transition ratio",
        transition["exact_to_leading_ratio"].min(),
        0.154,
        3,
    )
    audit.close(
        "maximum one-term overprediction factor",
        (transition["leading_decay_shift"] / transition["exact_decay_shift"]).max(),
        6.51,
        2,
    )
    audit.upper(
        "transition-aware shift relative error",
        transition["transition_relative_error"].max(),
        0.00099,
    )
    audit.upper(
        "transition-aware variance-loss relative error",
        transition["transition_variance_loss_relative_error"].max(),
        0.00084,
    )
    audit.truth(
        "all exact transition shifts positive",
        bool((transition["exact_decay_shift"] > 0).all()),
        "all positive",
    )
    audit.truth(
        "all approximate transition shifts positive",
        bool((transition["transition_decay_shift"] > 0).all()),
        "all positive",
    )
    transition_refinement = transition_metadata["quadrature_refinement"]
    audit.upper(
        "transition order-64 relative refinement",
        transition_refinement["64"]["max_relative_decay_shift_difference"],
        4.0e-9,
    )
    audit.upper(
        "transition order-128 relative refinement",
        transition_refinement["128"]["max_relative_decay_shift_difference"],
        3.1e-10,
    )

    dimension_kernel = pd.read_csv(
        data_directory / "dimension_kernel_robustness.csv"
    )
    promoted_dimension_kernel_path = (
        output_directory / "supportshift_dimension_kernel_robustness_20260804.csv"
    )
    promoted_dimension_kernel = pd.read_csv(promoted_dimension_kernel_path)
    dimension_kernel_metadata = _load_json(
        output_directory
        / "supportshift_dimension_kernel_robustness_20260804.metadata.json"
    )
    audit.truth(
        "paper dimension-kernel table matches promoted source",
        _tables_match(dimension_kernel, promoted_dimension_kernel),
        "schemas and discrete values are exact; floats agree within 5e-15 "
        "relative and 5e-18 absolute tolerance",
    )
    audit.exact("dimension-kernel rows", len(dimension_kernel), 72)
    audit.exact(
        "dimension-kernel dimensions",
        sorted(dimension_kernel["dimension"].unique().tolist()),
        [1, 2, 3],
    )
    audit.exact(
        "dimension-kernel families",
        sorted(dimension_kernel["kernel_family"].unique().tolist()),
        ["epanechnikov", "uniform"],
    )
    audit.exact(
        "dimension-kernel smoothnesses",
        sorted(dimension_kernel["smoothness"].unique().tolist()),
        [0.5, 1.0, 1.5, 2.5],
    )
    audit.exact(
        "dimension-kernel bandwidths",
        sorted(dimension_kernel["bandwidth"].unique().tolist()),
        [0.002, 0.004, 0.008],
    )
    audit.exact(
        "dimension-kernel quadrature order",
        sorted(dimension_kernel["quadrature_order"].unique().tolist()),
        [48],
    )
    audit.truth(
        "all dimension-kernel shifts positive",
        bool((dimension_kernel["decay_shift"] > 0).all()),
        "all positive",
    )
    smallest_bandwidth = float(dimension_kernel["bandwidth"].min())
    smallest = dimension_kernel[
        np.isclose(dimension_kernel["bandwidth"], smallest_bandwidth)
    ]
    audit.close(
        "maximum dimension-kernel coefficient error",
        np.abs(smallest["coefficient_ratio"] - 1.0).max(),
        0.169,
        3,
    )
    refinement = dimension_kernel_metadata["validation_gates"][
        "quadrature_refinement"
    ]["observed_maximum_relative_shift_difference"]
    audit.upper("dimension-kernel relative refinement", refinement, 4.0e-7)
    audit.exact(
        "dimension-kernel gates pass",
        dimension_kernel_metadata["validation_gates"]["all_passed"],
        True,
    )
    audit.exact(
        "dimension-kernel provenance is clean",
        dimension_kernel_metadata["provenance"]["git_dirty"],
        False,
    )
    audit.exact(
        "dimension-kernel promoted hash",
        dimension_kernel_metadata["result_csv"]["sha256"],
        _sha256(promoted_dimension_kernel_path),
    )

    multilag = pd.read_csv(data_directory / "multilag_composite.csv")
    promoted_multilag_path = output_directory / "supportshift_multilag_composite.csv"
    promoted_multilag = pd.read_csv(promoted_multilag_path)
    multilag_metadata = _load_json(
        output_directory / "supportshift_multilag_composite.metadata.json"
    )
    audit.truth(
        "paper multi-lag table matches promoted source",
        _tables_match(multilag, promoted_multilag),
        "schemas and values match promoted source",
    )
    audit.exact("multi-lag rows", len(multilag), 48)
    audit.exact(
        "multi-lag factor cells",
        len(multilag.drop_duplicates(["smoothness", "bandwidth"])),
        12,
    )
    audit.truth(
        "all multi-lag shifts positive",
        bool((multilag["composite_decay_shift"] > 0.0).all()),
        "all positive",
    )
    audit.truth(
        "all multi-lag minimum KL values positive",
        bool((multilag["minimum_composite_kl"] > 0.0).all()),
        "all positive",
    )
    smallest = multilag[np.isclose(multilag["bandwidth"], multilag["bandwidth"].min())]
    smallest_cells = smallest.drop_duplicates(["smoothness", "bandwidth"])
    audit.upper(
        "multi-lag smallest-bandwidth shift error",
        np.abs(smallest_cells["composite_shift_ratio"] - 1.0).max(),
        0.0141,
    )
    audit.upper(
        "multi-lag smallest-bandwidth KL error",
        np.abs(smallest_cells["minimum_kl_ratio"] - 1.0).max(),
        0.09851,
    )
    audit.close(
        "rough pair coefficient at lag 0.5",
        _selected_row(multilag, smoothness=0.5, bandwidth=0.005, lag=0.5)[
            "pair_shift_coefficient"
        ],
        1.61,
        2,
    )
    audit.close(
        "rough pair coefficient at lag 2",
        _selected_row(multilag, smoothness=0.5, bandwidth=0.005, lag=2.0)[
            "pair_shift_coefficient"
        ],
        0.40,
        2,
    )
    audit.exact(
        "multi-lag gates pass",
        multilag_metadata["validation_gates"]["all_passed"],
        True,
    )
    audit.exact(
        "multi-lag provenance is clean",
        multilag_metadata["provenance"]["git_dirty"],
        False,
    )
    audit.exact(
        "multi-lag promoted hash",
        multilag_metadata["result_csv"]["sha256"],
        _sha256(promoted_multilag_path),
    )

    full_likelihood = pd.read_csv(data_directory / "full_likelihood_phase.csv")
    promoted_full_path = output_directory / "supportshift_full_likelihood_phase.csv"
    promoted_full = pd.read_csv(promoted_full_path)
    full_metadata = _load_json(
        output_directory / "supportshift_full_likelihood_phase.metadata.json"
    )
    audit.truth(
        "paper full-likelihood table matches promoted source",
        _tables_match(full_likelihood, promoted_full),
        "schemas and values match promoted source",
    )
    audit.exact("full-likelihood phase rows", len(full_likelihood), 12)
    audit.truth(
        "all full-likelihood minimum KL values positive",
        bool((full_likelihood["minimum_kl"] > 0.0).all()),
        "all positive",
    )
    audit.truth(
        "selected full-likelihood design inflates range",
        bool((full_likelihood["decay_shift"] > 0.0).all()),
        "all positive",
    )
    full_smallest = full_likelihood[
        np.isclose(full_likelihood["bandwidth"], full_likelihood["bandwidth"].min())
    ]
    audit.upper(
        "full-likelihood smallest-bandwidth decay error",
        np.abs(full_smallest["decay_shift_ratio"] - 1.0).max(),
        0.0542,
    )
    audit.upper(
        "full-likelihood smallest-bandwidth variance error",
        np.abs(full_smallest["log_variance_shift_ratio"] - 1.0).max(),
        0.1268,
    )
    audit.upper(
        "full-likelihood smallest-bandwidth KL error",
        np.abs(full_smallest["minimum_kl_ratio"] - 1.0).max(),
        0.0177,
    )
    audit.exact("full-likelihood gates pass", full_metadata["validation_gates"]["all_passed"], True)
    audit.exact("full-likelihood provenance is clean", full_metadata["provenance"]["git_dirty"], False)
    audit.exact(
        "full-likelihood promoted hash",
        full_metadata["result_csv"]["sha256"],
        _sha256(promoted_full_path),
    )

    joint = pd.read_csv(data_directory / "joint_smoothness.csv")
    joint_summary = pd.read_csv(data_directory / "joint_smoothness_summary.csv")
    promoted_joint_path = output_directory / "supportshift_joint_smoothness.csv"
    promoted_joint_summary_path = output_directory / "supportshift_joint_smoothness.summary.csv"
    joint_metadata = _load_json(
        output_directory / "supportshift_joint_smoothness.metadata.json"
    )
    audit.truth(
        "paper joint fit table matches promoted source",
        _tables_match(joint, pd.read_csv(promoted_joint_path)),
        "schemas and values match promoted source",
    )
    audit.truth(
        "paper joint summary matches promoted source",
        _tables_match(joint_summary, pd.read_csv(promoted_joint_summary_path)),
        "schemas and values match promoted source",
    )
    audit.exact("joint fit rows", len(joint), 2_400)
    audit.exact("joint summary rows", len(joint_summary), 24)
    point_joint = joint_summary[joint_summary["model"] == "point_support"]
    corrected_joint = joint_summary[joint_summary["model"] == "support_aware"]
    audit.exact(
        "point support raises joint smoothness in every cell",
        int((point_joint["population_smoothness_target"] > point_joint["smoothness_true"]).sum()),
        8,
    )
    audit.exact(
        "point support reverses fixed-smoothness decay direction in every cell",
        int((point_joint["population_decay_target"] > 1.0).sum()),
        8,
    )
    audit.truth(
        "support-aware joint population target is truth",
        bool(
            np.allclose(corrected_joint["population_smoothness_target"], corrected_joint["smoothness_true"])
            and np.allclose(corrected_joint["population_decay_target"], 1.0)
        ),
        "all target coordinates equal truth",
    )
    audit.close(
        "joint rough h0.5 smoothness target",
        _selected_row(point_joint, bandwidth=0.5, smoothness_true=0.5)["population_smoothness_target"],
        2.0,
        1,
    )
    audit.close(
        "joint rough h0.5 decay target",
        _selected_row(point_joint, bandwidth=0.5, smoothness_true=0.5)["population_decay_target"],
        2.176,
        3,
    )
    audit.close(
        "joint smooth h0.5 smoothness target",
        _selected_row(point_joint, bandwidth=0.5, smoothness_true=2.5)["population_smoothness_target"],
        3.5,
        1,
    )
    audit.close(
        "joint smooth h0.5 decay target",
        _selected_row(point_joint, bandwidth=0.5, smoothness_true=2.5)["population_decay_target"],
        1.338,
        3,
    )
    audit.exact("joint gates pass", joint_metadata["validation_gates"]["all_passed"], True)
    audit.exact("joint provenance is clean", joint_metadata["provenance"]["git_dirty"], False)
    audit.exact("joint promoted hash", joint_metadata["result_csv"]["sha256"], _sha256(promoted_joint_path))
    audit.exact(
        "joint summary promoted hash",
        joint_metadata["summary_csv"]["sha256"],
        _sha256(promoted_joint_summary_path),
    )

    matched = pd.read_csv(data_directory / "matched_boundary.csv")
    promoted_matched_path = output_directory / "supportshift_matched_boundary.csv"
    matched_metadata = _load_json(
        output_directory / "supportshift_matched_boundary.metadata.json"
    )
    audit.truth(
        "paper matched-boundary table matches promoted source",
        _tables_match(matched, pd.read_csv(promoted_matched_path)),
        "schemas and values match promoted source",
    )
    audit.exact("matched-boundary rows", len(matched), 48)
    audit.exact("matched-boundary output dimensions", sorted(matched["output_dimension"].unique().tolist()), [16])
    matched_selected = matched[np.isclose(matched["bandwidth"], 0.7)]
    point_pivot = matched_selected[matched_selected["model"] == "point_support"].pivot(
        index="smoothness", columns="region", values="decay_target"
    )
    boundary_differences = np.abs(point_pivot["boundary"] - point_pivot["interior"])
    audit.close("minimum matched boundary target difference", boundary_differences.min(), 0.030, 3)
    audit.close("maximum matched boundary target difference", boundary_differences.max(), 0.062, 3)
    kl_ratios: list[float] = []
    for region in ("interior", "boundary"):
        region_rows = matched_selected[matched_selected["region"] == region]
        point_kl = region_rows[region_rows["model"] == "point_support"].set_index("smoothness")["minimum_kl"]
        partial_kl = region_rows[region_rows["model"] == "partial_support"].set_index("smoothness")["minimum_kl"]
        kl_ratios.extend((partial_kl / point_kl).tolist())
    audit.close("minimum partial-to-point KL ratio", min(kl_ratios), 0.167, 3)
    audit.close("maximum partial-to-point KL ratio", max(kl_ratios), 0.323, 3)
    corrected_matched = matched[matched["model"] == "support_aware"]
    audit.upper(
        "matched support-aware decay error",
        np.abs(corrected_matched["decay_target"] - 1.0).max(),
        4.1e-7,
    )
    audit.exact("matched-boundary gates pass", matched_metadata["validation_gates"]["all_passed"], True)
    audit.exact("matched-boundary provenance is clean", matched_metadata["provenance"]["git_dirty"], False)
    audit.exact(
        "matched-boundary promoted hash",
        matched_metadata["result_csv"]["sha256"],
        _sha256(promoted_matched_path),
    )

    anisotropy = pd.read_csv(data_directory / "anisotropic_phase.csv")
    anisotropy_metadata = _load_json(
        output_directory / "supportshift_anisotropic_final_20260803.metadata.json"
    )
    audit.exact("anisotropy rows", len(anisotropy), 2_128)
    audit.exact("anisotropy angle count", anisotropy["angle_degrees"].nunique(), 19)
    audit.exact("anisotropy bandwidth count", anisotropy["bandwidth"].nunique(), 14)
    audit.truth(
        "all anisotropic-support shifts positive",
        bool((anisotropy["decay_shift"] > 0).all()),
        "all positive",
    )
    for smoothness, reported_ratio in ((1.5, 1.833), (2.5, 3.250)):
        common = {
            "aspect_ratio": 4.0,
            "smoothness": smoothness,
            "bandwidth": float(anisotropy["bandwidth"].min()),
        }
        major = _selected_row(anisotropy, angle_degrees=0.0, **common)
        minor = _selected_row(anisotropy, angle_degrees=90.0, **common)
        audit.close(
            f"anisotropic coefficient ratio nu={smoothness:g}",
            major["leading_coefficient"] / minor["leading_coefficient"],
            reported_ratio,
            3,
        )
    for smoothness, major_reported, minor_reported in (
        (0.5, 13.46, 11.47),
        (1.5, 1.56, 0.76),
    ):
        major = _selected_row(
            anisotropy,
            aspect_ratio=4.0,
            smoothness=smoothness,
            bandwidth=0.15,
            angle_degrees=0.0,
        )
        minor = _selected_row(
            anisotropy,
            aspect_ratio=4.0,
            smoothness=smoothness,
            bandwidth=0.15,
            angle_degrees=90.0,
        )
        audit.close(
            f"major-axis range inflation nu={smoothness:g}",
            100.0 * (major["implied_range_ratio"] - 1.0),
            major_reported,
            2,
        )
        audit.close(
            f"minor-axis range inflation nu={smoothness:g}",
            100.0 * (minor["implied_range_ratio"] - 1.0),
            minor_reported,
            2,
        )
    contrast_errors: list[float] = []
    smallest_bandwidth = float(anisotropy["bandwidth"].min())
    for smoothness in sorted(anisotropy["smoothness"].unique()):
        major = _selected_row(
            anisotropy,
            aspect_ratio=4.0,
            smoothness=float(smoothness),
            bandwidth=smallest_bandwidth,
            angle_degrees=0.0,
        )
        minor = _selected_row(
            anisotropy,
            aspect_ratio=4.0,
            smoothness=float(smoothness),
            bandwidth=smallest_bandwidth,
            angle_degrees=90.0,
        )
        contrast_ratio = (major["decay_shift"] - minor["decay_shift"]) / (
            major["directional_contrast_coefficient_to_minor"] * smallest_bandwidth**2
        )
        contrast_errors.append(abs(float(contrast_ratio) - 1.0))
    audit.upper("smallest-bandwidth directional contrast error", max(contrast_errors), 4.3e-6)
    anisotropy_refinement = anisotropy_metadata["quadrature_refinement"]
    audit.upper(
        "anisotropy relative shift refinement",
        max(
            anisotropy_refinement[order]["max_relative_decay_shift_difference"]
            for order in ("64", "128")
        ),
        1.3e-8,
    )
    audit.upper(
        "anisotropy relative contrast refinement",
        max(
            anisotropy_refinement[order]["max_relative_directional_contrast_difference"]
            for order in ("64", "128")
        ),
        1.2e-9,
    )

    finite = pd.read_csv(data_directory / "finite_summary.csv")
    core = finite[
        finite["config_id"].str.match(r"d2_nu(05|10|15|25)_h(00|03|05|07)")
    ]
    audit.exact("finite configurations", finite["config_id"].nunique(), 21)
    audit.exact("finite fit count", int(finite["replicates"].sum()), 8_400)
    audit.upper(
        "corrected core target deviation",
        np.abs(core.loc[core["model"] == "corrected", "population_target"] - 1.0).max(),
        1.3e-7,
    )
    for smoothness, target, naive_mean, corrected_mean in (
        (0.5, 0.152, 0.151, 1.090),
        (1.0, 0.414, 0.407, 0.990),
        (1.5, 0.573, 0.570, 0.990),
        (2.5, 0.740, 0.738, 0.994),
    ):
        naive = _selected_row(
            core, model="naive", smoothness=smoothness, bandwidth=0.7
        )
        corrected = _selected_row(
            core, model="corrected", smoothness=smoothness, bandwidth=0.7
        )
        audit.close(f"finite naive target nu={smoothness:g}", naive["population_target"], target, 3)
        audit.close(f"finite naive mean nu={smoothness:g}", naive["mean_estimate"], naive_mean, 3)
        audit.close(
            f"finite corrected mean nu={smoothness:g}",
            corrected["mean_estimate"],
            corrected_mean,
            3,
        )
    rough_corrected = _selected_row(
        core, model="corrected", smoothness=0.5, bandwidth=0.7
    )
    audit.truth(
        "rough corrected mean within two MCSE",
        abs(rough_corrected["mean_estimate"] - rough_corrected["population_target"])
        <= 2.0 * rough_corrected["monte_carlo_standard_error_mean"],
        "absolute error no greater than two MCSE",
    )
    audit.exact("core boundary fits", int(core["boundary_fits"].sum()), 6)
    for config_id, model, reported in (
        ("d2_boundary_nu05_h07", "naive", 0.135),
        ("d2_irregular_nu05_h05", "naive", 0.284),
    ):
        row = _selected_row(finite, config_id=config_id, model=model)
        audit.close(f"stress target {config_id}", row["population_target"], reported, 3)
    for config_id, model, reported in (
        ("d1_n100_nu05_h04", "corrected", 0.529),
        ("d1_n400_nu05_h04", "corrected", 0.177),
        ("d1_n100_nu05_h04", "naive", 0.151),
        ("d1_n400_nu05_h04", "naive", 0.068),
    ):
        row = _selected_row(finite, config_id=config_id, model=model)
        audit.close(
            f"domain-growth standard deviation {config_id} {model}",
            row["standard_deviation"],
            reported,
            3,
        )
    for model, bound in (("corrected", 0.026), ("naive", 0.009)):
        row = _selected_row(finite, config_id="d1_n400_nu05_h04", model=model)
        audit.upper(
            f"domain-growth mean error at n=400 {model}",
            abs(row["mean_estimate"] - row["population_target"]),
            bound,
        )

    raw = pd.read_csv(data_directory / "supportshift_raw_example.csv")
    audit.exact("raw illustration rows", len(raw), 2_516)
    audit.exact("raw illustration replicates", raw["replicate"].nunique(), 4)
    audit.exact(
        "raw latent rows",
        int((raw["field_stage"] == "latent_input").sum()),
        2_116,
    )
    audit.exact(
        "raw averaged rows",
        int((raw["field_stage"] == "averaged_output").sum()),
        400,
    )

    highdim = pd.read_csv(output_directory / "supportshift_highdim_final_v2_20260803.csv")
    highdim_summary = pd.read_csv(data_directory / "supportshift_highdim_summary.csv")
    highdim_metadata = _load_json(
        output_directory / "supportshift_highdim_final_v2_20260803.metadata.json"
    )
    audit.exact("high-dimensional rows", len(highdim), 12_800)
    audit.exact("high-dimensional cells", len(highdim_summary), 64)
    audit.exact("candidate count", sorted(highdim["candidate_count"].unique()), [16_261])
    audit.exact(
        "high-dimensional dimensions",
        sorted(highdim["dimension_p"].unique().tolist()),
        [16, 36, 64, 100],
    )
    audit.exact(
        "high-dimensional replicate counts",
        sorted(highdim["sample_size"].unique().tolist()),
        [1, 4, 16, 64],
    )
    audit.upper(
        "finite-grid objective resolution",
        highdim_metadata["validation_gates"]["finite_grid_approximates_continuous_oracle"][
            "maximum_absolute_normalized_nll_gap"
        ],
        2.63e-5,
    )
    for model, decay_bound, variance_bound in (
        ("corrected", 1.68e-7, 4.06e-7),
        ("naive", 0.00499, 0.0297),
    ):
        selected_model = highdim[highdim["model"] == model]
        audit.upper(
            f"{model} continuous-to-grid decay distance",
            np.abs(
                selected_model["population_grid_decay_target"]
                - selected_model["population_continuous_decay_target"]
            ).max(),
            decay_bound,
        )
        audit.upper(
            f"{model} continuous-to-grid variance distance",
            np.abs(
                selected_model["population_grid_variance_target"]
                - selected_model["population_continuous_variance_target"]
            ).max(),
            variance_bound,
        )
    audit.truth(
        "all candidatewise certificate events hold",
        bool(highdim["simultaneous_candidatewise_bound_holds"].all()),
        "all true",
    )
    audit.upper(
        "maximum candidatewise deviation-to-radius ratio",
        highdim["max_candidatewise_deviation_to_bound_ratio"].max(),
        0.7894,
    )
    audit.truth(
        "all deterministic ERM inequalities hold",
        bool(highdim["erm_inequality_holds"].all()),
        "all true",
    )
    envelope_ratio = (
        highdim_summary["criterion_deviation_q95"]
        / highdim_summary["uniform_likelihood_bound"]
    )
    audit.close("maximum cell q95-to-envelope ratio", envelope_ratio.max(), 0.402, 3)
    audit.close("median cell q95-to-envelope ratio", envelope_ratio.median(), 0.298, 3)
    slopes: list[float] = []
    for _, group in highdim_summary.groupby(["model", "dimension_p", "smoothness"]):
        slopes.append(
            float(
                np.polyfit(
                    np.log(group["sample_size"]),
                    np.log(group["criterion_deviation_q95"]),
                    1,
                )[0]
            )
        )
    audit.close("minimum criterion-noise slope", min(slopes), -0.528, 3)
    audit.close("maximum criterion-noise slope", max(slopes), -0.454, 3)
    audit.close("median criterion-noise slope", float(np.median(slopes)), -0.505, 3)
    dimension_slopes: list[float] = []
    for _, group in highdim_summary.groupby(["model", "sample_size", "smoothness"]):
        dimension_slopes.append(
            float(
                np.polyfit(
                    np.log(group["dimension_p"]),
                    np.log(group["criterion_deviation_q95"]),
                    1,
                )[0]
            )
        )
    audit.close("minimum fixed-N dimension slope", min(dimension_slopes), -0.509, 3)
    audit.close("maximum fixed-N dimension slope", max(dimension_slopes), -0.297, 3)
    audit.close("median fixed-N dimension slope", float(np.median(dimension_slopes)), -0.413, 3)

    selected = highdim[
        (highdim["dimension_p"] == 100)
        & np.isclose(highdim["smoothness"], 1.5)
        & (highdim["model"] == "naive")
    ]
    audit.close(
        "p100 nu1.5 naive grid target",
        selected["population_grid_decay_target"].iloc[0],
        0.594,
        3,
    )
    audit.close(
        "p100 nu1.5 naive continuous target",
        selected["population_continuous_decay_target"].iloc[0],
        0.589,
        3,
    )
    for sample_size, reported in ((1, 0.124), (64, 0.015)):
        rows = selected[selected["sample_size"] == sample_size]
        audit.close(
            f"p100 nu1.5 naive RMSE to target N={sample_size}",
            _rmse(rows["decay_error_to_grid_target"]),
            reported,
            3,
        )
    selected_n64 = selected[selected["sample_size"] == 64]
    audit.close(
        "p100 nu1.5 naive RMSE to truth N=64",
        _rmse(selected_n64["decay_error_to_truth"]),
        0.409,
        3,
    )
    corrected = highdim[
        (highdim["dimension_p"] == 100)
        & np.isclose(highdim["smoothness"], 1.5)
        & (highdim["model"] == "corrected")
    ]
    for sample_size, reported in ((1, 0.206), (64, 0.024)):
        rows = corrected[corrected["sample_size"] == sample_size]
        audit.close(
            f"p100 nu1.5 corrected RMSE N={sample_size}",
            _rmse(rows["decay_error_to_grid_target"]),
            reported,
            3,
        )
    rough_naive_n64 = highdim[
        (highdim["dimension_p"] == 100)
        & np.isclose(highdim["smoothness"], 0.5)
        & (highdim["model"] == "naive")
        & (highdim["sample_size"] == 64)
    ]
    audit.close(
        "p100 nu0.5 naive RMSE to target N=64",
        _rmse(rough_naive_n64["decay_error_to_grid_target"]),
        0.015,
        3,
    )
    audit.close(
        "p100 nu0.5 naive RMSE to truth N=64",
        _rmse(rough_naive_n64["decay_error_to_truth"]),
        0.681,
        3,
    )

    failures = [asdict(check) for check in audit.checks if not check.passed]
    return {
        "schema_version": "1.0",
        "checks": [asdict(check) for check in audit.checks],
        "summary": {
            "checks": len(audit.checks),
            "passed": len(audit.checks) - len(failures),
            "failed": len(failures),
            "all_passed": not failures,
        },
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--paper-directory", type=Path, default=Path("paper"))
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    paper_directory = (
        args.paper_directory
        if args.paper_directory.is_absolute()
        else repository_root / args.paper_directory
    )
    ledger = audit_claims(repository_root, paper_directory)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary = ledger["summary"]
    if not summary["all_passed"]:
        messages = [
            f"{failure['claim']}: observed {failure['observed']!r}; "
            f"required {failure['requirement']}"
            for failure in ledger["failures"]
        ]
        raise SystemExit("SupportShift paper-claim audit failed:\n- " + "\n- ".join(messages))
    print(f"SupportShift paper claims verified: {summary['passed']} numerical checks.")


if __name__ == "__main__":
    main()
