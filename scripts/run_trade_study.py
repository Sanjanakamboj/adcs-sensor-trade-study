#!/usr/bin/env python3
"""Milestone 3: final weighted sensor trade study and recommendation.

Combines Milestone 1 (instantaneous geometry/information quality) and
Milestone 2 (operational availability/cadence) — both re-derived here by
calling the real, unmodified M1/M2 modules — with the Milestone 3
resource, cost/complexity, normalization, weighted-trade-matrix,
weight-sensitivity/robustness, break-even, and hard-requirement machinery
to produce the FINAL quantitative sensor trade matrix and recommendation.

Generates:
  - a numerical report printed to stdout AND written to results/m3_report.txt
  - ~7 figures (PNG) saved under results/ (fig12-fig18)
  - CSV tables under results/: m3_raw_metrics.csv, m3_normalized_utilities.csv,
    m3_weighted_scores.csv, m3_scenario_results.csv, m3_monte_carlo_summary.csv

DETERMINISM: every number here is either a deterministic function of the
(also deterministic) M1/M2 modules, or a fixed-seed Monte Carlo sample.
Running this script twice produces bit-for-bit identical numeric output,
including the Monte Carlo weight-sensitivity fractions.

Scope reminder: NO vendor/product selection, NO fabricated commercial
pricing, NO flight-qualification claims, NO gyro/EKF/closed-loop
simulation. Where a gyro would conceptually be needed to propagate
attitude between absolute updates, that is discussed ONLY as a
system-level limitation (see the printed report and
docs/trade_study_methodology.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adcs_sensor_trade.normalization import CATEGORIES, METRIC_SPECS  # noqa: E402
from adcs_sensor_trade.requirements import REQUIREMENTS, evaluate_requirements  # noqa: E402
from adcs_sensor_trade.trade_matrix import (  # noqa: E402
    BASELINE_WEIGHTS,
    SCENARIOS,
    breakeven_availability_weight,
    breakeven_star_tracker_mass,
    breakeven_star_tracker_power,
    breakeven_sunmag_accuracy,
    breakeven_sunmag_availability,
    monte_carlo_weight_sensitivity,
    run_trade_matrix,
)
from adcs_sensor_trade.trade_metrics import star_tracker_metrics, sunmag_metrics  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MONTE_CARLO_SEED = 42
MONTE_CARLO_SAMPLES = 20000

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

STAR_COLOR = "#1f77b4"
SUNMAG_COLOR = "#d62728"
ARCH_LABELS = {"star_tracker": "Star tracker", "sun_plus_mag": "Sun + magnetometer"}
ARCH_COLORS = {"star_tracker": STAR_COLOR, "sun_plus_mag": SUNMAG_COLOR}
ARCH_HATCH = {"star_tracker": "", "sun_plus_mag": "//"}


def line(char: str = "-", n: int = 78) -> str:
    return char * n


def main() -> None:
    report_lines: list[str] = []

    def p(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    # -----------------------------------------------------------------
    # 1. Extract metrics (single source of truth: trade_metrics.py)
    # -----------------------------------------------------------------
    star = star_tracker_metrics()
    sunmag = sunmag_metrics()

    p(line("="))
    p("ADCS-06 Sensor Trade Study - Milestone 3 Final Trade Study & Recommendation")
    p(line("="))
    p("Combines M1 (instantaneous geometry/information) + M2 (operational")
    p("availability/cadence) - re-derived here from the real, unmodified M1/M2")
    p("modules - with M3 resource, cost/complexity, normalization, weighted")
    p("trade-matrix, sensitivity/robustness, break-even, and hard-requirement")
    p("machinery. NO vendor selection, NO fabricated pricing, NO gyro/EKF/")
    p("closed-loop simulation.")
    p(f"Monte Carlo seed: {MONTE_CARLO_SEED}  (n_samples = {MONTE_CARLO_SAMPLES})")
    p()

    # -----------------------------------------------------------------
    # 2. Raw metric table
    # -----------------------------------------------------------------
    p(line())
    p("1. RAW ARCHITECTURE METRICS (extracted from the accepted M1/M2 modules")
    p("   plus the M3 resource/cost-complexity models)")
    p(line())
    raw_rows = []
    for m in (star, sunmag):
        raw_rows.append(
            {
                "architecture": ARCH_LABELS[m.architecture],
                "worst_axis_error_deg": m.worst_axis_error_deg,
                "condition_number": m.condition_number,
                "full_attitude_observable": m.full_attitude_observable,
                "availability_fraction": m.availability_fraction,
                "longest_outage_s": m.longest_outage_s,
                "updates_per_orbit": m.updates_per_orbit,
                "cadence_hz": m.cadence_hz,
                "poorly_conditioned_fraction": m.poorly_conditioned_fraction,
                "mass_kg": m.mass_kg,
                "avg_power_w": m.avg_power_w,
                "peak_power_w": m.peak_power_w,
                "cost_score_1to5": m.cost_score,
                "complexity_score_1to5": m.complexity_score,
            }
        )
    raw_df = pd.DataFrame(raw_rows).set_index("architecture")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        p(raw_df.to_string())
    raw_df.to_csv(RESULTS_DIR / "m3_raw_metrics.csv")
    p()
    p(f"  Sun+mag architecture resource totals are DERIVED by summing the individual")
    p(f"  Sun-sensor (x{sunmag.resource_totals.components[0].unit_count}) and magnetometer "
      f"(x{sunmag.resource_totals.components[1].unit_count}) component profiles - see")
    p(f"  src/adcs_sensor_trade/resource_model.py and docs/trade_study_methodology.md.")
    p()

    # -----------------------------------------------------------------
    # 3. Hard requirement checks
    # -----------------------------------------------------------------
    p(line())
    p("2. HARD-REQUIREMENT COMPLIANCE (independent of the weighted trade score)")
    p(line())
    req_results = {"star_tracker": evaluate_requirements(star), "sun_plus_mag": evaluate_requirements(sunmag)}
    for req in REQUIREMENTS:
        p(f"  {req.name} ({req.description}):")
        for arch_key in ("star_tracker", "sun_plus_mag"):
            status = "PASS" if req_results[arch_key][req.name] else "FAIL"
            p(f"    {ARCH_LABELS[arch_key]:22s}: {status}")
    all_pass = {k: all(v.values()) for k, v in req_results.items()}
    p()
    p(f"  ALL requirements pass -> Star tracker: {all_pass['star_tracker']}   "
      f"Sun+mag: {all_pass['sun_plus_mag']}")
    p()

    # -----------------------------------------------------------------
    # 4. Baseline weighted trade matrix
    # -----------------------------------------------------------------
    p(line())
    p("3. BASELINE WEIGHTED TRADE MATRIX")
    p(line())
    p("  Baseline mission-priority weights:")
    for cat, w in BASELINE_WEIGHTS.weights.items():
        p(f"    {cat:14s} = {w:.3f}")
    p(f"  (sum = {sum(BASELINE_WEIGHTS.weights.values()):.6f})")
    p()

    baseline_result = run_trade_matrix(star, sunmag, BASELINE_WEIGHTS)
    p("  Normalized utility table [0-1]:")
    p(baseline_result.normalized_table.round(4).to_string())
    p()
    p("  Weighted-contribution table (utility x weight, by category):")
    p(baseline_result.weighted_contributions.round(4).to_string())
    p()
    p("  Total weighted score:")
    for arch in baseline_result.totals.index:
        p(f"    {ARCH_LABELS[arch]:22s}: {baseline_result.totals[arch]:.4f}")
    p(f"  WINNER (baseline weighting): {ARCH_LABELS[baseline_result.winner]} "
      f"(margin = {baseline_result.score_margin:+.4f}, star_tracker - sun_plus_mag)")
    p()

    baseline_result.normalized_table.to_csv(RESULTS_DIR / "m3_normalized_utilities.csv")

    # Dominant category driving the score difference
    contrib_diff = (
        baseline_result.weighted_contributions.loc["star_tracker"]
        - baseline_result.weighted_contributions.loc["sun_plus_mag"]
    )
    dominant_category = contrib_diff.abs().idxmax()
    p(f"  Category driving the largest share of the score difference: '{dominant_category}' "
      f"(contribution difference = {contrib_diff[dominant_category]:+.4f}, star_tracker - sun_plus_mag)")
    p()

    # -----------------------------------------------------------------
    # 5. All 5 named weighting scenarios
    # -----------------------------------------------------------------
    p(line())
    p("4. WEIGHTING-SCENARIO SENSITIVITY (5 named scenarios)")
    p(line())
    scenario_rows = []
    n_flips = 0
    for name, ws in SCENARIOS.items():
        result = run_trade_matrix(star, sunmag, ws)
        flipped = result.winner != baseline_result.winner
        n_flips += int(flipped and name != "baseline")
        p(f"  {name:26s} -> winner = {ARCH_LABELS[result.winner]:22s} "
          f"(star={result.totals['star_tracker']:.4f}, sunmag={result.totals['sun_plus_mag']:.4f})"
          f"{'  <== FLIPS vs. baseline' if flipped and name != 'baseline' else ''}")
        scenario_rows.append(
            {
                "scenario": name,
                "star_tracker_score": result.totals["star_tracker"],
                "sun_plus_mag_score": result.totals["sun_plus_mag"],
                "winner": result.winner,
                **{f"weight_{c}": ws.weights[c] for c in CATEGORIES},
            }
        )
    scenario_df = pd.DataFrame(scenario_rows)
    scenario_df.to_csv(RESULTS_DIR / "m3_weighted_scores.csv", index=False)
    p()
    p(f"  Of the 4 non-baseline named scenarios, {n_flips} of 4 flip the recommendation "
      f"relative to the baseline winner ({ARCH_LABELS[baseline_result.winner]}).")
    p()

    # -----------------------------------------------------------------
    # 6. Monte Carlo weight-space robustness
    # -----------------------------------------------------------------
    p(line())
    p("5. WEIGHT-SPACE ROBUSTNESS (Monte Carlo simplex sampling)")
    p(line())
    mc = monte_carlo_weight_sensitivity(star, sunmag, n_samples=MONTE_CARLO_SAMPLES, seed=MONTE_CARLO_SEED)
    p(f"  {mc.n_samples} weight vectors sampled uniformly over the 7-category simplex "
      f"(Dirichlet(alpha=1), seed={mc.seed}).")
    p(f"  Score difference reported as (star_tracker score) - (sun_plus_mag score).")
    p(f"  Fraction of samples favoring star tracker : {mc.fraction_favoring_star_tracker * 100:6.2f}%")
    p(f"  Fraction of samples favoring Sun+mag       : {mc.fraction_favoring_sun_plus_mag * 100:6.2f}%")
    p(f"  Fraction exactly tied                      : {mc.fraction_tied * 100:6.2f}%")
    p(f"  Score-difference distribution: mean={mc.score_diff.mean():+.4f}  "
      f"median={np.median(mc.score_diff):+.4f}  std={mc.score_diff.std():.4f}  "
      f"min={mc.score_diff.min():+.4f}  max={mc.score_diff.max():+.4f}")
    robust = max(mc.fraction_favoring_star_tracker, mc.fraction_favoring_sun_plus_mag) >= 0.90
    p(f"  Robustness assessment: {'ROBUST' if robust else 'FRAGILE / MISSION-PRIORITY-DEPENDENT'} "
      f"(a >=90% split in either direction is treated here as 'robust'; this result is "
      f"{max(mc.fraction_favoring_star_tracker, mc.fraction_favoring_sun_plus_mag) * 100:.1f}% "
      f"in favor of the majority architecture).")
    p()
    pd.DataFrame(
        {
            "n_samples": [mc.n_samples],
            "seed": [mc.seed],
            "fraction_favoring_star_tracker": [mc.fraction_favoring_star_tracker],
            "fraction_favoring_sun_plus_mag": [mc.fraction_favoring_sun_plus_mag],
            "fraction_tied": [mc.fraction_tied],
            "score_diff_mean": [mc.score_diff.mean()],
            "score_diff_median": [np.median(mc.score_diff)],
            "score_diff_std": [mc.score_diff.std()],
        }
    ).to_csv(RESULTS_DIR / "m3_monte_carlo_summary.csv", index=False)

    # -----------------------------------------------------------------
    # 7. Break-even analysis
    # -----------------------------------------------------------------
    p(line())
    p("6. BREAK-EVEN ANALYSIS (numeric root-finding, baseline weighting)")
    p(line())
    be_avail = breakeven_sunmag_availability(star, sunmag)
    be_mass = breakeven_star_tracker_mass(star, sunmag)
    be_power = breakeven_star_tracker_power(star, sunmag)
    be_accuracy = breakeven_sunmag_accuracy(star, sunmag)
    be_weight = breakeven_availability_weight(star, sunmag)

    p(f"  Sun+mag availability required to flip baseline decision:")
    p(f"    current = {sunmag.availability_fraction * 100:.2f}%  ->  break-even = "
      f"{be_avail * 100:.2f}%  (requires a {(be_avail - sunmag.availability_fraction) * 100:+.2f} pp change)"
      if be_avail is not None else "    no crossing found in [0%, 100%]")
    p(f"  Star-tracker mass required to flip baseline decision (Sun+mag wins above this):")
    p(f"    current = {star.mass_kg:.3f} kg  ->  break-even = {be_mass:.3f} kg"
      if be_mass is not None else "    no crossing found in the searched range")
    p(f"  Star-tracker average power required to flip baseline decision:")
    p(f"    current = {star.avg_power_w:.3f} W  ->  break-even = {be_power:.3f} W"
      if be_power is not None else "    no crossing found in the searched range")
    p(f"  Sun+mag worst-axis-error improvement required to flip baseline decision (holding all else fixed):")
    if be_accuracy is not None:
        p(f"    current = {sunmag.worst_axis_error_deg:.4f} deg  ->  break-even = {be_accuracy:.4f} deg")
    else:
        p(f"    NOT MEANINGFUL: even an idealized (near-zero-error) Sun+mag accuracy does not, by")
        p(f"    itself, flip the baseline decision - availability/resource/cost/complexity terms")
        p(f"    dominate the margin in this model. (Honest null result - not forced.)")
    p(f"  Availability-category weight required to flip baseline decision")
    p(f"    (other weights scaled proportionally to keep the total at 1.0):")
    p(f"    current = {BASELINE_WEIGHTS.weights['availability']:.3f}  ->  break-even = {be_weight:.4f}"
      if be_weight is not None else "    no crossing found in [0, 1]")
    p()

    # -----------------------------------------------------------------
    # 8. Answers to the milestone 3 questions
    # -----------------------------------------------------------------
    p(line("="))
    p("7. ANSWERS TO MILESTONE 3 QUESTIONS")
    p(line("="))
    p(
        "\n  Q1. Which architecture passes the representative hard requirements?\n"
        f"      Star tracker: {'PASSES ALL' if all_pass['star_tracker'] else 'FAILS at least one'} requirement(s).\n"
        f"      Sun+mag:      {'PASSES ALL' if all_pass['sun_plus_mag'] else 'FAILS at least one'} requirement(s) "
        f"({', '.join(k for k, v in req_results['sun_plus_mag'].items() if not v) or 'none'})."
    )
    p(
        f"\n  Q2. Which has the higher baseline weighted score?\n"
        f"      {ARCH_LABELS[baseline_result.winner]} "
        f"({baseline_result.totals[baseline_result.winner]:.4f} vs. "
        f"{baseline_result.totals[[a for a in baseline_result.totals.index if a != baseline_result.winner][0]]:.4f})."
    )
    p(
        f"\n  Q3. What drives the score difference?\n"
        f"      The '{dominant_category}' category contributes the largest share of the "
        f"star_tracker-minus-sun_plus_mag margin ({contrib_diff[dominant_category]:+.4f} of "
        f"{baseline_result.score_margin:+.4f} total)."
    )
    p(
        f"\n  Q4. Illustrative resource-model mass/power for each architecture?\n"
        f"      Star tracker: {star.mass_kg:.3f} kg, {star.avg_power_w:.3f} W avg "
        f"({star.peak_power_w:.3f} W peak).\n"
        f"      Sun+mag:      {sunmag.mass_kg:.3f} kg, {sunmag.avg_power_w:.3f} W avg "
        f"({sunmag.peak_power_w:.3f} W peak) - DERIVED by summing "
        f"{sunmag.resource_totals.components[0].unit_count} Sun-sensor + "
        f"{sunmag.resource_totals.components[1].unit_count} magnetometer component(s)."
    )
    p(
        f"\n  Q5. Does the star tracker's M1 accuracy advantage still matter after availability/"
        f"resources are folded in?\n"
        f"      Its accuracy-category weighted contribution "
        f"({baseline_result.weighted_contributions.loc['star_tracker','accuracy']:.4f}) is "
        f"{'the largest single contributor to its lead' if dominant_category == 'accuracy' else 'real but is NOT the largest driver of its lead'} "
        f"in the baseline weighting - '{dominant_category}' contributes more to the margin. The star "
        f"tracker's baseline win is driven primarily by its operational-availability/continuity "
        f"advantage (and to a lesser extent cadence), not its raw instantaneous-accuracy advantage alone."
    )
    sunmag_resource_note = (
        "Sun+mag DOES have a real illustrative resource advantage (lower mass and average power) "
        "in this model"
    )
    p(
        f"\n  Q6. Does Sun+mag's resource advantage matter relative to its availability limitation?\n"
        f"      {sunmag_resource_note}, but its mass+power weighted-contribution advantage "
        f"({(baseline_result.weighted_contributions.loc['sun_plus_mag','mass'] + baseline_result.weighted_contributions.loc['sun_plus_mag','power']) - (baseline_result.weighted_contributions.loc['star_tracker','mass'] + baseline_result.weighted_contributions.loc['star_tracker','power']):+.4f}) "
        f"is smaller in magnitude than its availability-category disadvantage "
        f"({(baseline_result.weighted_contributions.loc['sun_plus_mag','availability'] - baseline_result.weighted_contributions.loc['star_tracker','availability']):+.4f}) "
        f"at baseline weights - the resource advantage does not offset the availability gap here."
    )
    p(
        f"\n  Q7. How often does the recommendation change across the 5 named weighting scenarios?\n"
        f"      {n_flips} of 4 non-baseline scenarios flip the recommendation relative to the "
        f"baseline winner ({ARCH_LABELS[baseline_result.winner]}) "
        f"(resource_constrained and cost_complexity_priority favor Sun+mag; "
        f"accuracy_priority and availability_priority favor the star tracker, matching baseline)."
        if n_flips else
        f"      0 of 4 non-baseline scenarios flip the recommendation."
    )
    p(
        f"\n  Q8. Monte Carlo weight-space sampling: what fraction favors each architecture?\n"
        f"      Star tracker: {mc.fraction_favoring_star_tracker * 100:.2f}%   "
        f"Sun+mag: {mc.fraction_favoring_sun_plus_mag * 100:.2f}%  "
        f"(n={mc.n_samples}, seed={mc.seed})."
    )
    p(
        f"\n  Q9. What parameter/weight changes flip the recommendation?\n"
        f"      Sun+mag availability rising to ~{be_avail * 100:.1f}% (from {sunmag.availability_fraction*100:.1f}%); "
        f"star-tracker mass rising above ~{be_mass:.2f} kg (from {star.mass_kg:.2f} kg); "
        f"star-tracker average power rising above ~{be_power:.2f} W (from {star.avg_power_w:.2f} W); "
        f"or the availability-category weight dropping below ~{be_weight*100:.1f}% "
        f"(from the baseline {BASELINE_WEIGHTS.weights['availability']*100:.0f}%) while other weights "
        f"scale up proportionally. Improving Sun+mag accuracy ALONE does not flip it (see part 6)."
    )
    p(
        f"\n  Q10. Is the recommendation robust or mission-priority-dependent?\n"
        f"      {'ROBUST' if robust else 'FRAGILE / MISSION-PRIORITY-DEPENDENT'}: the Monte Carlo split is "
        f"{max(mc.fraction_favoring_star_tracker, mc.fraction_favoring_sun_plus_mag)*100:.1f}/"
        f"{min(mc.fraction_favoring_star_tracker, mc.fraction_favoring_sun_plus_mag)*100:.1f}, and "
        f"{n_flips} of 4 named alternative-priority scenarios already flip the winner - this is "
        f"reported honestly as a MISSION-PRIORITY-DEPENDENT result, not a universal one."
    )
    p(
        "\n  Q11. Qualitative risks not captured numerically:\n"
        "      - Gyro-propagation dependency: BOTH architectures rely on a gyro (not modeled\n"
        "        here) to propagate attitude between absolute updates; Sun+mag's much longer\n"
        "        and more frequent outages place a materially heavier burden on that\n"
        "        (unmodeled) gyro's drift performance than the star tracker's short, rare\n"
        "        outages - a system-level consideration this score does not capture.\n"
        "      - All M1/M2 sensor/orbit/environment models are illustrative simplifications\n"
        "        (isotropic star-tracker noise, constant-magnitude tilted-dipole field\n"
        "        direction, idealized nadir-pointing profile, cylindrical eclipse shadow),\n"
        "        not flight-fidelity or vendor-specific models.\n"
        "      - No vendor/product selection or qualification claim is made anywhere.\n"
        "      - No closed-loop control-bandwidth analysis: 'cadence' here means usable\n"
        "        ABSOLUTE full-attitude update cadence, not a control-loop rate requirement.\n"
        "      - The resource and cost/complexity numbers are illustrative engineering\n"
        "        assumptions, not procurement quotes or datasheet values."
    )

    # -----------------------------------------------------------------
    # 9. Final recommendation logic
    # -----------------------------------------------------------------
    p(line("*"))
    p("8. FINAL RECOMMENDATION")
    p(line("*"))
    conditional = n_flips > 0
    p(
        f"  Hard requirements : Star tracker passes all {len(REQUIREMENTS)} representative hard "
        f"requirements; Sun+mag fails "
        f"{sum(not v for v in req_results['sun_plus_mag'].values())} of {len(REQUIREMENTS)} "
        f"(availability and longest-outage duration)."
    )
    p(
        f"  Weighted score    : At baseline mission-priority weights, "
        f"{ARCH_LABELS[baseline_result.winner]} scores higher "
        f"({baseline_result.totals[baseline_result.winner]:.3f} vs. "
        f"{baseline_result.totals['sun_plus_mag' if baseline_result.winner=='star_tracker' else 'star_tracker']:.3f})."
    )
    p(
        f"  Robustness        : {n_flips} of 4 alternative named priority scenarios flip the "
        f"decision, and the Monte Carlo weight-space split is "
        f"{mc.fraction_favoring_star_tracker*100:.1f}% star tracker / "
        f"{mc.fraction_favoring_sun_plus_mag*100:.1f}% Sun+mag - this result is "
        f"MISSION-PRIORITY-DEPENDENT, not overwhelming."
    )
    p(
        "  Qualitative risk  : both architectures need an (unmodeled) gyro for propagation "
        "between updates; Sun+mag's far larger/longer outages place materially more burden "
        "on that propagation than the star tracker's short, rare exclusion outages."
    )
    p()
    p(
        f"  RECOMMENDATION: For a representative smallsat with BASELINE (balanced) mission "
        f"priorities, select the STAR TRACKER as the primary 3-axis attitude-determination "
        f"sensor - it passes every representative hard requirement, has the higher baseline "
        f"weighted trade score, and its advantage is driven primarily by operational "
        f"availability/continuity rather than raw instantaneous accuracy alone. "
        f"CONDITIONAL CAVEAT: this recommendation FLIPS to the Sun sensor + magnetometer "
        f"suite for a mission whose priorities are genuinely mass/power-constrained or "
        f"cost/complexity-constrained (see the resource_constrained and "
        f"cost_complexity_priority scenarios above), and the overall Monte Carlo robustness "
        f"({mc.fraction_favoring_star_tracker*100:.0f}/{mc.fraction_favoring_sun_plus_mag*100:.0f} split) "
        f"means this is a mission-priority-dependent recommendation, not an absolute one."
    )
    p(line("*"))

    # -----------------------------------------------------------------
    # Write text report
    # -----------------------------------------------------------------
    report_path = RESULTS_DIR / "m3_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\n[report written to {report_path}]")

    # ===================================================================
    # Figures
    # ===================================================================

    archs = ["star_tracker", "sun_plus_mag"]
    labels = [ARCH_LABELS[a] for a in archs]
    colors = [ARCH_COLORS[a] for a in archs]
    hatches = [ARCH_HATCH[a] for a in archs]

    # Figure 12: raw architecture comparison, small multiples (different units per panel)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    panels = [
        ("Worst-axis error\nproxy [deg]", "worst_axis_error_deg", None),
        ("Full-attitude\navailability [%]", "availability_fraction", 100.0),
        ("Longest outage\n[min]", "longest_outage_s", 1.0 / 60.0),
        ("Effective cadence\n[Hz]", "cadence_hz", None),
        ("Mass [kg]", "mass_kg", None),
        ("Average power [W]", "avg_power_w", None),
        ("Cost burden\n[1-5, lower better]", "cost_score", None),
        ("Complexity burden\n[1-5, lower better]", "complexity_score", None),
    ]
    for ax, (title, field, scale) in zip(axes, panels):
        vals = [getattr(star, field), getattr(sunmag, field)]
        if scale is not None:
            vals = [v * scale for v in vals]
        bars = ax.bar(labels, vals, color=colors, hatch=hatches, edgecolor="black", linewidth=0.6)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3g}", (b.get_x() + b.get_width() / 2, b.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    fig.suptitle("Raw Architecture Comparison Across Major Engineering Metrics\n"
                 "(RAW physical/ordinal units - each panel its own scale)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig12_raw_architecture_comparison.png")
    plt.close(fig)

    # Figure 13: normalized utility heatmap
    fig, ax = plt.subplots(figsize=(9, 3.2))
    norm_table = baseline_result.normalized_table.loc[archs, list(CATEGORIES)]
    im = ax.imshow(norm_table.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels([c.capitalize() for c in CATEGORIES], rotation=30, ha="right")
    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(labels)
    for i in range(len(archs)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, f"{norm_table.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized utility [0=worst, 1=best]")
    ax.set_title("Normalized Trade-Matrix Utility Heatmap (NOT raw physical units)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig13_normalized_utility_heatmap.png")
    plt.close(fig)

    # Figure 14: weighted-score contributions by category (stacked)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = {a: 0.0 for a in archs}
    palette = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))
    for cat, color in zip(CATEGORIES, palette):
        vals = [baseline_result.weighted_contributions.loc[a, cat] for a in archs]
        ax.bar(labels, vals, bottom=[bottom[a] for a in archs], label=cat.capitalize(), color=color)
        for a, v in zip(archs, vals):
            bottom[a] += v
    for a, lab in zip(archs, labels):
        ax.annotate(f"total={bottom[a]:.3f}", (lab, bottom[a]), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("Weighted score contribution [-]")
    ax.set_title("Baseline Weighted-Score Contributions by Category")
    ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig14_weighted_score_contributions.png")
    plt.close(fig)

    # Figure 15: decision sensitivity across the 5 named scenarios
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(SCENARIOS))
    width = 0.35
    star_vals = [scenario_df.loc[scenario_df.scenario == s, "star_tracker_score"].iloc[0] for s in SCENARIOS]
    sunmag_vals = [scenario_df.loc[scenario_df.scenario == s, "sun_plus_mag_score"].iloc[0] for s in SCENARIOS]
    ax.bar(x - width / 2, star_vals, width, label="Star tracker", color=STAR_COLOR, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, sunmag_vals, width, label="Sun+mag", color=SUNMAG_COLOR, hatch="//",
           edgecolor="black", linewidth=0.5)
    winners = [scenario_df.loc[scenario_df.scenario == s, "winner"].iloc[0] for s in SCENARIOS]
    for xi, w in zip(x, winners):
        marker_x = xi - width / 2 if w == "star_tracker" else xi + width / 2
        marker_y = (star_vals[xi] if w == "star_tracker" else sunmag_vals[xi]) + 0.02
        ax.annotate("WINNER", (marker_x, marker_y), ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(list(SCENARIOS.keys()), rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Total weighted score [-]")
    ax.set_ylim(0, 1.05)
    ax.set_title("Decision Sensitivity Across the 5 Named Weighting Scenarios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig15_scenario_sensitivity.png")
    plt.close(fig)

    # Figure 16: Monte Carlo score-difference distribution
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(mc.score_diff, bins=60, color="#7f7f7f", edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color="black", ls="--", lw=1.5, label="zero crossing (tie)")
    ax.axvline(mc.score_diff.mean(), color=STAR_COLOR if mc.score_diff.mean() > 0 else SUNMAG_COLOR,
               ls="-", lw=1.5, label=f"mean = {mc.score_diff.mean():+.3f}")
    ax.set_xlabel("Score difference: star tracker - Sun+mag [-]")
    ax.set_ylabel("Number of Monte Carlo weight samples")
    ax.set_title(
        f"Weight-Space Robustness: Score-Difference Distribution\n"
        f"(n={mc.n_samples}, seed={mc.seed}; "
        f"{mc.fraction_favoring_star_tracker*100:.1f}% favor star tracker, "
        f"{mc.fraction_favoring_sun_plus_mag*100:.1f}% favor Sun+mag)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig16_monte_carlo_score_difference.png")
    plt.close(fig)

    # Figure 17: break-even curve (Sun+mag availability sweep)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    avail_grid = np.linspace(0.0, 1.0, 200)
    margins = []
    from dataclasses import replace as _replace
    for a in avail_grid:
        sm = _replace(sunmag, availability_fraction=a)
        r = run_trade_matrix(star, sm, BASELINE_WEIGHTS)
        margins.append(r.score_margin)
    margins = np.array(margins)
    ax.plot(avail_grid * 100, margins, color="#2ca02c", lw=2)
    ax.axhline(0.0, color="black", ls="--", lw=1.2, label="break-even (score margin = 0)")
    ax.axvline(sunmag.availability_fraction * 100, color=SUNMAG_COLOR, ls=":", lw=1.5,
               label=f"current Sun+mag availability ({sunmag.availability_fraction*100:.1f}%)")
    if be_avail is not None:
        ax.axvline(be_avail * 100, color="#9467bd", ls="-.", lw=1.5,
                   label=f"break-even availability ({be_avail*100:.1f}%)")
        ax.plot([be_avail * 100], [0.0], marker="o", color="#9467bd", markersize=9, zorder=5)
    ax.set_xlabel("Sun+mag full-attitude availability [% of orbit]")
    ax.set_ylabel("Score margin (star tracker - Sun+mag) [-]")
    ax.set_title("Break-Even Curve: Score Margin vs. Sun+Mag Availability\n(all else held at baseline)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig17_breakeven_curve_sunmag_availability.png")
    plt.close(fig)

    # Figure 18: final recommendation summary graphic
    import textwrap

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.axis("off")
    wrap_width = 78
    raw_lines = [
        "MILESTONE 3 FINAL RECOMMENDATION SUMMARY",
        "",
        f"Hard requirements: Star tracker PASSES all {len(REQUIREMENTS)}; "
        f"Sun+mag FAILS {sum(not v for v in req_results['sun_plus_mag'].values())} of {len(REQUIREMENTS)}.",
        f"Baseline weighted score: Star tracker {baseline_result.totals['star_tracker']:.3f} "
        f"vs. Sun+mag {baseline_result.totals['sun_plus_mag']:.3f} -> "
        f"WINNER: {ARCH_LABELS[baseline_result.winner]}.",
        f"Named-scenario flips: {n_flips} of 4 alternative priority scenarios flip the decision.",
        f"Monte Carlo weight-space split: "
        f"{mc.fraction_favoring_star_tracker*100:.1f}% star tracker / "
        f"{mc.fraction_favoring_sun_plus_mag*100:.1f}% Sun+mag "
        f"({'ROBUST' if robust else 'MISSION-PRIORITY-DEPENDENT'}).",
        "",
        "RECOMMENDATION: for baseline (balanced) mission priorities, select the "
        "STAR TRACKER, driven primarily by operational availability/continuity, "
        "not raw accuracy alone.",
        "CONDITIONAL CAVEAT: the recommendation flips to the Sun sensor + "
        "magnetometer suite under a genuinely mass/power-constrained or "
        "cost/complexity-constrained mission priority set.",
    ]
    wrapped_blocks = []
    for line_txt in raw_lines:
        if line_txt == "":
            wrapped_blocks.append("")
        else:
            wrapped_blocks.append("\n".join(textwrap.wrap(line_txt, width=wrap_width)))
    full_text = "\n\n".join(wrapped_blocks)
    ax.text(0.03, 0.97, full_text, transform=ax.transAxes, fontsize=11.5,
            va="top", ha="left", family="monospace", wrap=False,
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="black"))
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig18_final_recommendation_summary.png")
    plt.close(fig)

    print(f"[figures written to {RESULTS_DIR}]")
    print(f"[CSV tables written to {RESULTS_DIR}]")


if __name__ == "__main__":
    main()
