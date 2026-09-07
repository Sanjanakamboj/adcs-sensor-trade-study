#!/usr/bin/env python3
"""Milestone 2 verification / analysis script: sensor AVAILABILITY study.

Generates:
  - a numerical report printed to stdout AND written to results/m2_report.txt
  - professional figures (PNG) saved under results/ (fig6-fig11)

Everything here is DETERMINISTIC - no Monte Carlo, no randomness anywhere
(every sensor ``.measure()`` call uses ``rng=None``). Running this script
twice produces bit-for-bit identical numeric outputs and visually identical
figures.

Scope reminder (Milestone 2): this script characterizes ORBIT/TIME-BASED
SENSOR AVAILABILITY and asynchronous UPDATE CADENCE, building on the
Milestone 1 sensor-physics and instantaneous-geometry results. It still
does NOT build a weighted trade matrix, does NOT model mass/power/cost,
and does NOT make a final sensor-suite recommendation - see
docs/availability_methodology.md and the "written engineering conclusions"
section of the printed report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from adcs_sensor_trade.availability import (  # noqa: E402
    OrbitAvailabilityConfig,
    compute_full_attitude_update_cadence,
    default_sync_window_s,
    outage_intervals,
    run_availability_timeline,
    sensitivity_eclipse_beta_angle,
    sensitivity_exclusion_half_angle,
    sensitivity_fov_half_angle,
    sensitivity_update_rate,
    star_tracker_usable_updates,
    summarize_boolean_series,
)
from adcs_sensor_trade.orbit_environment import (  # noqa: E402
    EARTH_RADIUS_KM,
    eclipse_fraction,
)

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 20240607  # fixed seed for reproducibility (unused by any RNG here; see docs)

# Plot style: match the Milestone 1 script for visual consistency.
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

STAR_COLOR = "#1f77b4"
SUN_COLOR = "#ff7f0e"
MAG_COLOR = "#17becf"
SUNMAG_COLOR = "#d62728"


def line(char: str = "-", n: int = 78) -> str:
    return char * n


def main() -> None:
    np.random.default_rng(RANDOM_SEED)  # fixed for reproducibility (no RNG actually used)

    report_lines: list[str] = []

    def p(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    # -----------------------------------------------------------------
    # Baseline configuration
    # -----------------------------------------------------------------
    cfg = OrbitAvailabilityConfig(dt_s=1.0, n_orbits=1.0)
    period_min = cfg.period_s / 60.0
    ecl_frac = eclipse_fraction(cfg.altitude_km, cfg.beta_angle_deg)
    sync_window_s = default_sync_window_s(
        cfg.sun_sensor_config.update_rate_hz, cfg.magnetometer_config.update_rate_hz
    )

    p(line("="))
    p("ADCS-06 Sensor Trade Study - Milestone 2 Availability Verification Report")
    p(line("="))
    p("Scope: orbit/time-based sensor AVAILABILITY and asynchronous UPDATE")
    p("CADENCE only. Builds on Milestone 1's sensor physics and instantaneous")
    p("attitude-information geometry (reused unmodified). No trade matrix, no")
    p("mass/power/cost modeling, no final sensor-suite recommendation.")
    p(f"Fixed random seed: {RANDOM_SEED} (informational only - nothing here uses RNG)")
    p()

    p(line())
    p("1. REPRESENTATIVE ORBIT/ENVIRONMENT ASSUMPTIONS (illustrative, not a")
    p("   specific mission orbit - see docs/availability_methodology.md)")
    p(line())
    p(f"  Altitude                    = {cfg.altitude_km:.1f} km (circular, two-body Kepler)")
    p(f"  Orbital period              = {cfg.period_s:.1f} s ({period_min:.2f} min)")
    p(f"  Beta angle                  = {cfg.beta_angle_deg:.1f} deg")
    p(f"  Eclipse fraction (closed-form) = {ecl_frac:.4f} ({ecl_frac * 100:.2f}% of orbit)")
    p(f"  Attitude profile            = idealized nadir-pointing (LVLH): +Z nadir,")
    p("                                 +X velocity direction, +Y ~orbit-normal")
    p("  Star-tracker mounting       = anti-nadir (deep-space, -Z) boresight")
    p(
        f"  Star-tracker exclusion half-angle = "
        f"{np.degrees(cfg.star_tracker_config.exclusion_half_angle_rad):.1f} deg"
    )
    p(f"  Sun-sensor FOV half-angle   = {cfg.sun_sensor_config.fov_half_angle_deg:.1f} deg")
    p(f"  Magnetic field model        = tilted dipole DIRECTION, {cfg.dip_tilt_deg:.1f} deg tilt,")
    p(f"                                 constant magnitude {cfg.b_field_magnitude_tesla:.2e} T")
    p(
        f"  Update rates                 = star tracker {cfg.star_tracker_config.update_rate_hz:.1f} Hz, "
        f"Sun sensor {cfg.sun_sensor_config.update_rate_hz:.1f} Hz, "
        f"magnetometer {cfg.magnetometer_config.update_rate_hz:.1f} Hz"
    )
    p(f"  Sun+mag synchronization window = {sync_window_s * 1000:.1f} ms (half the slower sensor's period)")
    p(f"  Conditioning threshold      = cond(J) <= {cfg.condition_number_threshold:.1f} => 'well-conditioned'")
    p(f"                                 (sqrt({cfg.condition_number_threshold:.0f}) = "
      f"{np.sqrt(cfg.condition_number_threshold):.2f}x worst/best-axis error ratio)")
    p()

    # -----------------------------------------------------------------
    # Single-orbit availability timeline
    # -----------------------------------------------------------------
    df = run_availability_timeline(cfg)

    p(line())
    p("2. ONE-ORBIT AVAILABILITY SUMMARY")
    p(line())

    summaries = {}
    for label, col in [
        ("Star tracker", "star_tracker_available"),
        ("Sun sensor", "sun_sensor_available"),
        ("Magnetometer", "magnetometer_available"),
        ("Sun+mag (full attitude)", "sunmag_available"),
    ]:
        s = summarize_boolean_series(df[col].values, cfg.dt_s)
        summaries[col] = s
        p(
            f"  {label:26s}: available {s['fraction_available'] * 100:6.2f}% of orbit | "
            f"longest outage = {s['longest_outage_s']:7.1f} s "
            f"({s['longest_outage_s'] / 60:5.2f} min) | "
            f"num outages = {s['num_outages']}"
        )
    p()

    p("  Star-tracker unavailability cause breakdown:")
    for reason, count in df["star_tracker_reason"].value_counts().items():
        p(f"    {reason:15s}: {count / len(df) * 100:6.2f}% of samples")
    p("  Sun-sensor unavailability cause breakdown:")
    for reason, count in df["sun_sensor_reason"].value_counts().items():
        p(f"    {reason:15s}: {count / len(df) * 100:6.2f}% of samples")
    p("  Sun+mag unavailability cause breakdown:")
    for reason, count in df["sunmag_reason"].value_counts().items():
        p(f"    {reason:15s}: {count / len(df) * 100:6.2f}% of samples")
    p()

    # Longest-outage root cause (dominant reason during the longest interval)
    def longest_outage_reason(col_available: str, col_reason: str) -> str:
        intervals = outage_intervals(df[col_available].values, cfg.dt_s)
        if not intervals:
            return "N/A (no outages)"
        longest = max(intervals, key=lambda iv: iv["duration_s"])
        reasons = df[col_reason].values[longest["start_index"]:longest["end_index"]]
        vals, counts = np.unique(reasons, return_counts=True)
        return str(vals[np.argmax(counts)])

    p("  Longest-outage root cause:")
    p(f"    Star tracker : {longest_outage_reason('star_tracker_available', 'star_tracker_reason')}")
    p(f"    Sun sensor   : {longest_outage_reason('sun_sensor_available', 'sun_sensor_reason')}")
    p(f"    Sun+mag      : {longest_outage_reason('sunmag_available', 'sunmag_reason')}")
    p()

    # -----------------------------------------------------------------
    # Quality-while-available statistics
    # -----------------------------------------------------------------
    p(line())
    p("3. SUN+MAG ATTITUDE-INFORMATION QUALITY WHILE GEOMETRICALLY AVAILABLE")
    p(line())
    usable_full_rank = df[df["sunmag_full_rank"]]
    cond_vals = usable_full_rank["sunmag_condition_number"].values
    n_usable = len(usable_full_rank)
    n_well_cond = int(usable_full_rank["sunmag_well_conditioned"].sum())
    pct_poor_of_usable = 100.0 * (1.0 - n_well_cond / n_usable) if n_usable > 0 else float("nan")

    p(f"  Samples with Sun+mag vectors usable AND full-rank: {n_usable} of {len(df)} "
      f"({n_usable / len(df) * 100:.2f}% of orbit)")
    if n_usable > 0:
        p(f"    condition number: min={cond_vals.min():.3f}  median={np.median(cond_vals):.3f}  "
          f"P95={np.percentile(cond_vals, 95):.3f}  max={cond_vals.max():.3f}")
        worst_axis = usable_full_rank["sunmag_worst_axis_error_deg"].values
        p(f"    worst-axis error proxy [deg]: min={worst_axis.min():.4f}  "
          f"median={np.median(worst_axis):.4f}  P95={np.percentile(worst_axis, 95):.4f}  "
          f"max={worst_axis.max():.4f}")
        p(f"    fraction of 'full-rank' time that is POORLY conditioned "
          f"(cond > {cfg.condition_number_threshold:.1f}): {pct_poor_of_usable:.2f}%")
    p()

    # -----------------------------------------------------------------
    # Update-cadence analysis
    # -----------------------------------------------------------------
    p(line())
    p("4. ASYNCHRONOUS UPDATE-RATE / CADENCE ANALYSIS")
    p(line())
    cad = compute_full_attitude_update_cadence(cfg)
    star_cad = star_tracker_usable_updates(cfg)
    p(f"  Star tracker (direct full attitude, own {cfg.star_tracker_config.update_rate_hz:.1f} Hz rate):")
    p(f"    usable full-attitude updates/orbit = {star_cad['updates_per_orbit']:.1f}  "
      f"(effective {star_cad['cadence_hz']:.4f} Hz)")
    p(
        f"  Sun+mag (async {cfg.sun_sensor_config.update_rate_hz:.1f} Hz / "
        f"{cfg.magnetometer_config.update_rate_hz:.1f} Hz, sync window {sync_window_s * 1000:.1f} ms):"
    )
    p(f"    usable full-attitude updates/orbit = {cad['updates_per_orbit']:.1f}  "
      f"(effective {cad['cadence_hz']:.4f} Hz)")
    p()

    # Does a faster magnetometer alone help?
    from dataclasses import replace

    from adcs_sensor_trade.sensors.magnetometer import MagnetometerConfig

    cfg_fast_mag = replace(
        cfg, magnetometer_config=MagnetometerConfig(update_rate_hz=cfg.magnetometer_config.update_rate_hz * 5)
    )
    cad_fast_mag = compute_full_attitude_update_cadence(cfg_fast_mag)
    p(f"  Sensitivity check: raising magnetometer rate {cfg.magnetometer_config.update_rate_hz:.0f} Hz -> "
      f"{cfg_fast_mag.magnetometer_config.update_rate_hz:.0f} Hz (5x) with Sun-sensor rate UNCHANGED:")
    p(f"    usable full-attitude updates/orbit: {cad['updates_per_orbit']:.1f} -> "
      f"{cad_fast_mag['updates_per_orbit']:.1f} "
      f"(change: {cad_fast_mag['updates_per_orbit'] - cad['updates_per_orbit']:+.1f})")
    p("    Conclusion: a faster magnetometer ALONE does not materially raise the usable")
    p("    full-attitude update cadence, because a magnetometer measurement alone is")
    p("    rank-2 (see information.py) - the Sun sensor's own availability/rate remains")
    p("    the bottleneck for Sun+mag full-attitude updates.")
    p()

    # -----------------------------------------------------------------
    # Sensitivity studies
    # -----------------------------------------------------------------
    p(line())
    p("5. SENSITIVITY STUDIES (all deterministic)")
    p(line())

    df_excl = sensitivity_exclusion_half_angle(cfg, np.array([5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0]))
    p("  (a) Star-tracker availability vs. exclusion half-angle:")
    for _, row in df_excl.iterrows():
        p(f"      {row['exclusion_half_angle_deg']:5.1f} deg -> "
          f"{row['star_tracker_availability_fraction'] * 100:6.2f}% available")
    excl_diffs = np.diff(df_excl["star_tracker_availability_fraction"].values)
    p(f"      Monotonic non-increasing: {bool(np.all(excl_diffs <= 1e-9))}")
    p()

    df_fov = sensitivity_fov_half_angle(cfg, np.array([15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 179.0]))
    p("  (b) Sun-sensor / Sun+mag availability vs. FOV half-angle:")
    for _, row in df_fov.iterrows():
        p(f"      {row['fov_half_angle_deg']:5.1f} deg -> "
          f"sun={row['sun_sensor_availability_fraction'] * 100:6.2f}%  "
          f"sunmag={row['sunmag_availability_fraction'] * 100:6.2f}%")
    fov_diffs = np.diff(df_fov["sun_sensor_availability_fraction"].values)
    p(f"      Monotonic non-decreasing (Sun sensor): {bool(np.all(fov_diffs >= -1e-9))}")
    p()

    df_beta = sensitivity_eclipse_beta_angle(cfg, np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]))
    p("  (c) Eclipse fraction / Sun-sensor / Sun+mag availability vs. beta angle:")
    for _, row in df_beta.iterrows():
        p(f"      beta={row['beta_angle_deg']:5.1f} deg -> eclipse={row['eclipse_fraction'] * 100:5.2f}%  "
          f"sun={row['sun_sensor_availability_fraction'] * 100:6.2f}%  "
          f"sunmag={row['sunmag_availability_fraction'] * 100:6.2f}%")
    beta_diffs = np.diff(df_beta["eclipse_fraction"].values)
    p(f"      Eclipse fraction monotonic non-increasing in beta: {bool(np.all(beta_diffs <= 1e-12))}")
    p()

    df_rate = sensitivity_update_rate(
        cfg,
        star_rates_hz=np.array([0.5, 1.0, 2.0, 4.0, 8.0]),
        sun_rates_hz=np.array([1.0, 2.0, 5.0, 10.0, 20.0]),
        mag_rates_hz=np.array([5.0, 10.0, 20.0, 50.0]),
    )
    p("  (d) Usable full-attitude updates/orbit vs. sensor update rate:")
    for _, row in df_rate.iterrows():
        p(f"      {row['parameter']:22s} = {row['rate_hz']:6.1f} Hz -> "
          f"{row['updates_per_orbit']:8.1f} updates/orbit")
    star_rate_vals = df_rate[df_rate["parameter"] == "star_tracker_rate_hz"]["updates_per_orbit"].values
    sun_rate_vals = df_rate[df_rate["parameter"] == "sun_sensor_rate_hz"]["updates_per_orbit"].values
    mag_rate_vals = df_rate[df_rate["parameter"] == "magnetometer_rate_hz"]["updates_per_orbit"].values
    p(f"      Star-tracker-rate updates monotonic increasing: {bool(np.all(np.diff(star_rate_vals) > 0))}")
    p(f"      Sun-sensor-rate updates monotonic increasing: {bool(np.all(np.diff(sun_rate_vals) > 0))}")
    p(f"      Magnetometer-rate updates essentially FLAT (by construction): "
      f"{bool(np.ptp(mag_rate_vals) < 1e-6)}")
    p()

    # -----------------------------------------------------------------
    # Answers to the milestone questions
    # -----------------------------------------------------------------
    p(line("="))
    p("6. ANSWERS TO MILESTONE 2 QUESTIONS")
    p(line("="))
    star_frac = summaries["star_tracker_available"]["fraction_available"] * 100
    sun_frac = summaries["sun_sensor_available"]["fraction_available"] * 100
    sunmag_frac = summaries["sunmag_available"]["fraction_available"] * 100
    p(
        f"  Q1. Fraction of orbit each architecture can do a full attitude observation:\n"
        f"      Star tracker = {star_frac:.2f}%   Sun+mag = {sunmag_frac:.2f}%\n"
        f"      (Sun sensor ALONE = {sun_frac:.2f}% of orbit geometrically visible, but a\n"
        f"       single Sun-sensor vector is only rank-2 - it never gives full attitude\n"
        f"       by itself; magnetometer is modeled as always geometrically 'available'\n"
        f"       but likewise only rank-2 alone.)"
    )
    p(
        f"\n  Q2. Longest-outage root cause:\n"
        f"      Star tracker: {longest_outage_reason('star_tracker_available', 'star_tracker_reason')} "
        f"({summaries['star_tracker_available']['longest_outage_s']:.1f} s)\n"
        f"      Sun sensor:   {longest_outage_reason('sun_sensor_available', 'sun_sensor_reason')} "
        f"({summaries['sun_sensor_available']['longest_outage_s']:.1f} s)\n"
        f"      Sun+mag:      {longest_outage_reason('sunmag_available', 'sunmag_reason')} "
        f"({summaries['sunmag_available']['longest_outage_s']:.1f} s)"
    )
    p(
        f"\n  Q3. Fraction of Sun+mag 'geometrically usable' time that is poorly conditioned:\n"
        f"      {pct_poor_of_usable:.2f}% of the full-rank-usable time "
        f"(cond(J) > {cfg.condition_number_threshold:.1f})."
    )
    p(
        f"\n  Q4. Effective Sun+mag full-attitude update cadence:\n"
        f"      {cad['updates_per_orbit']:.1f} updates/orbit ({cad['cadence_hz']:.4f} Hz effective),\n"
        f"      vs. star tracker direct {star_cad['updates_per_orbit']:.1f} updates/orbit "
        f"({star_cad['cadence_hz']:.4f} Hz)."
    )
    p(
        "\n  Q5. Star-tracker availability sensitivity to exclusion angle:\n"
        f"      monotonically non-increasing, from {df_excl['star_tracker_availability_fraction'].iloc[0]*100:.1f}% "
        f"at {df_excl['exclusion_half_angle_deg'].iloc[0]:.0f} deg down to "
        f"{df_excl['star_tracker_availability_fraction'].iloc[-1]*100:.1f}% at "
        f"{df_excl['exclusion_half_angle_deg'].iloc[-1]:.0f} deg."
    )
    p(
        "\n  Q6. Sun+mag availability sensitivity to eclipse fraction (beta) and FOV:\n"
        f"      Widening FOV from {df_fov['fov_half_angle_deg'].iloc[0]:.0f} to "
        f"{df_fov['fov_half_angle_deg'].iloc[-1]:.0f} deg raises Sun+mag availability from "
        f"{df_fov['sunmag_availability_fraction'].iloc[0]*100:.1f}% to "
        f"{df_fov['sunmag_availability_fraction'].iloc[-1]*100:.1f}%.\n"
        f"      Raising beta from {df_beta['beta_angle_deg'].iloc[0]:.0f} to "
        f"{df_beta['beta_angle_deg'].iloc[-1]:.0f} deg drops eclipse fraction from "
        f"{df_beta['eclipse_fraction'].iloc[0]*100:.1f}% to "
        f"{df_beta['eclipse_fraction'].iloc[-1]*100:.1f}%, and (in this nadir-pointing/velocity-\n"
        f"      boresight geometry) Sun+mag availability moves from "
        f"{df_beta['sunmag_availability_fraction'].iloc[0]*100:.1f}% to "
        f"{df_beta['sunmag_availability_fraction'].iloc[-1]*100:.1f}% (FOV-geometry coupling can\n"
        f"      dominate the eclipse effect at high beta - see docs/availability_methodology.md)."
    )
    p(
        "\n  Q7. Does the magnetometer's higher rate alone materially improve full-attitude\n"
        "      update availability? NO - confirmed above: usable updates/orbit stayed at\n"
        f"      {mag_rate_vals[0]:.1f} across magnetometer rates "
        f"{df_rate[df_rate['parameter']=='magnetometer_rate_hz']['rate_hz'].min():.0f}-"
        f"{df_rate[df_rate['parameter']=='magnetometer_rate_hz']['rate_hz'].max():.0f} Hz. This is BY\n"
        "      CONSTRUCTION: a magnetometer measurement alone is rank-2 (geometry.py), so no\n"
        "      amount of extra magnetometer samples recovers the missing 3rd degree of\n"
        "      freedom - only a second, non-collinear, AVAILABLE vector (the Sun sensor,\n"
        "      whose own eclipse/FOV-limited availability is the true bottleneck) can."
    )
    p(
        f"\n  Q8. Stronger operational availability/continuity (availability + quality ONLY,\n"
        f"      not mass/power/cost): STAR TRACKER - {star_frac:.2f}% available with "
        f"{summaries['star_tracker_available']['longest_outage_s']:.0f} s longest outage,\n"
        f"      vs. Sun+mag at {sunmag_frac:.2f}% available with "
        f"{summaries['sunmag_available']['longest_outage_s']:.0f} s longest outage. The star\n"
        "      tracker's ONLY outage driver in this model is Sun-exclusion geometry; Sun+mag\n"
        "      additionally loses availability to eclipse and Sun-sensor FOV geometry."
    )
    p(
        "\n  Q9. What limitations prevent M2 alone from making the final sensor-suite\n"
        "      recommendation?\n"
        "      - No mass, power, or cost modeling (a star tracker is typically heavier,\n"
        "        more power-hungry, and more expensive than a Sun sensor + magnetometer).\n"
        "      - No full gyro-propagated estimator (MEKF/EKF) - outages here are treated as\n"
        "        instantaneous measurement gaps, not propagated through a real attitude\n"
        "        filter with process noise and gyro drift between updates.\n"
        "      - No closed-loop ADCS/control-bandwidth analysis of what update cadence is\n"
        "        actually SUFFICIENT for a given mission's pointing requirements.\n"
        "      - Illustrative orbit/attitude/eclipse/field models (circular orbit, fixed\n"
        "        nadir-pointing profile, cylindrical shadow, tilted-dipole field direction),\n"
        "        not high-fidelity ephemeris or a full IGRF geomagnetic field model.\n"
        "      - No vendor/product-specific hardware qualification claims."
    )
    p()
    p(line("*"))
    p("* NO FINAL SENSOR-SUITE RECOMMENDATION IS MADE IN THIS MILESTONE. *")
    p("* Milestone 3 is expected to add mass/power/cost modeling and the      *")
    p("* eventual weighted trade matrix + final sensor-suite recommendation. *")
    p(line("*"))

    # -----------------------------------------------------------------
    # Write text report
    # -----------------------------------------------------------------
    report_path = RESULTS_DIR / "m2_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\n[report written to {report_path}]")

    # ===================================================================
    # Figures
    # ===================================================================

    t_min = df["t_s"].values / 60.0

    # Figure 6: one-orbit availability timeline (step plot, 4 architectures)
    fig, axes = plt.subplots(4, 1, figsize=(10, 7), sharex=True)
    series = [
        ("Star tracker", df["star_tracker_available"].values, STAR_COLOR),
        ("Sun sensor", df["sun_sensor_available"].values, SUN_COLOR),
        ("Magnetometer", df["magnetometer_available"].values, MAG_COLOR),
        ("Sun+mag\n(full attitude)", df["sunmag_available"].values, SUNMAG_COLOR),
    ]
    for ax, (label, vals, color) in zip(axes, series):
        ax.fill_between(t_min, 0, vals.astype(float), step="post", color=color, alpha=0.7)
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["unavail.", "avail."])
        ax.set_ylabel(label, fontsize=9)
    axes[-1].set_xlabel("Time since epoch [min]")
    fig.suptitle("One-Orbit Sensor-Availability Timeline (600 km, beta = 20 deg)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig6_availability_timeline.png")
    plt.close(fig)

    # Figure 7: Sun-mag separation angle and condition number vs time
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax1.plot(t_min, df["separation_angle_deg"].values, color="#9467bd", lw=1.8)
    ax1.set_ylabel("Sun-mag separation angle [deg]")
    ax1.set_title("Sun-Magnetic Separation Angle Over One Orbit")

    cond_plot = df["sunmag_condition_number"].astype(float).values
    ax2.plot(t_min, cond_plot, color=SUNMAG_COLOR, lw=1.5)
    ax2.axhline(cfg.condition_number_threshold, color="black", ls="--", lw=1.2,
                label=f"well-conditioned threshold ({cfg.condition_number_threshold:.0f})")
    ax2.set_yscale("log")
    ax2.set_ylabel("Sun+mag information condition number [-]")
    ax2.set_xlabel("Time since epoch [min]")
    ax2.set_title("Attitude-Information Conditioning Over One Orbit (gaps = sensor unavailable)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig7_separation_and_conditioning_vs_time.png")
    plt.close(fig)

    # Figure 8: architecture availability/outage comparison bar chart
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Star\ntracker", "Sun\nsensor", "Magneto-\nmeter", "Sun+mag\n(full attitude)"]
    frac_vals = [
        summaries["star_tracker_available"]["fraction_available"] * 100,
        summaries["sun_sensor_available"]["fraction_available"] * 100,
        summaries["magnetometer_available"]["fraction_available"] * 100,
        summaries["sunmag_available"]["fraction_available"] * 100,
    ]
    outage_vals = [
        summaries["star_tracker_available"]["longest_outage_s"] / 60.0,
        summaries["sun_sensor_available"]["longest_outage_s"] / 60.0,
        summaries["magnetometer_available"]["longest_outage_s"] / 60.0,
        summaries["sunmag_available"]["longest_outage_s"] / 60.0,
    ]
    colors = [STAR_COLOR, SUN_COLOR, MAG_COLOR, SUNMAG_COLOR]
    axA.bar(labels, frac_vals, color=colors)
    axA.set_ylabel("Availability [% of orbit]")
    axA.set_title("Availability Fraction")
    axA.set_ylim(0, 105)
    axB.bar(labels, outage_vals, color=colors)
    axB.set_ylabel("Longest continuous outage [min]")
    axB.set_title("Longest Outage")
    fig.suptitle("Architecture Availability / Outage Comparison (One Orbit)", fontsize=13)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig8_architecture_comparison.png")
    plt.close(fig)

    # Figure 9: star-tracker exclusion-angle sensitivity
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df_excl["exclusion_half_angle_deg"], df_excl["star_tracker_availability_fraction"] * 100,
        color=STAR_COLOR, marker="o", lw=2,
    )
    ax.axvline(
        np.degrees(cfg.star_tracker_config.exclusion_half_angle_rad), color="black", ls=":", lw=1.2,
        label="baseline exclusion half-angle",
    )
    ax.set_xlabel("Bright-source exclusion half-angle [deg]")
    ax.set_ylabel("Star-tracker availability [% of orbit]")
    ax.set_title("Star-Tracker Availability Sensitivity to Exclusion Half-Angle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig9_exclusion_angle_sensitivity.png")
    plt.close(fig)

    # Figure 10: Sun-sensor FOV and eclipse (beta) sensitivity
    #
    # NOTE: in this baseline configuration, Sun+mag availability tracks the
    # Sun sensor's availability almost exactly (whenever the Sun sensor can
    # see the Sun, the resulting Sun-mag geometry also happens to be
    # full-rank and well-conditioned) - so the two curves are visually
    # near-identical. Drawn with a thick pale underlay + thin overlay so
    # BOTH are visibly present rather than one silently hiding the other.
    fig, (axF, axE) = plt.subplots(1, 2, figsize=(12, 5))
    axF.plot(df_fov["fov_half_angle_deg"], df_fov["sun_sensor_availability_fraction"] * 100,
             color=SUN_COLOR, marker="o", lw=6, alpha=0.5, label="Sun sensor")
    axF.plot(df_fov["fov_half_angle_deg"], df_fov["sunmag_availability_fraction"] * 100,
             color=SUNMAG_COLOR, marker="s", lw=1.5, ls="--", label="Sun+mag")
    axF.axvline(cfg.sun_sensor_config.fov_half_angle_deg, color="black", ls=":", lw=1.2, label="baseline FOV")
    axF.set_xlabel("Sun-sensor FOV half-angle [deg]")
    axF.set_ylabel("Availability [% of orbit]")
    axF.set_title("Sensitivity to FOV Half-Angle\n(curves overlap: Sun+mag tracks Sun-sensor availability here)")
    axF.legend(fontsize=8)

    axE.plot(df_beta["beta_angle_deg"], df_beta["eclipse_fraction"] * 100,
              color="#2ca02c", marker="^", lw=2, label="Eclipse fraction")
    axE.plot(df_beta["beta_angle_deg"], df_beta["sunmag_availability_fraction"] * 100,
              color=SUNMAG_COLOR, marker="s", lw=2, label="Sun+mag availability")
    axE.axvline(cfg.beta_angle_deg, color="black", ls=":", lw=1.2, label="baseline beta")
    axE.set_xlabel("Beta angle [deg]")
    axE.set_ylabel("Percent of orbit [%]")
    axE.set_title("Sensitivity to Beta Angle (Eclipse Geometry)")
    axE.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig10_fov_and_eclipse_sensitivity.png")
    plt.close(fig)

    # Figure 11: update-rate / usable-update-cadence trade
    fig, ax = plt.subplots(figsize=(9, 5.5))
    star_rows = df_rate[df_rate["parameter"] == "star_tracker_rate_hz"]
    sun_rows = df_rate[df_rate["parameter"] == "sun_sensor_rate_hz"]
    mag_rows = df_rate[df_rate["parameter"] == "magnetometer_rate_hz"]
    ax.plot(star_rows["rate_hz"], star_rows["updates_per_orbit"], color=STAR_COLOR, marker="o", lw=2,
            label="Star-tracker rate (star tracker alone)")
    ax.plot(sun_rows["rate_hz"], sun_rows["updates_per_orbit"], color=SUN_COLOR, marker="s", lw=2,
            label="Sun-sensor rate (Sun+mag)")
    ax.plot(mag_rows["rate_hz"], mag_rows["updates_per_orbit"], color=MAG_COLOR, marker="^", lw=2,
            label="Magnetometer rate (Sun+mag) - essentially flat")
    ax.set_xlabel("Sensor update rate [Hz]")
    ax.set_ylabel("Usable full-attitude updates / orbit [-]")
    ax.set_title("Usable Full-Attitude Update Cadence vs. Sensor Update Rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig11_update_rate_cadence_tradeoff.png")
    plt.close(fig)

    print(f"[figures written to {RESULTS_DIR}]")


if __name__ == "__main__":
    main()
