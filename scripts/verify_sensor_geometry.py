#!/usr/bin/env python3
"""Milestone 1 verification / analysis script.

Generates:
  - a numerical report printed to stdout AND written to results/m1_report.txt
  - professional figures (PNG) saved under results/

Everything here is DETERMINISTIC — a fixed random seed is set for
reproducibility, though in fact none of the local information/geometry
proxies computed here use randomness at all (see docs/conventions.md).
Running this script twice produces bit-for-bit identical numeric outputs
and visually identical figures.

Scope reminder (Milestone 1): this script characterizes SENSOR PHYSICS and
INSTANTANEOUS ATTITUDE-INFORMATION GEOMETRY only. It does NOT build a
weighted trade matrix and does NOT recommend a sensor suite — see the
"Written engineering conclusions" section of the printed report.
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

from adcs_sensor_trade.geometry import (  # noqa: E402
    condition_number,
    geometry_rank,
    singular_values,
)
from adcs_sensor_trade.information import (  # noqa: E402
    combined_vector_information,
    star_tracker_information,
    summarize_information,
)
from adcs_sensor_trade.sensors.star_tracker import StarTrackerConfig  # noqa: E402
from adcs_sensor_trade.sensors.sun_sensor import SunSensorConfig  # noqa: E402
from adcs_sensor_trade.sensors.magnetometer import MagnetometerConfig  # noqa: E402
from adcs_sensor_trade.sweep import run_separation_sweep, sun_mag_body_vectors  # noqa: E402

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 20240607  # fixed seed for reproducibility (see docs/conventions.md)

# ---------------------------------------------------------------------------
# Representative baseline configuration (see docs/sensor_models.md)
# ---------------------------------------------------------------------------
star_cfg = StarTrackerConfig()
sun_cfg = SunSensorConfig()
mag_cfg = MagnetometerConfig()

SIGMA_STAR_RAD = star_cfg.sigma_rad
SIGMA_SUN_RAD = sun_cfg.sigma_rad
SIGMA_MAG_RAD = mag_cfg.sigma_rad

SUN_VECTOR_INERTIAL = np.array([1.0, 0.0, 0.0])
B_FIELD_VECTOR_INERTIAL_TESLA = np.array([0.0, 0.0, 3.0e-5])

# Plot style: clean, professional, readable.
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
SUNMAG_COLOR = "#d62728"


def line(char: str = "-", n: int = 78) -> str:
    return char * n


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)  # noqa: F841  (fixed for reproducibility)

    report_lines: list[str] = []

    def p(msg: str = "") -> None:
        print(msg)
        report_lines.append(msg)

    p(line("="))
    p("ADCS-06 Sensor Trade Study — Milestone 1 Verification Report")
    p(line("="))
    p("Scope: sensor physics models + instantaneous 3-axis attitude-information")
    p("geometry ONLY. No trade matrix, no final sensor-suite recommendation.")
    p(f"Fixed random seed: {RANDOM_SEED}")
    p()

    # -----------------------------------------------------------------
    # Section 1: baseline assumptions
    # -----------------------------------------------------------------
    p(line())
    p("1. REPRESENTATIVE BASELINE ASSUMPTIONS (illustrative, not vendor specs)")
    p(line())
    p(f"  Star tracker : sigma = {star_cfg.sigma_arcsec:.1f} arcsec (1-sigma/axis), "
      f"update rate = {star_cfg.update_rate_hz:.1f} Hz")
    p(f"  Sun sensor   : sigma = {sun_cfg.sigma_deg:.2f} deg (1-sigma), "
      f"FOV half-angle = {sun_cfg.fov_half_angle_deg:.1f} deg, "
      f"update rate = {sun_cfg.update_rate_hz:.1f} Hz")
    p(f"  Magnetometer : sigma = {mag_cfg.sigma_deg:.2f} deg (1-sigma), "
      f"update rate = {mag_cfg.update_rate_hz:.1f} Hz, "
      f"bias = {mag_cfg.bias_tesla} T, scale = {mag_cfg.scale_factor}")
    p(f"  Inertial Sun vector      : {SUN_VECTOR_INERTIAL}")
    p(f"  Inertial B-field vector  : {B_FIELD_VECTOR_INERTIAL_TESLA} T "
      f"(|B| = {np.linalg.norm(B_FIELD_VECTOR_INERTIAL_TESLA):.2e} T, representative LEO magnitude)")
    p()

    # -----------------------------------------------------------------
    # Section 2: single-vector rank deficiency
    # -----------------------------------------------------------------
    p(line())
    p("2. SINGLE-VECTOR RANK DEFICIENCY")
    p(line())
    sun_body_example = np.array([1.0, 0.0, 0.0])
    mag_body_example = np.array([0.0, 1.0, 0.0])
    star_rank_note = "N/A (star tracker measures full attitude directly, not a single vector)"
    p(f"  rank(single Sun vector)          = {geometry_rank([sun_body_example])}  (expected 2 of 3)")
    p(f"  rank(single magnetometer vector) = {geometry_rank([mag_body_example])}  (expected 2 of 3)")
    p(f"  rank(two orthogonal vectors)     = {geometry_rank([sun_body_example, mag_body_example])}  (expected 3 of 3)")
    p(f"  Star-tracker rank                = {star_rank_note}")
    p()

    # -----------------------------------------------------------------
    # Section 3: healthy-geometry comparison (90 deg separation)
    # -----------------------------------------------------------------
    p(line())
    p("3. HEALTHY-GEOMETRY COMPARISON (Sun-mag separation = 90 deg)")
    p(line())
    sun_body_90, mag_body_90 = sun_mag_body_vectors(90.0)
    J_star = star_tracker_information(SIGMA_STAR_RAD)
    star_report = summarize_information(J_star)
    J_sm_90 = combined_vector_information([sun_body_90, mag_body_90], [SIGMA_SUN_RAD, SIGMA_MAG_RAD])
    sm_report_90 = summarize_information(J_sm_90)

    p("  Star tracker (isotropic model):")
    p(f"    information eigenvalues [rad^-2] = {np.array2string(star_report.eigenvalues_J, precision=3)}")
    p(f"    condition number of J            = {star_report.condition_number_J:.3f}")
    p(f"    RMS attitude-error proxy         = {star_report.rms_attitude_error_deg * 3600:.2f} arcsec "
      f"({star_report.rms_attitude_error_deg:.6f} deg)")
    p(f"    worst-axis 1-sigma error proxy   = {star_report.worst_axis_error_deg * 3600:.2f} arcsec "
      f"({star_report.worst_axis_error_deg:.6f} deg)")
    p()
    p("  Sun + magnetometer (90 deg separation):")
    p(f"    information eigenvalues [rad^-2] = {np.array2string(sm_report_90.eigenvalues_J, precision=3)}")
    p(f"    condition number of J            = {sm_report_90.condition_number_J:.3f}")
    p(f"    RMS attitude-error proxy         = {sm_report_90.rms_attitude_error_deg:.4f} deg")
    p(f"    worst-axis 1-sigma error proxy   = {sm_report_90.worst_axis_error_deg:.4f} deg")
    p()

    # -----------------------------------------------------------------
    # Section 4: separation-angle sweep
    # -----------------------------------------------------------------
    p(line())
    p("4. SUN-MAGNETIC SEPARATION ANGLE SWEEP (1 to 179 deg)")
    p(line())
    angles_deg = np.arange(1.0, 179.01, 1.0)
    df = run_separation_sweep(angles_deg, SIGMA_SUN_RAD, SIGMA_MAG_RAD, SIGMA_STAR_RAD)

    for probe_angle in [5.0, 45.0, 90.0, 135.0, 175.0]:
        row = df.iloc[(df["separation_angle_deg"] - probe_angle).abs().idxmin()]
        p(
            f"  angle = {row['separation_angle_deg']:5.1f} deg | "
            f"cond(J) = {row['sunmag_condition_number_J']:10.3f} | "
            f"worst-axis error = {row['sunmag_worst_axis_error_deg']:7.4f} deg | "
            f"rank = {int(row['rank'])}"
        )
    p()
    p("  Singularity check at EXACT 0 deg and 180 deg:")
    df_sing = run_separation_sweep(np.array([0.0, 180.0]), SIGMA_SUN_RAD, SIGMA_MAG_RAD, SIGMA_STAR_RAD)
    for _, row in df_sing.iterrows():
        p(
            f"    angle = {row['separation_angle_deg']:5.1f} deg | rank = {int(row['rank'])} | "
            f"min singular value = {row['min_singular_value']:.3e} | "
            f"cond(J) = {row['sunmag_condition_number_J']}"
        )
    p()

    # -----------------------------------------------------------------
    # Section 5: scaling sanity checks
    # -----------------------------------------------------------------
    p(line())
    p("5. INFORMATION / COVARIANCE SCALING SANITY CHECKS")
    p(line())
    sigma_a = np.radians(20.0 / 3600.0)
    sigma_b = sigma_a / 2.0
    J_a = star_tracker_information(sigma_a)
    J_b = star_tracker_information(sigma_b)
    p(f"  Halving star-tracker sigma multiplies isotropic information by "
      f"{(J_b[0, 0] / J_a[0, 0]):.3f} (expected 4.0)")
    ra = summarize_information(J_a)
    rb = summarize_information(star_tracker_information(sigma_a * 2))
    p(f"  Doubling star-tracker sigma multiplies worst-axis error proxy by "
      f"{(rb.worst_axis_error_rad / ra.worst_axis_error_rad):.3f} (expected 2.0)")
    p()

    # -----------------------------------------------------------------
    # Section 6: written engineering conclusions
    # -----------------------------------------------------------------
    p(line("="))
    p("6. WRITTEN ENGINEERING CONCLUSIONS")
    p(line("="))
    p(
        "  Q: Why can a star tracker provide full instantaneous 3-axis attitude\n"
        "     information?\n"
        "  A: It does not measure a single line-of-sight vector — it triangulates\n"
        "     many catalog star positions in its focal plane simultaneously and\n"
        "     solves directly for a full attitude quaternion/DCM. Under the\n"
        "     isotropic simplification used here, its information matrix is\n"
        "     J_star = (1/sigma^2) I_3, full rank by construction."
    )
    p(
        "\n  Q: Why can a single Sun sensor not?\n"
        "  A: A Sun sensor outputs one body-frame unit vector. Its measurement\n"
        "     Jacobian H = -skew(v_b) has rank 2 (skew-symmetric matrices of a\n"
        "     nonzero vector are always rank-deficient by exactly one): rotation\n"
        "     of the body about the Sun-line itself changes nothing measurable."
    )
    p(
        "\n  Q: Why can a single magnetometer not?\n"
        "  A: Same argument as the Sun sensor — it is also a single unit-vector\n"
        "     measurement, so rotation about the local field-line direction is\n"
        "     unobservable from that one reading."
    )
    p(
        "\n  Q: Under what geometry does Sun + magnetometer recover full attitude?\n"
        "  A: Whenever the two vectors are non-collinear, the stacked 6x3 Jacobian\n"
        "     reaches rank 3, recovering full 3-axis instantaneous information.\n"
        "     Conditioning is BEST near 90 deg separation."
    )
    p(
        "\n  Q: When does that architecture become poorly conditioned?\n"
        "  A: As the Sun-magnetic separation angle approaches 0 deg or 180 deg,\n"
        f"     the condition number of J grows without bound; at exactly 0/180 deg\n"
        "     the geometry is singular (rank drops to 2, min singular value -> 0)."
    )
    p(
        "\n  Q: Which architecture has more uniform instantaneous attitude accuracy\n"
        "     under the simplified assumptions?\n"
        "  A: The star tracker: its isotropic model gives attitude information that\n"
        "     is, by construction, independent of geometry/attitude. The Sun+mag\n"
        "     architecture's accuracy varies strongly (and can degrade sharply)\n"
        "     with the instantaneous Sun-magnetic separation angle."
    )
    p(
        "\n  Q: What does this analysis NOT yet say?\n"
        "  A: Nothing here addresses mass, power, cost, eclipse/Sun-exclusion\n"
        "     availability duty cycle, blinding/stray light, update-rate effects\n"
        "     on a real closed-loop control bandwidth, or full estimator (EKF)\n"
        "     performance with gyro propagation and measurement history. Those are\n"
        "     explicitly DEFERRED to a later milestone."
    )
    p()
    p(line("*"))
    p("* NO FINAL SENSOR-SUITE RECOMMENDATION IS MADE IN THIS MILESTONE. *")
    p("* Final architecture selection is explicitly deferred to a later     *")
    p("* milestone that incorporates availability, mass/power/cost, and     *")
    p("* full estimator performance.                                       *")
    p(line("*"))

    # -----------------------------------------------------------------
    # Write text report
    # -----------------------------------------------------------------
    report_path = RESULTS_DIR / "m1_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\n[report written to {report_path}]")

    # ===================================================================
    # Figures
    # ===================================================================

    # Figure 1: information ellipsoid / principal-axis comparison at 90 deg
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"}, figsize=(11, 5.2))
    _plot_error_ellipsoid(axes[0], star_report.P, "Star tracker\n(isotropic model)", STAR_COLOR)
    _plot_error_ellipsoid(axes[1], sm_report_90.P, "Sun + magnetometer\n(90 deg separation)", SUNMAG_COLOR)
    fig.suptitle("Instantaneous Attitude-Error Covariance-Proxy Ellipsoids", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_information_ellipsoids.png")
    plt.close(fig)

    # Figure 2: condition number vs separation angle
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(
        df["separation_angle_deg"], df["sunmag_condition_number_J"],
        color=SUNMAG_COLOR, lw=2, label="Sun + magnetometer",
    )
    ax.axhline(
        star_report.condition_number_J, color=STAR_COLOR, lw=2, ls="--",
        label="Star tracker (isotropic reference)",
    )
    ax.set_xlabel("Sun–magnetic separation angle [deg]")
    ax.set_ylabel("Information-matrix condition number [–]")
    ax.set_title("Attitude-Information Conditioning vs. Sun–Magnetic Separation Angle")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_condition_number_vs_angle.png")
    plt.close(fig)

    # Figure 3: worst-axis error proxy vs separation angle
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        df["separation_angle_deg"], df["sunmag_worst_axis_error_deg"],
        color=SUNMAG_COLOR, lw=2, label="Sun + magnetometer",
    )
    ax.axhline(
        star_report.worst_axis_error_deg, color=STAR_COLOR, lw=2, ls="--",
        label="Star tracker (isotropic reference)",
    )
    ax.set_xlabel("Sun–magnetic separation angle [deg]")
    ax.set_ylabel("Worst-axis 1σ attitude-error proxy [deg]")
    ax.set_title("Worst-Axis Attitude-Error Proxy vs. Sun–Magnetic Separation Angle")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_worst_axis_error_vs_angle.png")
    plt.close(fig)

    # Figure 4: rank / min singular value vs separation angle (include exact 0/180)
    angles_full = np.concatenate([[0.0], angles_deg, [180.0]])
    df_full = run_separation_sweep(angles_full, SIGMA_SUN_RAD, SIGMA_MAG_RAD, SIGMA_STAR_RAD)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df_full["separation_angle_deg"], df_full["min_singular_value"], color="#2ca02c", lw=2)
    ax1.set_xlabel("Sun–magnetic separation angle [deg]")
    ax1.set_ylabel("Minimum singular value of measurement Jacobian [–]", color="#2ca02c")
    ax1.tick_params(axis="y", labelcolor="#2ca02c")
    ax2 = ax1.twinx()
    ax2.plot(df_full["separation_angle_deg"], df_full["rank"], color="black", lw=1.5, ls=":", marker="o", markersize=3)
    ax2.set_ylabel("Geometry rank [–]")
    ax2.set_yticks([0, 1, 2, 3])
    ax1.set_title("Measurement-Geometry Singularity Near Collinearity")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_rank_and_singular_value_vs_angle.png")
    plt.close(fig)

    # Figure 5: 3D vector geometry, healthy vs near-degenerate
    fig = plt.figure(figsize=(11, 5.2))
    ax_healthy = fig.add_subplot(1, 2, 1, projection="3d")
    ax_degen = fig.add_subplot(1, 2, 2, projection="3d")
    _plot_vector_geometry(ax_healthy, *sun_mag_body_vectors(90.0), "Healthy geometry (90 deg separation)")
    _plot_vector_geometry(ax_degen, *sun_mag_body_vectors(5.0), "Near-degenerate geometry (5 deg separation)")
    fig.suptitle("Sun / Magnetic-Field Body-Frame Vector Geometry", fontsize=14)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig5_vector_geometry_3d.png")
    plt.close(fig)

    print(f"[figures written to {RESULTS_DIR}]")


def _plot_error_ellipsoid(ax, P, title: str, color: str) -> None:
    """Plot a 3D attitude-error covariance-proxy ellipsoid (in degrees) on ax."""
    ax.set_title(title, fontsize=11)
    if P is None:
        ax.text2D(0.5, 0.5, "SINGULAR\n(no covariance proxy)", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        return

    P_deg2 = P * np.degrees(1.0) ** 2  # rad^2 -> deg^2 for readability
    eigvals, eigvecs = np.linalg.eigh(P_deg2)
    radii = np.sqrt(np.clip(eigvals, 0, None))

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x, y, z], axis=-1) @ eigvecs.T
    ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], color=color, alpha=0.55, linewidth=0)
    max_r = max(radii.max(), 1e-6) * 1.3
    ax.set_xlim(-max_r, max_r)
    ax.set_ylim(-max_r, max_r)
    ax.set_zlim(-max_r, max_r)
    ax.set_xlabel("Axis 1 [deg]")
    ax.set_ylabel("Axis 2 [deg]")
    ax.set_zlabel("Axis 3 [deg]")


def _plot_vector_geometry(ax, v1: np.ndarray, v2: np.ndarray, title: str) -> None:
    ax.quiver(0, 0, 0, *v1, color="#ff7f0e", length=1.0, normalize=True, label="Sun direction (body)")
    ax.quiver(0, 0, 0, *v2, color="#17becf", length=1.0, normalize=True, label="Magnetic-field direction (body)")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("Body X")
    ax.set_ylabel("Body Y")
    ax.set_zlabel("Body Z")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8)


if __name__ == "__main__":
    main()
