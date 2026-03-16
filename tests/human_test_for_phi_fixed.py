"""Manual checks for fixed-phi beam-size behavior."""

import numpy as np
import laserbeamsize as lbs


def beam_size_checked(
    beam_image: np.ndarray, fixed_phi: float | None = None
) -> tuple[float, float, float, float, float]:
    """Return beam_size results, asserting the minor diameter is present."""
    x_center, y_center, major_diameter, minor_diameter, fitted_phi = lbs.beam_size(beam_image, phi_fixed=fixed_phi)
    if minor_diameter is None:
        raise RuntimeError("beam_size() unexpectedly returned d_minor=None in manual phi_fixed test")
    return x_center, y_center, major_diameter, minor_diameter, fitted_phi


def equivalent_diameter(major_diameter: float, minor_diameter: float) -> float:
    """Return the equivalent circular diameter for a forced 45-degree offset fit."""
    return np.sqrt((major_diameter**2 + minor_diameter**2) / 2)


print("laserbeamsize version is", lbs.__version__)
phi_true = np.radians(45)
h = 600
v = 600
xc: float = 250
yc: float = 250
d_major: float = 150
d_minor: float = 100
phi_true = np.radians(45)
beam = lbs.create_test_image(h, v, xc, yc, d_major, d_minor, phi_true)

print()
print(" x_center  y_center  d_major   d_minor   ɸ_fixed    ɸ_calc")
print("%8.2f  %8.2f  %8.2f  %8.2f            %8.2f° --- truth" % (250, 250, 150, 100, np.degrees(phi_true)))
print("-------------------------------------------------------------------------------------")

xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f       None %8.2f° --- default fit for angle "
    % (xc, yc, d_major, d_minor, np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(0)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- should match"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)
equiv_d = equivalent_diameter(d_major, d_minor)

phi_fixed = phi_true + np.radians(90)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor≈%.1f"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc), equiv_d)
)

phi_fixed = phi_true + np.radians(-90)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor≈%.1f"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc), equiv_d)
)

phi_fixed = phi_true + np.radians(45)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor≈%.1f"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc), equiv_d)
)

phi_fixed = phi_true + np.radians(-45)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor≈%.1f"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc), equiv_d)
)

h = 600
v = 600
xc = 250
yc = 350
d_major = 150
d_minor = 100
phi_true = np.radians(-30)
beam = lbs.create_test_image(h, v, xc, yc, d_major, d_minor, phi_true)

print()
print(" x_center  y_center  d_major   d_minor   ɸ_fixed    ɸ_calc")
print("%8.2f  %8.2f  %8.2f  %8.2f            %8.2f° --- truth" % (250, 350, 150, 100, np.degrees(phi_true)))
print("-------------------------------------------------------------------------------------")

xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f       None %8.2f° --- default fit for angle "
    % (xc, yc, d_major, d_minor, np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(0)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- should match"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)
equiv_d = equivalent_diameter(d_major, d_minor)

phi_fixed = phi_true + np.radians(90)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- major/minor swapped"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(-90)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- major/minor swapped"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(45)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor≈%.1f"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc), equiv_d)
)

phi_fixed = phi_true + np.radians(-45)
xc, yc, d_major, d_minor, phi_calc = beam_size_checked(beam, fixed_phi=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor≈%.1f"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc), equiv_d)
)
