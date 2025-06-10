import numpy as np
import laserbeamsize as lbs

print("laserbeamsize version is", lbs.__version__)
phi_true = np.radians(45)
h = 600
v = 600
xc = 250
yc = 250
d_major = 150
d_minor = 100
phi_true = np.radians(45)
beam = lbs.create_test_image(h, v, xc, yc, d_major, d_minor, phi_true)

print()
print(" x_center  y_center  d_major   d_minor   ɸ_fixed    ɸ_calc")
print("%8.2f  %8.2f  %8.2f  %8.2f            %8.2f° --- truth" % (250, 350, 150, 100, np.degrees(phi_true)))
print("-------------------------------------------------------------------------------------")

xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f       None %8.2f° --- default fit for angle "
    % (xc, yc, d_major, d_minor, np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(0)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- should match"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(90)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- major/minor swapped"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(-90)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- major/minor swapped"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(45)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor=127.5"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(-45)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor=127.5"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
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

xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f       None %8.2f° --- default fit for angle "
    % (xc, yc, d_major, d_minor, np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(0)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- should match"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(90)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- major/minor swapped"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(-90)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- major/minor swapped"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(45)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor=127.5"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)

phi_fixed = phi_true + np.radians(-45)
xc, yc, d_major, d_minor, phi_calc = lbs.beam_size(beam, phi_fixed=phi_fixed)
print(
    "%8.2f  %8.2f  %8.2f  %8.2f  %8.2f° %8.2f° --- d_major=d_minor=127.5"
    % (xc, yc, d_major, d_minor, np.degrees(phi_fixed), np.degrees(phi_calc))
)
