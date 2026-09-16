"""
MECH5720 Lab 2 -- Multiplexing for Relighting.

Relight a scene from a stack of images captured under multiplexed
illumination, and measure the signal-to-noise advantage that
multiplexing buys you over illuminating one site at a time.

Run this by stepping through it in your debugger rather than executing
it end to end. Set a breakpoint at the first section and step forward,
watching each figure as it appears. Every step you complete should
produce a visible change in the figures.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard

import imaging as im
import utils

# ======================================================================
# Configuration
# ======================================================================
# Everything you need to change to switch between the provided data and
# your own capture lives in this block.

# Pick which scene to use
IS_EXAMPLE = False
"""Using provided scene"""
IS_BONUS = True
"""Using bonus scene"""
IS_INPUT_RAW = False
"""Using raw dngs or pre-debayered pngs"""

EXAMPLE_PATH = "./Lab2/pics"
"""Path to example pngs."""
BONUS_PATH = "./Lab2/data/bonus"
"""Path to bonus pngs."""
OWN_DEBAYERED_PATH = "./Lab2/data/debayered"
"""Path to own scene pngs."""
OWN_RAW_PATH = "./Lab2/data/debayered"
"""Path to own scene dngs. Only used for own scene."""

DATA_PATH = EXAMPLE_PATH if IS_EXAMPLE else (
            BONUS_PATH if IS_BONUS else(
            OWN_RAW_PATH if IS_INPUT_RAW else OWN_DEBAYERED_PATH
))
"""Directory holding the image files.

If IS_EXAMPLE is True, it should use the example pngs.
Otherwise, if IS_INPUT_RAW is False, it should use our own debayered pngs.
Else, use the raw dngs outputed by the camera and only debayer relevant images.
"""

BLACK_LEVEL = 16
"""Sensor black level"""

N_SITES = 3 if IS_BONUS else 15
"""Illumination sites == multiplexed frames"""
N_AMBIENT = 1 if IS_BONUS else 16
"""Frames with all sites off"""
N_ALLON = 1 if IS_BONUS else 16
"""Frames with all sites on"""
N_FIRSTON = 1 if IS_BONUS else 16
"""Frames with only site 1 on"""

GAMMA_DISPLAY = 2.0
"""Display only; never applied to the data"""
GAMMA_ERROR = 3.3
"""Stronger gamma, to see small error values"""

plt.ion()                       # figures update as the script runs


# ======================================================================
# 1. The multiplexing matrix
# ======================================================================
# An "S-matrix" built from a Hadamard matrix, following Schechner et al.
# 2007, "Multiplexing for Optimal Lighting", Appendix A. Row i gives the
# illumination pattern used for multiplexed frame i: a 1 means that site
# was on. Dropping the first row and column of the Hadamard matrix
# removes the all-on pattern, which carries no differential information.
# H must match the matrix used at capture time.

H = hadamard(N_SITES + 1)
H = H[1:, 1:]
H = (1 - H) / 2

fig = im.show(H[:, :, None], "Multiplexing matrix H", gamma=1.0, fignum=1)
im.save_figure(fig, "01_multiplexing_matrix")

print(f"Each multiplexed frame has {(H[0].sum(), N_SITES)} of {(H[0].sum(), N_SITES)} sites on.")


# ======================================================================
# 2. Load and inspect the multiplexed frames
# ======================================================================

mux: np.ndarray
if IS_INPUT_RAW:
    mux = utils.load_raw_stack(DATA_PATH, "Multiplexed", N_SITES, BLACK_LEVEL)
else:
    mux = im.load_stack(DATA_PATH, "Multiplexed", N_SITES, BLACK_LEVEL / utils.MAX_PIXEL)
print("Multiplexed stack:", mux.shape)

fig = im.show_stack(mux, "Multiplexed input frames", GAMMA_DISPLAY, fignum=2)
im.save_figure(fig, "02_multiplexed_input")


# ======================================================================
# 3. Estimate the ambient illumination
# ======================================================================
# The ambient frames were captured with every illumination site off, so
# they record whatever light reaches the scene that you do not control.
# A single ambient frame is as noisy as any other frame; averaging the
# stack gives a much lower-noise estimate.

ambient_stack: np.ndarray
if IS_INPUT_RAW:
    ambient_stack = utils.load_raw_stack(DATA_PATH, "Ambient", N_AMBIENT, BLACK_LEVEL)
else:
    ambient_stack = im.load_stack(
        DATA_PATH, "Ambient", N_AMBIENT, BLACK_LEVEL / utils.MAX_PIXEL
)


# NOTE QUESTION
# 1: Average the ambient frames to get a low-noise ambient estimate.
#    Result should have shape (H, W, C), i.e. one image, not a stack.
ambient = np.mean(ambient_stack, axis=0)

fig = im.show(
    ambient,
    f"Ambient, averaged over {N_AMBIENT} frames",
    GAMMA_DISPLAY,
    fignum=4
)
im.save_figure(fig, "03_ambient_averaged")


# ======================================================================
# 4. Remove the ambient contribution
# ======================================================================
# The multiplexing model describes only the light you control. Ambient
# light is a constant offset present in every frame, so subtract it.
#
# Note what we do NOT do here. It is tempting to clamp the result at
# zero, on the grounds that there is no such thing as negative light.
# Do not. Where the controlled illumination is weaker than the noise,
# roughly half the pixels legitimately read below ambient, and clamping
# them discards the negative half of a symmetric noise distribution.
# That leaves a positive bias in every frame, which propagates into
# every measurement you make later. Negative values here are noise, not
# an error, and they must be allowed to cancel. Clamping is for display
# only, and imaging.show() already handles it.

# NOTE QUESTION
# 2: subtract the ambient estimate from the multiplexed stack.
mux = mux - ambient

fig = im.show_stack(mux, "Multiplexed input, ambient removed",
                    GAMMA_DISPLAY, fignum=5)
im.save_figure(fig, "04_multiplexed_ambient_removed")


# ======================================================================
# 5. Demultiplex
# ======================================================================
# Each multiplexed frame is a sum of the single-site images selected by
# one row of H:   mux = H @ single_site.  Recovering the single-site
# images is therefore a linear solve, applied identically to every pixel
# and every colour channel. Flatten so that each row is one frame.

original_shape = mux.shape
mux_flat = mux.reshape(N_SITES, -1)
print("Flattened for solving:", mux_flat.shape)

# NOTE QUESTION
# 3: solve for the single-site images.
#    Use np.linalg.solve rath1er than forming H^-1 explicitly: it is
#    better conditioned and faster. The answer is the same either
#    way, so this is a numerical choice, not a mathematical one.
demux_flat = np.linalg.solve(H, mux_flat)

demux = demux_flat.reshape(original_shape)

fig = im.show_stack(im.side_by_side(mux, demux),
                    "Multiplexed (left) and demultiplexed (right)",
                    GAMMA_DISPLAY, fignum=6)
im.save_figure(fig, "05_mux_vs_demux")


# ======================================================================
# 6. Relight the scene
# ======================================================================
# You now hold the scene's appearance under each illumination site
# separately. Any illumination pattern you like is a weighted sum of
# those images. Add the ambient back in to get a physically plausible
# image.

fig = im.show_stack(demux + ambient,
                    "Single site illuminated, with ambient",
                    GAMMA_DISPLAY, fignum=7)

# NOTE QUESTION
# 4: synthesise the scene as it would appear with all sites on.

all_on_weights = np.ones(demux.shape[0])

allon_est_demux: np.ndarray

if IS_INPUT_RAW:
    allon_est_demux = ambient + np.sum(demux * all_on_weights[:, None, None], axis=0)
else:
    allon_est_demux = ambient + np.sum(demux * all_on_weights[:, None, None, None], axis=0)

fig = im.show(allon_est_demux, "Synthesised: all sites on", GAMMA_DISPLAY, fignum=8)
im.save_figure(fig, "06_synth_allon")

# NOTE QUESTION
# 5: Synthesise the scene lit by a checkerboard pattern.
#    Choose which sites to switch on. Which indices form a
#    checkerboard depends on how the sites were laid out on the
#    display at capture time, so work out the layout for the
#    dataset you are using before you pick them.

# NOTE: Change to be correct for new data
checkerboard_weights: np.ndarray

if IS_BONUS:
    checkerboard_weights = np.array([0, 1, 0])
else:
    checkerboard_weights = np.array([
        1, 0, 1, 0,
        0, 1, 0, 1,
        1, 0, 1, 0,
        0, 1, 0
    ])

relit: np.ndarray

if IS_INPUT_RAW:
    relit = ambient + np.sum(demux * checkerboard_weights[:, None, None], axis=0)
    relit = utils.debayer_raw(relit)
    relit = utils.normalise_float_image(relit)
    relit = utils.rotate_image(relit)
else:
    relit = ambient + np.sum(demux * checkerboard_weights[:, None, None, None], axis=0)

fig = im.show(relit, "Synthesised: checkerboard illumination", GAMMA_DISPLAY, fignum=9)
im.save_figure(fig, "07_synth_checkerboard")

# --- Free experimentation -------------------------------------------
# Try: several sites at once to accentuate shadows; different weights
# per colour channel; scaling the ambient down to make the relighting
# more dramatic.
# Also doubles as bonus

if not IS_INPUT_RAW and not IS_BONUS:

    AMBIENT_FACTOR = 0.02
    creative = np.zeros_like(demux[0])
    STAMP_STRENGTH = 1
    GRADIENT_STRENGTH = 0.75

    stamp = [
        [0, 0, 0,   1],
        [0, 0, 0.5, 0],
        [1, 0, 0.5, 0],
        [0, 1, 0,   0],
    ]

    for col_i in range(4):
        for row_i in range(4):
            if row_i == 3 and col_i == 3:
                continue

            red_weight =   GRADIENT_STRENGTH * (1.0 - row_i / 3.0)
            green_weight = STAMP_STRENGTH * stamp[row_i][col_i]
            blue_weight =  GRADIENT_STRENGTH * (1.0 - col_i / 3.0)

            creative[:, :, 0] += demux[row_i + col_i*4, :, :, 0] * red_weight
            creative[:, :, 1] += demux[row_i + col_i*4, :, :, 1] * green_weight
            creative[:, :, 2] += demux[row_i + col_i*4, :, :, 2] * blue_weight


    creative += ambient * AMBIENT_FACTOR

    fig = im.show(creative, "Creative relighting", GAMMA_DISPLAY, fignum=10)
    im.save_figure(fig, "08_creative")

elif IS_BONUS:

    AMBIENT_FACTOR = 1.0
    
    stamps = np.array([
        [[0.5, 0.5, 0.1], [0.6, 0.1, 0.9], [0.1, 0.6, 0.6]],
        [[0.8, 0.2, 0.2], [0.1, 0.2, 1.0], [0.1, 1.0, 0.2]],
        [[0.4, 0.2, 0.3], [0.9, 0.2, 0.1], [0.2, 0.1, 0.8]],
        [[0.5, 0.4, 0.6], [0.7, 0.8, 0.7], [0.9, 0.5, 0.7]]
    ])
    STAMP_STRENGTH = 1
    GRADIENT_STRENGTH = 0.75
    fig_num = 10
    for stamp in stamps:
        creative = np.zeros_like(demux[0])

        for i in range(3):
            red_weight =   STAMP_STRENGTH * stamp[i][0]
            green_weight = STAMP_STRENGTH * stamp[i][1]
            blue_weight =  STAMP_STRENGTH * stamp[i][2]

            creative[:, :, 0] += demux[i, :, :, 0] * red_weight
            creative[:, :, 1] += demux[i, :, :, 1] * green_weight
            creative[:, :, 2] += demux[i, :, :, 2] * blue_weight


        creative += ambient * AMBIENT_FACTOR

        fig = im.show(creative, "Creative relighting", GAMMA_DISPLAY, fignum=fig_num)
        im.save_figure(fig, f"08_{fig_num}_creative")
        fig_num += 1

# ======================================================================
# 7. Qualitative comparison against single-site capture
# ======================================================================
# The impulse frames were captured with one site on at a time: the
# direct alternative to multiplexing. They cost the same number of
# exposures, so this is a fair comparison.

impulse: np.ndarray
if IS_INPUT_RAW:
    impulse = utils.load_raw_stack(DATA_PATH, "Impulse", N_SITES, BLACK_LEVEL)
else:
    impulse = im.load_stack(DATA_PATH, "Impulse", N_SITES, BLACK_LEVEL / utils.MAX_PIXEL)

fig = im.show_stack(im.side_by_side(impulse, demux + ambient),
                    "Impulse (left) vs demultiplexed (right)",
                    GAMMA_DISPLAY, fignum=100)
im.save_figure(fig, "09_impulse_vs_demux")
# Zoom in to better see the noise levels.

# The all-on frames, averaged, are the gold standard: a direct
# measurement of what both methods are trying to predict.
allon_gold_standard: np.ndarray
if IS_INPUT_RAW:
    allon_gold_standard = utils.load_raw_stack(
        DATA_PATH, "AllOn", N_ALLON, BLACK_LEVEL
    ).mean(axis=0)
else:
    allon_gold_standard = im.load_stack(
        DATA_PATH, "AllOn", N_ALLON, BLACK_LEVEL / utils.MAX_PIXEL
    ).mean(axis=0)

# NOTE QUESTION
# 6: Estimate the all-on image from the impulse frames.
#    Think carefully about the ambient.

allon_est_impulse = ambient * (N_SITES - 1) + np.sum(impulse, axis=0)

fig = im.show_stack(
    im.side_by_side(allon_gold_standard, allon_est_impulse, allon_est_demux)[None],
    "All sites on:  gold standard  /  "
    "impulse estimate  /  demultiplexed estimate",
    GAMMA_DISPLAY, fignum=101)
im.save_figure(fig, "10_allon_three_way")


# ======================================================================
# 8. Quantitative evaluation
# ======================================================================

error_demux = allon_est_demux - allon_gold_standard
error_impulse = allon_est_impulse - allon_gold_standard

if IS_INPUT_RAW:
    error_demux = utils.debayer_raw(error_demux)
    error_demux = utils.normalise_float_image(error_demux)
    error_demux = utils.rotate_image(error_demux)
    error_impulse = utils.debayer_raw(error_impulse)
    error_impulse = utils.normalise_float_image(error_impulse)
    error_impulse = utils.rotate_image(error_impulse)

fig = im.show_stack(
    im.side_by_side(error_impulse ** 2, error_demux ** 2)[None],
    "Squared error vs the all-on gold standard:  "
    "impulse (left)  /  demultiplexed (right)",
    GAMMA_ERROR, fignum=102)
im.save_figure(fig, "11_error_images")


# NOTE
# 7: Find the mean squared error of each estimate against the gold
#    standard, then the peak signal-to-noise ratio of each, and the
#    advantage multiplexing gives you. Peak value here is 1.0, since
#    the images are on [0, 1]. Find PSNR and PSNR advantage in dB.

demux_noise = allon_gold_standard - allon_est_demux
impulse_noise = allon_gold_standard - allon_est_impulse

if IS_INPUT_RAW:
    demux_noise /= utils.MAX_PIXEL
    impulse_noise /= utils.MAX_PIXEL

mse_demux = np.mean((demux_noise) ** 2)
mse_impulse = np.mean((impulse_noise) ** 2)

psnr_demux_db = 10 * np.log10(1.0 / np.std(demux_noise))
psnr_impulse_db = 10 * np.log10(1.0 / np.std(impulse_noise))
psnr_advantage_db = psnr_demux_db - psnr_impulse_db


# ======================================================================
# 9. The same comparison for a single illuminant
# ======================================================================
# The advantage above is for the all-on image, which is the easiest case
# for multiplexing. Repeat the measurement for one illumination site on
# its own, the hardest case, using the FirstOn frames as the gold
# standard: those were captured with site 1 on and nothing else.

first_gold_standard: np.ndarray
if IS_INPUT_RAW:
    first_gold_standard = utils.load_raw_stack(
        DATA_PATH, "FirstOn", N_FIRSTON, BLACK_LEVEL
    ).mean(axis=0)
else:
    first_gold_standard = im.load_stack(
        DATA_PATH, "FirstOn", N_FIRSTON, BLACK_LEVEL / utils.MAX_PIXEL
    ).mean(axis=0)


# NOTE
# 8: Estimate the site-1 image by each method. The impulse method
#    measured it directly, in one frame. For the demultiplexed
#    estimate don't forget the impact of ambient light.
first_est_impulse = impulse[0]
first_est_demux = ambient + demux[0]

fig = im.show_stack(
    im.side_by_side(first_gold_standard, first_est_impulse,
                    first_est_demux)[None],
    "Site 1 only:  gold standard  /  "
    "impulse estimate  /  demultiplexed estimate",
    GAMMA_DISPLAY, fignum=103)
im.save_figure(fig, "12_single_site_three_way")

demux_first_noise = first_gold_standard - first_est_demux
impulse_first_noise = first_gold_standard - first_est_impulse

if IS_INPUT_RAW:
    demux_first_noise /= utils.MAX_PIXEL
    impulse_first_noise /= utils.MAX_PIXEL

mse_first_demux = np.mean((demux_first_noise) ** 2)
mse_first_impulse = np.mean((impulse_first_noise) ** 2)

psnr_first_demux_db = 10 * np.log10(1.0 / np.std(demux_first_noise))
psnr_first_impulse_db = 10 * np.log10(1.0 / np.std(impulse_first_noise))
psnr_first_advantage_db = psnr_first_demux_db - psnr_first_impulse_db


# ======================================================================
# 10. Report these numbers
# ======================================================================
def row(label, value, unit=""):
    """durr"""
    print(f"  {label:<32} {value} {unit}".rstrip())

ambient_mean = ambient.mean()
ambient_std = ambient_stack.std(axis=0).mean()

if IS_INPUT_RAW:
    ambient_mean /= utils.MAX_PIXEL
    ambient_std /= utils.MAX_PIXEL

print("")
print("=" * 58)
row("ambient mean level", f"{ambient_mean:.5f}")
row("ambient std dev", f"{ambient_std:.5f}")
print("-" * 58)
row("ALL-ON IMAGE", "")
row("  MSE, impulse estimate", f"{mse_impulse:.3e}")
row("  MSE, demultiplexed estimate", f"{mse_demux:.3e}")
row("  PSNR, impulse estimate", f"{psnr_impulse_db:.3f}", "dB")
row("  PSNR, demultiplexed estimate", f"{psnr_demux_db:.3f}", "dB")
row("  PSNR ADVANTAGE", f"{psnr_advantage_db:.3f}", "dB")
print("-" * 58)
row("SINGLE-SITE IMAGE (site 1)", "")
row("  PSNR, impulse estimate", f"{psnr_first_impulse_db:.3f}", "dB")
row("  PSNR, demultiplexed estimate", f"{psnr_first_demux_db:.3f}", "dB")
row("  PSNR ADVANTAGE", f"{psnr_first_advantage_db:.3f}", "dB")
print("=" * 58)
print("")


# ======================================================================
# Cleanup / close
# ======================================================================

plt.show(block=False)
plt.pause(0.2)          # let every figure draw before we block
try:
    input("press enter to close the figures ")
except EOFError:        # no interactive stdin: fall back to blocking
    plt.show(block=True)
plt.close("all")
