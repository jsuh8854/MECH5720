"""
ADMM reconstruction for DiffuserCam lensless imaging.

The ADMM update equations are unchanged from the supplied solver.
The code has been refactored for readability and explicit parameter passing.
""" # pylint: disable=invalid-name

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from PIL import Image
import yaml


def downsample(image, factor):
    """Downsample an image by repeated 2x2 averaging."""
    num_downsamples = int(-np.log2(factor))

    for _ in range(num_downsamples):
        image = 0.25 * (
            image[::2, ::2, ...]
            + image[1::2, ::2, ...]
            + image[::2, 1::2, ...]
            + image[1::2, 1::2, ...]
        )

    return image


def load_data(psf_filename, image_filename, downsample_factor, show_image=True):
    """Load, background-correct, downsample, and normalize the data."""

    psf = np.asarray(Image.open(psf_filename), dtype=np.float32)
    data = np.asarray(Image.open(image_filename), dtype=np.float32)

    # Subtract the camera background using a dark region of the PSF.
    background = np.mean(psf[5:15, 5:15])
    psf -= background
    data -= background

    # Downsample for a more manageable reconstruction size.
    psf = downsample(psf, downsample_factor)
    data = downsample(data, downsample_factor)

    # Normalize the total power of the PSF and measurement.
    psf /= np.linalg.norm(psf.ravel())
    data /= np.linalg.norm(data.ravel())

    if show_image:
        fig, axes = plt.subplots(1, 2)

        axes[0].imshow(psf, cmap="gray")
        axes[0].set_title("PSF")
        axes[0].axis("off")

        axes[1].imshow(data, cmap="gray")
        axes[1].set_title("Raw data")
        axes[1].axis("off")

        plt.show()
        plt.close(fig)

    return psf, data


def soft_threshold(x, threshold):
    """Apply element-wise soft thresholding."""
    return np.sign(x) * np.maximum(0, np.abs(x) - threshold)


def gradient(image):
    """Calculate horizontal and vertical image gradients."""
    vertical = np.roll(image, 1, axis=0) - image
    horizontal = np.roll(image, 1, axis=1) - image

    return np.stack((vertical, horizontal), axis=2)


def gradient_transpose(gradient_image):
    """Apply the transpose of the gradient operator."""
    vertical = np.roll(gradient_image[..., 0], -1, axis=0)
    vertical -= gradient_image[..., 0]

    horizontal = np.roll(gradient_image[..., 1], -1, axis=1)
    horizontal -= gradient_image[..., 1]

    return vertical + horizontal


def crop_sensor_region(image, full_size, sensor_size):
    """Extract the sensor-sized central region from a full-size image."""

    top = (full_size[0] - sensor_size[0]) // 2
    bottom = (full_size[0] + sensor_size[0]) // 2
    left = (full_size[1] - sensor_size[1]) // 2
    right = (full_size[1] + sensor_size[1]) // 2

    return image[top:bottom, left:right]


def pad_sensor_data(data, full_size, sensor_size):
    """Zero-pad sensor data to the full reconstruction size."""

    vertical_pad = (full_size[0] - sensor_size[0]) // 2
    horizontal_pad = (full_size[1] - sensor_size[1]) // 2

    return np.pad(
        data,
        ((vertical_pad, vertical_pad),
        (horizontal_pad, horizontal_pad)),
        mode="constant",
    )


def forward_model(image, psf_fft):
    """Apply the DiffuserCam forward model."""
    shifted_image = fft.ifftshift(image)
    image_fft = fft.fft2(shifted_image)
    result = fft.ifft2(image_fft * psf_fft)

    return np.real(fft.fftshift(result))


def adjoint_model(sensor_data, psf_fft):
    """Apply the adjoint of the DiffuserCam forward model."""
    shifted_data = fft.ifftshift(sensor_data)
    data_fft = fft.fft2(shifted_data)

    result = fft.ifft2(data_fft * np.conj(psf_fft))

    return np.real(fft.fftshift(result))


def precompute_x_division(mu1, full_size, sensor_size):
    """Precompute the denominator used in the X update."""
    sensor_ones = np.ones(sensor_size)
    padded_ones = pad_sensor_data(sensor_ones, full_size, sensor_size)

    return 1.0 / (padded_ones + mu1)


def precompute_gradient_transpose_gradient(full_size):
    """Precompute the Fourier representation of Psi^T Psi."""
    gradient_matrix = np.zeros(full_size)

    gradient_matrix[0, 0] = 4
    gradient_matrix[0, 1] = -1
    gradient_matrix[1, 0] = -1
    gradient_matrix[0, -1] = -1
    gradient_matrix[-1, 0] = -1

    return fft.fft2(gradient_matrix)


def precompute_r_division(
    psf_fft,
    gradient_transpose_gradient,
    mu1, mu2, mu3,
):
    """Precompute the frequency-domain denominator for the V update."""

    forward_component = mu1 * np.abs(psf_fft) ** 2
    gradient_component = mu2 * np.abs(gradient_transpose_gradient)
    identity_component = mu3

    return 1.0 / (
        forward_component
        + gradient_component
        + identity_component
    )


def update_u(eta, image_estimate, tau, mu2):
    """Update U using the TV soft-thresholding step."""
    return soft_threshold(
        gradient(image_estimate) + eta / mu2,
        tau / mu2,
    )


def update_x(
    xi,
    image_estimate,
    psf_fft,
    sensor_data,
    x_division,
    mu1,
):
    """Update X."""
    forward = forward_model(image_estimate, psf_fft)

    return x_division * (
        xi
        + mu1 * forward
        + pad_sensor_data(
            sensor_data,
            image_estimate.shape,
            sensor_data.shape,
        )
    )

def update_x_two_plane(
    xi,
    image_estimates,
    psf_ffts,
    sensor_data,
    x_division,
    mu1,
):
    """Update X. The blurred copy is the sum of every plane's forward model."""
    forward = sum(
        forward_model(image, psf_fft)
        for image, psf_fft in zip(image_estimates, psf_ffts)
    )

    return x_division * (
        xi
        + mu1 * forward
        + pad_sensor_data(
            sensor_data,
            image_estimates[0].shape,
            sensor_data.shape,
        )
    )


def update_w(rho, image_estimate, mu3):
    """Update W with the non-negativity constraint."""
    return np.maximum(rho / mu3 + image_estimate, 0)


def calculate_r(
    w,
    rho,
    u,
    eta,
    x,
    xi,
    psf_fft,
    mu1,
    mu2,
    mu3,
):
    """Calculate the right-hand side of the V update."""

    return (
        mu3 * w
        - rho
        + gradient_transpose(mu2 * u - eta)
        + adjoint_model(mu1 * x - xi, psf_fft)
    )


def update_v(
    w,
    rho,
    u,
    eta,
    x,
    xi,
    psf_fft,
    r_division,
    mu1,
    mu2,
    mu3,
):
    """Update V in the frequency domain."""

    r = calculate_r(
        w,
        rho,
        u,
        eta,
        x,
        xi,
        psf_fft,
        mu1,
        mu2,
        mu3,
    )

    frequency_result = (
        r_division
        * fft.fft2(fft.ifftshift(r))
    )

    return np.real(
        fft.fftshift(
            fft.ifft2(frequency_result)
        )
    )


def update_xi(xi, image_estimate, psf_fft, x, mu1):
    """Update the dual variable xi."""
    return xi + mu1 * (
        forward_model(image_estimate, psf_fft) - x
    )

def update_xi_two_plane(xi, image_estimates, psf_ffts, x, mu1):
    """Update the dual variable xi using the summed forward model."""
    forward = sum(
        forward_model(image, psf_fft)
        for image, psf_fft in zip(image_estimates, psf_ffts)
    )

    return xi + mu1 * (forward - x)


def update_eta(eta, image_estimate, u, mu2):
    """Update the dual variable eta."""
    return eta + mu2 * (
        gradient(image_estimate) - u
    )


def update_rho(rho, image_estimate, w, mu3):
    """Update the dual variable rho."""
    return rho + mu3 * (
        image_estimate - w
    )


def initialise_variables(psf_fft, full_size):
    """Initialise all ADMM variables."""

    x = np.zeros(full_size)
    u = np.zeros((*full_size, 2))
    v = np.zeros(full_size)
    w = np.zeros(full_size)

    xi = np.zeros_like(forward_model(v, psf_fft))
    eta = np.zeros_like(gradient(v))
    rho = np.zeros_like(w)

    return x, u, v, w, xi, eta, rho

def initialise_variables_two_plane(psf_ffts, full_size):
    """
    Initialise all ADMM variables.
    per-plane: u, v, w, eta, rho
    shared: x, xi
    """

    num_planes = len(psf_ffts)

    x = np.zeros(full_size)
    u = [np.zeros((*full_size, 2)) for _ in range(num_planes)]
    v = [np.zeros(full_size) for _ in range(num_planes)]
    w = [np.zeros(full_size) for _ in range(num_planes)]

    xi = np.zeros(full_size)
    eta = [np.zeros((*full_size, 2)) for _ in range(num_planes)]
    rho = [np.zeros(full_size) for _ in range(num_planes)]

    return x, u, v, w, xi, eta, rho


def precompute_psf_fft(psf, full_size, sensor_size):
    """Calculate the Fourier transform of the padded PSF."""
    padded_psf = pad_sensor_data(psf, full_size, sensor_size)

    return fft.fft2(fft.ifftshift(padded_psf))


def admm_step(
    variables,
    psf_fft,
    data,
    x_division,
    r_division,
    tau,
    mu1,
    mu2,
    mu3,
):
    """Perform one ADMM iteration."""

    x, u, v, w, xi, eta, rho = variables

    u = update_u(eta, v, tau, mu2)

    x = update_x(
        xi,
        v,
        psf_fft,
        data,
        x_division,
        mu1,
    )

    v = update_v(
        w,
        rho,
        u,
        eta,
        x,
        xi,
        psf_fft,
        r_division,
        mu1,
        mu2,
        mu3,
    )

    w = update_w(rho, v, mu3)

    xi = update_xi(xi, v, psf_fft, x, mu1)
    eta = update_eta(eta, v, u, mu2)
    rho = update_rho(rho, v, w, mu3)

    return x, u, v, w, xi, eta, rho

def admm_step_two_plane(
    variables,
    psf_ffts,
    data,
    x_division,
    r_divisions,
    tau,
    mu1,
    mu2,
    mu3,
):
    """Perform one two-plane ADMM iteration."""

    x, u, v, w, xi, eta, rho = variables
    planes = range(len(psf_ffts))

    # Copy so the previous iteration's lists are not mutated
    u, v, w = list(u), list(v), list(w)
    eta, rho = list(eta), list(rho)

    for k in planes:
        u[k] = update_u(eta[k], v[k], tau, mu2)

    x = update_x_two_plane(xi, v, psf_ffts, data, x_division, mu1)

    # Each plane is solved against the data minus what the other planes explained
    for k in planes:
        # Trying to solve for x = h1*v1 + h2*v2

        # Blurs every other plane with its own PSF and subtracts from x
        # What's left is x_k, which is the remainder of x that the current plane is responsible for.
        # E.g. for the near plane, x_near = x - psf_far * v_far
        # where x = blurred image, v = image estimate
        # (why are they even swapped in the code? kinda confusing)
        x_k = x - sum(
            forward_model(v[j], psf_ffts[j]) for j in planes if j != k
        )
        # Then same updates as normal but using the new x_k
        v[k] = update_v(
            w[k], rho[k], u[k], eta[k], x_k, xi,
            psf_ffts[k], r_divisions[k], mu1, mu2, mu3,
        )
        w[k] = update_w(rho[k], v[k], mu3)

    xi = update_xi_two_plane(xi, v, psf_ffts, x, mu1)
    for k in planes:
        eta[k] = update_eta(eta[k], v[k], u[k], mu2)
        rho[k] = update_rho(rho[k], v[k], w[k], mu3)

    return x, u, v, w, xi, eta, rho

def run_admm(
    psf,
    data,
    tau,
    mu1,
    mu2,
    mu3,
    iters,
    display_interval=None,
    snapshots=(5, 20, 100, 300),
):
    """Run the ADMM reconstruction and returns a dict with images at the snapshots.
    """

    snapshots = sorted(set(snapshots)) if snapshots else []
    if snapshots:
        iters = max(iters, snapshots[-1])

    sensor_size = np.array(psf.shape)
    full_size = 2 * sensor_size

    psf_fft = precompute_psf_fft(psf, full_size, sensor_size)
    x_division = precompute_x_division(mu1, full_size, sensor_size)
    gradient_transpose_gradient = precompute_gradient_transpose_gradient(full_size)
    r_division = precompute_r_division(
        psf_fft, gradient_transpose_gradient, mu1, mu2, mu3,
    )
    variables = initialise_variables(psf_fft, full_size)

    results = {}

    for iteration in range(iters):

        variables = admm_step(
            variables,
            psf_fft,
            data,
            x_division,
            r_division,
            tau,
            mu1, mu2, mu3,
        )

        count = iteration + 1  # iterations completed so far

        if count in snapshots or count == iters:
            _, _, v, _, _, _, _ = variables
            results[count] = crop_sensor_region(
                np.maximum(v, 0), full_size, sensor_size,
            ).copy()
            print(f"Stored snapshot at iteration {count}")

        if display_interval and iteration % display_interval == 0:
            _, _, v, _, _, _, _ = variables
            image = np.maximum(
                crop_sensor_region(v, full_size, sensor_size), 0,
            )
            fig = plt.figure()
            plt.imshow(image, cmap="gray")
            plt.title(f"Reconstruction after iteration {iteration}")
            plt.axis("off")
            plt.show()
            plt.close(fig)

    return results

def reconstruct_image(
    psf,
    raw_image,
    config_filename="admm_config.yml",
    snapshots=(5, 20, 100, 300),
    tau=None,
    display=False,
):
    """Reconstruct the image, returning a dictionary with snapshots."""

    with open(config_filename, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return run_admm(
        psf,
        raw_image,
        config["tau"] if tau is None else tau,
        config["mu1"],
        config["mu2"],
        config["mu3"],
        config["iters"],
        config["disp_pic"] if display else None,
        snapshots,
    )

def run_admm_two_plane(
    psf_near,
    psf_far,
    data,
    tau,
    mu1,
    mu2,
    mu3,
    iters,
    display_interval=None,
    snapshots=(5, 20, 100, 300),
):
    """Run the joint two-plane ADMM reconstruction, y = crop(h1*x1 + h2*x2).

    Returns a dict {iteration_count: (near_image, far_image)}.
    """

    snapshots = sorted(set(snapshots)) if snapshots else []
    if snapshots:
        iters = max(iters, snapshots[-1])

    sensor_size = np.array(psf_near.shape)
    full_size = 2 * sensor_size

    psf_ffts = [
        precompute_psf_fft(psf, full_size, sensor_size)
        for psf in (psf_near, psf_far)
    ]
    x_division = precompute_x_division(mu1, full_size, sensor_size)
    gradient_transpose_gradient = precompute_gradient_transpose_gradient(full_size)
    r_divisions = [
        precompute_r_division(psf_fft, gradient_transpose_gradient, mu1, mu2, mu3)
        for psf_fft in psf_ffts
    ]
    variables = initialise_variables_two_plane(psf_ffts, full_size)

    results = {}

    for iteration in range(iters):

        variables = admm_step_two_plane(
            variables,
            psf_ffts,
            data,
            x_division,
            r_divisions,
            tau,
            mu1, mu2, mu3,
        )

        count = iteration + 1  # iterations completed so far

        if count in snapshots or count == iters:
            _, _, v, _, _, _, _ = variables
            results[count] = tuple(
                crop_sensor_region(
                    np.maximum(plane, 0), full_size, sensor_size,
                ).copy()
                for plane in v
            )
            print(f"Stored snapshot at iteration {count}")

        if display_interval and iteration % display_interval == 0:
            _, _, v, _, _, _, _ = variables
            fig, axes = plt.subplots(1, 2)
            for ax, plane, label in zip(axes, v, ("near", "far")):
                ax.imshow(
                    np.maximum(crop_sensor_region(plane, full_size, sensor_size), 0),
                    cmap="gray",
                )
                ax.set_title(f"{label}, iteration {iteration}")
                ax.axis("off")
            plt.show()
            plt.close(fig)

    return results

def reconstruct_two_plane(
    psf_near,
    psf_far,
    raw_image,
    config_filename="admm_config.yml",
    snapshots=(5, 20, 100, 300),
    tau=None,
    display=False,
):
    """Like the other one but two rastnieorstdnoi"""

    with open(config_filename, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return run_admm_two_plane(
        psf_near,
        psf_far,
        raw_image,
        config["tau"] if tau is None else tau,
        config["mu1"],
        config["mu2"],
        config["mu3"],
        config["iters"],
        config["disp_pic"] if display else None,
        snapshots,
    )


def example():
    """Load configuration, run reconstruction, and optionally save."""

    with open("admm_config.yml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    psf, data = load_data(
        config["psfname"],
        config["imgname"],
        config["f"],
        show_image=True,
    )

    tau = config["tau"]
    mu1 = config["mu1"]
    mu2 = config["mu2"]
    mu3 = config["mu3"]
    iters = config["iters"]
    display_interval = config["disp_pic"]

    final_image = run_admm(
        psf,
        data,
        tau,
        mu1, mu2, mu3,
        iters,
        display_interval,
    )

    fig = plt.figure()
    plt.imshow(final_image, cmap="gray")
    plt.title(
        f"Final reconstructed image after {iters} iterations"
    )
    plt.axis("off")
    plt.show()
    plt.close(fig)

    save_image = input("Save final image? (y/n) ")

    if save_image.lower() == "y":
        filename = input("Name of file: ")
        plt.imsave(
            f"{filename}.png",
            final_image,
            cmap="gray",
        )


if __name__ == "__main__":
    example()
