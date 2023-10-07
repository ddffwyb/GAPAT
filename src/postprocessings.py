import bm3d
import cv2
import numpy as np
import scipy.ndimage.filters as filters


def bm3d_denoise_2d(img, noise_std):
    """
    Applies BM3D denoising algorithm to a 2D image.

    Args:
        img (numpy.ndarray): The input image to be denoised.
        noise_std (float): The standard deviation of the noise.

    Returns:
        numpy.ndarray: The denoised image.
    """
    # Normalize the image to 0-1
    img = img.astype(np.float32)
    img_normalized = img / np.max(img)
    # Noise reduction using BM3D algorithm
    img_normalized_denoised = bm3d.bm3d(img_normalized, noise_std)
    # Scale the image to full range 0-255
    img_denoised = np.uint8(
        img_normalized_denoised * 255 / np.max(img_normalized_denoised)
    )
    # Return the denoised image
    return img_denoised


def frangi_filter_2d(
    img, sigmas=np.arange(10), alpha=0.5, beta=2, gamma=15, black_ridges=True
):
    """
    Applies the Frangi filter to a 2D image.

    Args:
        img (ndarray): The input image.
        sigmas (ndarray): The scales at which to calculate the Hessian matrix.
        alpha (float): Frangi correction constant.
        beta (float): Frangi correction constant.
        gamma (float): Frangi correction constant.
        black_ridges (bool): Whether to enhance black ridges or white ridges.

    Returns:
        ndarray: The filtered image.
    """
    # Calculate the Hessian matrix at each scale
    Dxx_list = []
    Dxy_list = []
    Dyy_list = []
    for sigma in sigmas:
        Dxx, Dxy, Dyy = (
            filters.gaussian_filter(img, sigma, order=(2, 0), mode="constant", cval=0),
            filters.gaussian_filter(img, sigma, order=(1, 1), mode="constant", cval=0),
            filters.gaussian_filter(img, sigma, order=(0, 2), mode="constant", cval=0),
        )
        Dxx_list.append(Dxx)
        Dxy_list.append(Dxy)
        Dyy_list.append(Dyy)

    # Calculate the Frangi response for each scale
    response_list = []
    for Dxx, Dxy, Dyy in zip(Dxx_list, Dxy_list, Dyy_list):
        eigenvalues, _ = np.linalg.eig(np.array([[Dxx, Dxy], [Dxy, Dyy]]))
        lambda1 = np.where(
            np.abs(eigenvalues[0, :, :]) <= np.abs(eigenvalues[1, :, :]),
            eigenvalues[0, :, :],
            eigenvalues[1, :, :],
        )
        lambda2 = np.where(
            np.abs(eigenvalues[0, :, :]) > np.abs(eigenvalues[1, :, :]),
            eigenvalues[0, :, :],
            eigenvalues[1, :, :],
        )
        Rb = (lambda2 / lambda1) ** 2
        S = np.sqrt(lambda1**2 + lambda2**2)
        Ra = S**2 / (S ** (2 * alpha))

        if black_ridges:
            response = (1 - np.exp(-Rb / (2 * beta**2))) * np.exp(
                -Ra / (2 * gamma**2)
            )
        else:
            response = (1 - np.exp(-Rb / (2 * beta**2))) * (
                1 - np.exp(-Ra / (2 * gamma**2))
            )

        response_list.append(response)

    # Combine Frangi responses at various scales
    combined_response = np.max(response_list, axis=0)

    return combined_response


def generate_3d_video(data, fps=30.0, output_path="output.mp4"):
    """
    Generate a 3D video from a 3D numpy array.

    Args:
        data (numpy.ndarray): A 3D numpy array of shape (depth, height, width).
        fps (float): Frames per second of the output video. Default is 30.0.
        output_path (str): Output file path of the video. Default is "output.mp4".

    Returns:
        None
    """

    def get_mip_images(data, rotation_matrix):
        """
        Get the maximum intensity projection (MIP) images from a 3D numpy array.

        Args:
            data (numpy.ndarray): A 3D numpy array of shape (depth, height, width).
            rotation_matrix (numpy.ndarray): A 2x3 numpy array representing the rotation matrix.

        Returns:
            numpy.ndarray: A 2D numpy array of shape (depth, width) representing the MIP images.
        """
        data_rotated = np.array(
            [
                cv2.warpAffine(slice, rotation_matrix, slice.shape[::-1])
                for slice in data
            ]
        )
        mip = np.max(data_rotated, axis=1)
        return mip

    depth, height, width = data.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        output_path, fourcc, fps, (width, depth), isColor=False
    )

    for angle in range(360):
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        # Get the MIP images
        mip_image = np.flip(get_mip_images(data, rotation_matrix), axis=0)
        # Normalize the image to 0-255
        mip_image = np.uint8(mip_image * 255 / np.max(mip_image))
        # Write the image to video
        video_writer.write(mip_image)

    # Release the video writer
    video_writer.release()


# -------------------------------------------------------------------------------------------------
# Further post-processing or analysis can be performed based on your specific requirements.
# -------------------------------------------------------------------------------------------------
