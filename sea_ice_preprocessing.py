"""
Shared preprocessing module for the GVF, YOLOE, and SAM 2 pipelines.

Perspective correction is Liss's (2023) work, taken from
SEA_ICE_Functions.py in the LSSOSC001_MEC4128S repository, so all three
segmentation pipelines in this project run the same correction before
their own model specific logic.

initial_perspective_transform() is interactive. It opens an OpenCV window
with trackbars so you can line up a trapezoid over the near field ice
region on the first frame of a video. Press Esc once it looks right to
lock in the calibration. Run it once per video, not once per frame.

perspective_transform() reapplies that locked in calibration to any later
frame from the same video, no further input needed.

enhance_gray() is gray level homogenization, matching Passerotti et al.
(2026) Section 3.3. It combines an edge preserving bilateral filter, an
anisotropic diffusion filter, and CLAHE contrast enhancement, with each
step individually toggleable so their effects can be tested in isolation
against the same footage.

preprocess_frame() chains perspective_transform() and enhance_gray()
into a single call, taking a raw BGR frame and the locked in calibration
values, and returning a ready to segment enhanced grayscale frame. Added
for use by benchmark_pipeline.py, the Liss contouring benchmark, so it
runs through the same preprocessing as the other three pipelines.
"""

import cv2
import numpy as np


def nothing(x):
    """
    This function is used for the trackbar so that it can keep updating in the while loop
    """
    pass


def initial_perspective_transform(imagefile):
    """
    This function is used to set the perspective transform of the first frame of the video
    :param imagefile: path of the first frame of the video
    :return: transformed image, scaling factor, x shift number of pixels, y shift number of pixels, width of transformed image, height of transformed image
    """

    # Read in image
    image = cv2.imread(imagefile)

    # Sizing in original image
    h, w, c = image.shape

    x = int(w/2)
    y = int(h/5)
    length = 25

    # Sizing off the trapazoid based of the camera angle above open ocean
    # The back horizontal line are what the other lines are based off
    orth_x = 10/5
    orth_y = 214/125

    # Create a black image, a window
    # Output image size
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)

    # Create a trackbar for scale
    cv2.createTrackbar("scale", "image", 1, 50, nothing)
    # Create a trackbar for x position
    cv2.createTrackbar("x shift", "image", 100, 1500, nothing)
    # Create a trackbar for y position
    cv2.createTrackbar("y shift", "image", 50, 1000, nothing)

    while 1:
        # Read in image
        img = cv2.imread(imagefile)

        # Get trackbar positions
        factor = cv2.getTrackbarPos("scale", "image")
        x = cv2.getTrackbarPos("x shift", "image")
        y = cv2.getTrackbarPos("y shift", "image")

        # Pixel values in original image
        T_L = [(x - (length * factor) / 2), y]
        T_R = [(x + (length * factor) / 2), y]
        B_L = [(x - (length * factor * orth_x) / 2), (y + (length * factor * orth_y))]
        B_R = [(x + (length * factor * orth_x) / 2), (y + (length * factor * orth_y))]

        # Create point matrix
        point_matrix = np.float32([T_L, T_R, B_L, B_R])

        # Draw lines to combine the points
        cv2.line(img, (int(T_L[0]), int(T_L[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)
        cv2.line(img, (int(B_L[0]), int(B_L[1])), (int(T_L[0]), int(T_L[1])), (0, 0, 255), 2)
        cv2.line(img, (int(B_R[0]), int(B_R[1])), (int(B_L[0]), int(B_L[1])), (0, 0, 255), 2)
        cv2.line(img, (int(B_R[0]), int(B_R[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)

        k = cv2.waitKey(1) & 0xFF

        if k == 27:
            print("done with K")
            break
        img = cv2.imshow('image', img)


    # Output image size of transformed image
    width, height = int(length * factor * 2), int(length * factor * 12 / 5)

    # Desired points value in output images
    converted_T_L = [0, 0]
    converted_T_R = [width, 0]
    converted_B_L = [0, height]
    converted_B_R = [width, height]

    # Convert points
    converted_points = np.float32([converted_T_L, converted_T_R,
                                   converted_B_L, converted_B_R])

    # perspective transform
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
    image_trans = cv2.warpPerspective(image, perspective_transform, (width, height))
    cv2.imwrite(str("image_trans.jpg"), image_trans)
    cv2.imshow("transformed Image", image_trans)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    return image_trans, factor, x, y, width, height


def perspective_transform(image, factor, x, y):
    """
    This function is used execute the perspective transform on the rest of the frames of the video
    :param image: frame from the video
    :param factor: the scaling factor from the initial_perspective_transform function
    :param x: the x shift number of pixels from the initial_perspective_transform function
    :param y: the y shift number of pixels from the initial_perspective_transform function
    :return: transformed image
    """

    # Sizing in original image
    h, w, c = image.shape
    length = 25

    orth_x = 10 / 5
    orth_y = 214 / 125

    # Pixel values in original image
    T_L = [(x - (length * factor) / 2), y]
    T_R = [(x + (length * factor) / 2), y]
    B_L = [(x - (length * factor * orth_x) / 2), (y + (length * factor * orth_y))]
    B_R = [(x + (length * factor * orth_x) / 2), (y + (length * factor * orth_y))]


    # Output image size of transformed image
    width, height = int(length * factor * 2), int(length * factor * 12 / 5)

    # Create point matrix
    point_matrix = np.float32([T_L, T_R, B_L, B_R])

    # Desired points value in output images
    converted_T_L = [0, 0]
    converted_T_R = [width, 0]
    converted_B_L = [0, height]
    converted_B_R = [width, height]

    # Convert points
    converted_points = np.float32([converted_T_L, converted_T_R,
                                   converted_B_L, converted_B_R])

    # perspective transform
    perspective_transform = cv2.getPerspectiveTransform(point_matrix, converted_points)
    image_trans = cv2.warpPerspective(image, perspective_transform, (width, height))

    # Draw lines to combine the points after the transform has taken place
    cv2.line(image, (int(T_L[0]), int(T_L[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_L[0]), int(B_L[1])), (int(T_L[0]), int(T_L[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_R[0]), int(B_R[1])), (int(B_L[0]), int(B_L[1])), (0, 0, 255), 2)
    cv2.line(image, (int(B_R[0]), int(B_R[1])), (int(T_R[0]), int(T_R[1])), (0, 0, 255), 2)

    return image_trans


def enhance_gray(gray, apply_bilateral=True, apply_diffusion=True, apply_clahe=True,
                  clahe_clip_limit=2.0, clahe_tile_size=8):
    """Gray level homogenization, matching Passerotti et al. (2026)
    Section 3.3. Combines an edge preserving bilateral filter, an
    anisotropic diffusion filter (Perona and Malik, 1990), and CLAHE
    contrast enhancement. Each step is individually toggleable, and
    CLAHE's own parameters are exposed, so the contribution of each
    piece can be tested in isolation against the same footage, rather
    than only ever running or not running as one combined block."""

    result = gray

    if apply_bilateral:
        # Edge preserving smoothing, flattens noise within a floe while
        # keeping strong boundaries between floes intact
        result = cv2.bilateralFilter(result, d=9, sigmaColor=50, sigmaSpace=50)

    if apply_diffusion:
        # Anisotropic diffusion, several small steps of diffusion that
        # resist crossing strong gradients, further reducing noise inside
        # floes without blurring their edges
        diffused = result.astype(np.float64)
        kappa = 20.0     # gradient threshold, lower values preserve more edges
        gamma = 0.15     # step size, kept below 0.25 for numerical stability
        n_iter = 8
        for _ in range(n_iter):
            n = np.roll(diffused, -1, axis=0) - diffused
            s = np.roll(diffused, 1, axis=0) - diffused
            e = np.roll(diffused, -1, axis=1) - diffused
            w = np.roll(diffused, 1, axis=1) - diffused
            cn = np.exp(-(n / kappa) ** 2)
            cs = np.exp(-(s / kappa) ** 2)
            ce = np.exp(-(e / kappa) ** 2)
            cw = np.exp(-(w / kappa) ** 2)
            diffused += gamma * (cn * n + cs * s + ce * e + cw * w)
        result = np.clip(diffused, 0, 255).astype(np.uint8)

    if apply_clahe:
        # Contrast limited adaptive histogram equalization, boosts local
        # contrast so faint boundaries between similarly shaded floes
        # stand out more clearly. This is also the step most likely to
        # sharpen faint texture or lighting variation inside a single
        # floe into something that reads as a false internal boundary,
        # worth lowering clahe_clip_limit if that shows up in testing.
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size, clahe_tile_size))
        result = clahe.apply(result)

    return result


def preprocess_frame(frame, factor, x, y, apply_bilateral=True, apply_diffusion=True, apply_clahe=True,
                      clahe_clip_limit=2.0, clahe_tile_size=8):
    """
    Runs the full shared preprocessing chain on a single raw BGR frame,
    perspective correction followed by gray level homogenization, and
    returns a ready to segment enhanced grayscale frame.

    :param frame: a single raw BGR frame, as a numpy array
    :param factor: scaling factor from initial_perspective_transform
    :param x: x shift from initial_perspective_transform
    :param y: y shift from initial_perspective_transform
    :param apply_bilateral, apply_diffusion, apply_clahe, clahe_clip_limit, clahe_tile_size:
        passed straight through to enhance_gray
    :return: enhanced grayscale frame
    """
    image_trans = perspective_transform(frame, factor, x, y)
    gray = cv2.cvtColor(image_trans, cv2.COLOR_BGR2GRAY)
    return enhance_gray(gray, apply_bilateral, apply_diffusion, apply_clahe,
                         clahe_clip_limit, clahe_tile_size)