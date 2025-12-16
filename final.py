import numpy as np
import glob
import cv2 as cv
from tqdm import tqdm

BOARD_CORNERS = (7, 7)
SQUARE_SIZE = 2.5  # 2.5 cm
DATASET = 'Calibration Dataset/*.jpg'
PREPROCESSED_DATASET_FOLDER = 'Preprocessed calibration dataset'
IMG_SIZE = (3000, 4000)


def preprocess_dataset(images_paths):
    '''Rotate the image in the dataset that has width != 3000 and height != 4000'''
    for img_path in images_paths:
        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img.shape[0:2] != (4000, 3000):
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        img = cv.resize(img, (450, 600))
        cv.imwrite(img_path, img)


def get_3d_coordinates_of_corners(images_paths):
    # The 3D coordinates of the checkerboard corners
    board_points = np.array([(i, j, 0) for i in range(BOARD_CORNERS[0]) for j in range(BOARD_CORNERS[1])], dtype=np.float32) * SQUARE_SIZE
    print(board_points.shape)
    board_points = board_points.reshape(-1, 1, 3)
    print(board_points.shape)
    return [board_points] * len(images_paths)


def get_2d_coordinates_of_corners(images_paths, visualize=False, delay=500):
    # Load images in the grayscale format
    grays = [cv.imread(img_path, flags=cv.IMREAD_GRAYSCALE) for img_path in images_paths]

    # To store the 2D coordinates of corners for each image in the dataset
    image_points = []

    # Criteria used for refining the corners
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for img in tqdm(grays):
        # Find corners
        ret, corners = cv.findChessboardCorners(img,
                                                BOARD_CORNERS,
                                                flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK)

        if ret:
            # Refine corners locations
            refined_corners = cv.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            # Append the refind corners to the list of 2D coordinates
            image_points.append(refined_corners)

            # Visualize
            if visualize:
                new = cv.drawChessboardCorners(img, BOARD_CORNERS, refined_corners, ret)
                cv.imshow('Corners', new)
                cv.waitKey(delay)
    
    cv.destroyAllWindows()
    return image_points


def calibrate_camera(real_corners_set, image_corners_set, image_size):
    # Calibrate the camera to find the intrinsic matrix K
    ret, K, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(real_corners_set,
                                                           image_corners_set,
                                                           image_size,
                                                           None,
                                                           None)
    
    # Print info about the result of calibration
    print("RMS reprojection error:", ret)
    print("Camera matrix K:\n", K)
    print("Distortion coefficients:\n", dist_coeffs)

    return K


if __name__ == '__main__':
    # Paths of images in the dataset
    images_paths = glob.glob(DATASET)

    # # Preprocess the dataset to ensure consistent size across images
    # preprocess_dataset(images_paths)

    # 3D coordinates of corners of each image in the dataset (The same 3D coordinates for them all)
    real_corners = get_3d_coordinates_of_corners(images_paths)
    print(f'len(real_corners): {len(real_corners)}')
    for rc in real_corners[0]:
        print(rc)

    # 2D coordinates of corners of each image in the dataset
    image_corners = get_2d_coordinates_of_corners(images_paths, visualize=False)
    print(f'len(image_corners): {len(image_corners)}')
    for ic in image_corners[0]:
        print(ic)

    # The intrinsic matrix K
    K = calibrate_camera(real_corners, image_corners, IMG_SIZE)
    