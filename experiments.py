from camera import Projector
import utilities as util
import numpy as np
import cv2
import glob

from display import project_point


# intrinsic parameters of a smartphone wide camera (approximately!)
focal_length = 4.5e-3  # m
pixel_pitch = 1.2e-6  # m/px (square pixels)
alpha = 1 / pixel_pitch  # px/m
beta = 1 / pixel_pitch  # px/m
u0 = 2000  # px
v0 = 1500  # px

# chessboard model
BOARD_CORNERS = (7, 7)
SQUARE_SIZE = 2.5  # cm (arbitrary scale)

# camera object
camera = Projector(focal_length, alpha, beta, u0, v0)

# a point from the 3d scene [x, y, z]
point = np.array([0.2, 0.3, 0.6])


def test_projection_without_transformation():
    # the projection of the point onto the 2d plane
    res = camera(point)
    # ignore z-axis because it is useless in 2d plane
    print(res[:-1])


def test_projection_with_transformation():
    # rotate about x, y, and z axes by 20 degrees
    R = util.rotation_3d(20, 20, 20)
    # translate by 2 miters along all axes
    t = np.array([2.0, 2.0, 2.0])
    # the projection of the point onto the 2d plane
    res = camera(point, rotation_matrix=R, translation_vector=t)

    # ignore z-axis because it is useless in 2d plane
    print(res[:-1])


def video_feature_detection():
    # capturer
    cap = cv2.VideoCapture(0)

    # fast detector
    fast = cv2.FastFeatureDetector_create()
    fast.setThreshold(30)

    # main loop
    while True:
        ret, frame = cap.read()  # ret = True if frame was read successfully
        if not ret:
            break

        gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key_points: list[cv2.KeyPoint] = fast.detect(gs_frame)
        frame_kp = cv2.drawKeypoints(frame, key_points, None, (0, 0, 255))

        cv2.imshow("Webcam", frame_kp)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def chessboard_corners_detection():
    # Create a list of paths of images
    images = glob.glob("Calibration Dataset/*.jpg")
    # The path of the first image
    IMG_PATH = images[8]
    # Read the image
    img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
    if img is None:
        print("Can't open the image")
        return

    # Resize for smooth visualization
    img = cv2.resize(img, (450, 600))
    # Display image
    cv2.imshow("Chessboard", img)
    cv2.waitKey(0)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find corners
    ret, corners = cv2.findChessboardCorners(
        gray,
        BOARD_CORNERS,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
        + cv2.CALIB_CB_FAST_CHECK,
    )

    if ret:
        # Refine corners loactions
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )
        refined_corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        # Draw the refined corners on the BGR image and display it
        new_img = cv2.drawChessboardCorners(
            img, BOARD_CORNERS, refined_corners, ret
        )
        cv2.imshow("Chessboard Corners", new_img)
        cv2.waitKey(0)
    else:
        print("Corners not found")

    cv2.destroyAllWindows()


def _build_board_points():
    """3D coordinates of the checkerboard corners in the board coordinate system."""
    board_points = np.array(
        [
            (i * SQUARE_SIZE, j * SQUARE_SIZE, 0.0)
            for i in range(BOARD_CORNERS[0])
            for j in range(BOARD_CORNERS[1])
        ],
        dtype=np.float32,
    )
    return board_points.reshape(-1, 1, 3)


def _build_cube_vertices_on_board(board_points, height_scale=1.0):
    """
    Build 8 vertices of a cube whose base is centered on the chessboard.
    We reuse the cube construction utility you wrote, but shift the base
    so the cube is roughly in the middle of the board.
    """
    # Compute approximate board center in 3D (in board coordinates)
    center = board_points.reshape(-1, 3).mean(axis=0)
    cube_size = SQUARE_SIZE * height_scale
    # Place the cube so that its base square is centered at the board center
    base_point = (center[0] - cube_size / 2.0, center[1] - cube_size / 2.0, center[2])
    vertices = util.cube_vertices(base_point, cube_size)
    return np.array(vertices, dtype=np.float32)


def video_chessboard_cube():
    """
    Use the webcam, detect a chessboard, estimate its pose, and draw a simple 3D cube
    sitting on the board using your Projector and cube utilities.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    # Grab one frame to infer image size and build an approximate intrinsic matrix
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from webcam")
        cap.release()
        return

    h, w = frame.shape[:2]
    # Simple pinhole camera guess for the webcam
    focal = float(w)
    alpha_webcam = 1.0
    beta_webcam = 1.0
    u0_webcam = w / 2.0
    v0_webcam = h / 2.0

    projector = Projector(focal, alpha_webcam, beta_webcam, u0_webcam, v0_webcam)

    K = np.array(
        [
            [focal * alpha_webcam, 0.0, u0_webcam],
            [0.0, focal * beta_webcam, v0_webcam],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    board_points = _build_board_points()
    cube_vertices_3d = _build_cube_vertices_on_board(board_points)

    # Edge list for drawing cube lines, matching the vertex layout in
    # utilities.cube_vertices (0..3: base, 4..7: top)
    cube_edges = [
        [0, 1],
        [1, 3],
        [3, 2],
        [2, 0],  # base rectangle
        [4, 5],
        [5, 7],
        [7, 6],
        [6, 4],  # top rectangle
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],  # vertical edges
    ]

    # Cube faces and colors (BGR for OpenCV), ordered consistently with the
    # vertex layout so that each quad is a proper rectangle
    cube_faces = [
        [0, 1, 3, 2],  # bottom
        [4, 5, 7, 6],  # top
        [0, 1, 5, 4],  # sides
        [1, 3, 7, 5],
        [3, 2, 6, 7],
        [2, 0, 4, 6],
    ]
    face_colors = [
        (0, 255, 255),  # yellow
        (255, 0, 0),    # blue
        (0, 0, 255),    # red
        (0, 165, 255),  # orange
        (0, 255, 0),    # green
        (128, 0, 128),  # purple
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            BOARD_CORNERS,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK,
        )

        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                30,
                0.001,
            )
            refined_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )

            success, rvec, tvec = cv2.solvePnP(
                board_points, refined_corners, K, dist_coeffs
            )

            if success:
                R, _ = cv2.Rodrigues(rvec)
                R = R.astype(np.float32)
                t = tvec.reshape(3).astype(np.float32)

                # Project cube vertices using your Projector + project_point helper
                projected_points = []
                for vertex in cube_vertices_3d:
                    p2d = project_point(
                        vertex, projector, rotation_matrix=R, translation_vector=t
                    )
                    projected_points.append(p2d)

                pts = np.array(projected_points, dtype=np.int32)

                # Draw filled faces
                for face_idx, face in enumerate(cube_faces):
                    face_pts = pts[face].reshape(-1, 1, 2)
                    cv2.fillConvexPoly(frame, face_pts, face_colors[face_idx])

                # Draw cube edges on top of faces for clarity
                for edge in cube_edges:
                    i, j = edge
                    cv2.line(
                        frame,
                        projected_points[i],
                        projected_points[j],
                        (0, 0, 0),
                        2,
                    )

                # Draw black vertices
                for (x, y) in projected_points:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 0), -1)

        cv2.imshow("Webcam Chessboard with Cube", frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_chessboard_cube()
