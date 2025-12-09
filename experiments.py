from camera import Projector
import utilities as util
import numpy as np
import cv2


# intrinsic parameters of a smartphone wide camera (approximately!)
focal_length = 4.5e-3 # m

pixel_pitch = 1.2e-6 # m/px (square pixels)

alpha = 1 / pixel_pitch # px/m
beta = 1 / pixel_pitch # px/m
u0 = 2000 # px
v0 = 1500 # px
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
    t = np.array([2. , 2., 2.])
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
        ret, frame = cap.read()    # ret = True if frame was read successfully
        if not ret:
            break

        gs_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        key_points: list[cv2.KeyPoint] = fast.detect(gs_frame)
        frame_kp = cv2.drawKeypoints(frame, key_points, None, (0, 0, 255))

        cv2.imshow("Webcam", frame_kp)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def chessboard_corners_detection():
    pass


if __name__ == '__main__':
    video_feature_detection()
