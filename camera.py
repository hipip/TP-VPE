import numpy as np


class Projector:
    def __init__(self, focal, alpha, beta, u0, v0):
        self.__k = np.array([
            [focal * alpha, 0,            u0],
            [0,             focal * beta, v0],
            [0,             0,            1 ]
        ])


    def __intrinsic(self, point):
        x, y, z = point
        p = np.array([x / z, y / z, 1])
        return np.matmul(self.__k, p)
    

    def __extrinsic(self, point, rotation_matrix, translation_vector):
        return np.matmul(rotation_matrix, point) + translation_vector


    def __call__(self, point, rotation_matrix=None, translation_vector=None):
        if rotation_matrix is not None and translation_vector is not None:
            point = self.__extrinsic(point, rotation_matrix, translation_vector)
        return self.__intrinsic(point)


if __name__ == "__main__":
    focal = 1.0
    alpha = 1.0
    beta = 1.0
    u0 = 1.0
    v0 = 1.0
    projector = Projector(focal, alpha, beta, u0, v0)
    scene_point = np.array([3., 3., 3.])
    image_point = projector(scene_point)
    print(image_point)
    