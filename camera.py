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
    

    def __extrinsic(self, point, rotation_matrix=None, traslation_vector=None):
        R_T = np.concatenate((rotation_matrix, traslation_vector), axis=1)
        p = np.append(point, np.array([1]), axis=0)
        projection_matrix = np.matmul(self.__k, R_T)
        return np.matmul(projection_matrix, p)


    def __call__(self, point, rotation_matrix=None, traslation_vector=None):
        if rotation_matrix is not None and traslation_vector is not None:
            return self.__extrinsic(point, rotation_matrix, traslation_vector)
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
    