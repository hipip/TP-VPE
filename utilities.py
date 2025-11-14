import numpy as np


def rotation_matrix(theta, axis):
    cos = np.cos(theta)
    sin = np.sin(theta)
    if axis == 'x':
        return np.array([
            [1., 0.,  0.  ],
            [0., cos, -sin],
            [0., sin, cos ],
        ])
    elif axis == 'y':
        return np.array([
            [cos,  0., sin],
            [0.,   1., 0.],
            [-sin, 1., cos],
        ])
    else:
        return np.array([
            [cos, -sin, 0.],
            [sin, cos,  0.],
            [0.,  0.,   0.],
        ])


def rotation_3d(alpha, beta, gamma):
    R_z = rotation_matrix(alpha, axis='z')
    R_y = rotation_matrix(beta, axis='y')
    R_x = rotation_matrix(gamma, axis='x')
    R = np.matmul(R_z, np.matmul(R_y, R_x))
    return R


def cube_vertices(base_point: tuple, diameter: float) -> tuple[tuple]:
    x, y, z = base_point
    return (base_point,
            (x + diameter, y, z),
            (x, y + diameter, z),
            (x + diameter, y + diameter, z),
            (x, y, z + diameter),
            (x + diameter, y, z + diameter),
            (x, y + diameter, z + diameter),
            (x + diameter, y + diameter, z + diameter))


if __name__ == "__main__":
    cube = cube_vertices((1, 1, 1), 2)
    for vertex in cube:
        print(vertex)
