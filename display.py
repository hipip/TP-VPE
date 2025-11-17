import pygame
import numpy as np
from camera import Projector

def init_display(width=800, height=600):
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Rotating 3D Cube")
    return screen


def project_point(point_3d, projector, rotation_matrix=None, translation_vector=None):
    """Project a 3D point to 2D screen coordinates using Projector class"""
    projected = projector(point_3d, rotation_matrix, translation_vector)
    return (int(projected[0]), int(projected[1]))


def draw_line(screen, point1, point2, color=(255, 255, 255), width=1):
    """Draw a line between two 2D points on the screen."""
    pygame.draw.line(screen, color, point1, point2, width)

def create_cube_vertices(size):
    """Create the 8 vertices of a cube centered at the origin."""
    points = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])
    return points * size




if __name__ == "__main__":
    point_3d = np.array([3., 3., 3.])
    projector = Projector(1.0, 1.0, 1.0, 1.0, 1.0)
    projected_point = project_point(point_3d, projector)
    print(projected_point)