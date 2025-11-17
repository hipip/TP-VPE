import pygame
import numpy as np
from display import init_display, draw_line, project_point,create_cube_vertices
from camera import Projector

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Cube definition
CUBE_SIZE = 100
CUBE_FACES = [
    [0, 1, 2, 3],   # back face
    [4, 5, 6, 7],   # front face
    [0, 1, 5, 4],   # bottom face
    [2, 3, 7, 6],   # top face
    [0, 3, 7, 4],   # left face
    [1, 2, 6, 5],   # right face
]
FACE_COLORS = [
    RED,      # back face
    BLUE,     # front face
    GREEN,    # bottom face
    YELLOW,   # top face
    ORANGE,   # left face
    PURPLE,   # right face
]
CUBE_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  # back face edges
    [4, 5], [5, 6], [6, 7], [7, 4],  # front face edges
    [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
]

vertices = create_cube_vertices(CUBE_SIZE)

def draw_cube(screen, vertices, projector):
    """Draw a static cube on the screen showing all edges"""
    identity_rotation = np.eye(3)  # Identity matrix = no rotation
    translation_vector = np.array([0., 0., 500])
    
    # Project all vertices to 2D using Projector's built-in translation
    projected_points = []
    for vertex in vertices:
        point_2d = project_point(vertex, projector, identity_rotation, translation_vector)
        projected_points.append(point_2d)
    
    # Draw all edges
    for edge in CUBE_EDGES:
        start_idx, end_idx = edge
        draw_line(screen, projected_points[start_idx], projected_points[end_idx], BLACK, 2)


if __name__ == "__main__":
    # Initialize display
    screen = init_display(WIDTH, HEIGHT)
    
    projector = Projector(focal=500, alpha=1, beta=1, u0=WIDTH/2, v0=HEIGHT/2)
    
    clock = pygame.time.Clock()
    is_running = True
    
    while is_running:
        clock.tick(60)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
        
        screen.fill(WHITE)
        draw_cube(screen, vertices, projector)
        pygame.display.flip()
    
    pygame.quit()

