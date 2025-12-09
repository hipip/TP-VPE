import pygame
import numpy as np
from utilities import rotation_3d
from display import init_display, draw_line, project_point, create_cube_vertices
from camera import Projector
from experiments import video_feature_detection

def cube_procedure():
    # Constants
    WIDTH, HEIGHT = 1280, 720
    FOCAL=500
    ALPHA=1
    BETA=1
    U0=WIDTH/2
    V0=HEIGHT/2
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)


    # Cube definition
    CUBE_SIZE = 200
    CUBE_FACES = [
        [0, 1, 2, 3],   # back face
        [4, 5, 6, 7],   # front face
        [0, 1, 5, 4],   # bottom face
        [2, 3, 7, 6],   # top face
        [0, 3, 7, 4],   # left face
        [1, 2, 6, 5],   # right face
    ]
    FACE_COLORS = [
        YELLOW,      # back face
        BLUE,     # front face    # bottom face
        RED,   # top face
        ORANGE,
        GREEN,   # left face
        PURPLE,   # right face
    ]
    CUBE_EDGES = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # back face edges
        [4, 5], [5, 6], [6, 7], [7, 4],  # front face edges
        [0, 4], [1, 5], [2, 6], [3, 7]   # connecting edges
    ]

    vertices = create_cube_vertices(CUBE_SIZE)

    def draw_cube(screen, vertices, projector, rotation_matrix):
        """Draw a cube on the screen showing faces and edges"""
        translation_vector = np.array([0., 0., 500])

        projected_points = []
        for vertex in vertices:
            point_2d = project_point(vertex, projector, rotation_matrix, translation_vector)
            projected_points.append(point_2d)

        # Draw faces
        for i, face in enumerate(CUBE_FACES):
            face_points = [projected_points[vertex_idx] for vertex_idx in face]
            pygame.draw.polygon(screen, FACE_COLORS[i], face_points)

        # Draw edges
        for edge in CUBE_EDGES:
            start_idx, end_idx = edge
            draw_line(screen, projected_points[start_idx], projected_points[end_idx], BLACK, 2)


    def main():
        screen = init_display(WIDTH, HEIGHT)
        projector = Projector(focal=FOCAL, alpha=ALPHA, beta=BETA, u0=U0, v0=V0)

        rotation_x = rotation_y = rotation_z = 0
        clock = pygame.time.Clock()
        is_running = True

        while is_running:
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

            rotation_x += 0.01
            rotation_y += 0.01
            rotation_z += 0.005
            rotation_matrix = rotation_3d(rotation_z, rotation_y, rotation_x)

            screen.fill(WHITE)
            draw_cube(screen, vertices, projector, rotation_matrix)
            pygame.display.flip()

        pygame.quit()
    
    # calling main
    main()


if __name__ == "__main__":
    video_feature_detection()
