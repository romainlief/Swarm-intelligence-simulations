import pygame
import numpy as np
from simulation import Simulation
from const import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(TITLE)
clock = pygame.time.Clock()


def main():
    sim = Simulation(num_boids=NUM_BOIDS)
    running = True
    size = 6
    base_shape = np.array(
        [[size, 0], [-size * 0.6, size * 0.5], [-size * 0.6, -size * 0.5]]
    )

    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sim.update()
        screen.fill((0, 0, 0))

        positions = np.array([boid.position for boid in sim.boids])
        velocities = np.array([boid.velocity for boid in sim.boids])

        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        rot_matrices = np.empty((len(sim.boids), 2, 2))
        rot_matrices[:, 0, 0] = cos_a
        rot_matrices[:, 0, 1] = -sin_a
        rot_matrices[:, 1, 0] = sin_a
        rot_matrices[:, 1, 1] = cos_a

        rotated_shapes = np.matmul(base_shape, rot_matrices.transpose(0, 2, 1))
        all_boids_points = rotated_shapes + positions[:, np.newaxis, :]

        for points in all_boids_points:
            pygame.draw.polygon(screen, (255, 255, 255), points)

        pygame.display.set_caption(f"Boids - FPS: {int(clock.get_fps())}")
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
