import pygame
import numpy as np
from .simulation import Simulation
from .const import *

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(TITLE)
clock = pygame.time.Clock()

def main():
    sim = Simulation(num_wolves=NUM_WOLVES, num_sheeps=NUM_SHEEPS)
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
        
        sheep_positions = np.array([sheep.position for sheep in sim.sheeps])
        sheep_velocities = np.array([sheep.velocity for sheep in sim.sheeps])
        
        wolf_positions = np.array([wolf.position for wolf in sim.wolves])
        wolf_velocities = np.array([wolf.velocity for wolf in sim.wolves])
        
        angles_sheep = np.arctan2(sheep_velocities[:, 1], sheep_velocities[:, 0])
        cos_a_sheep = np.cos(angles_sheep)
        sin_a_sheep = np.sin(angles_sheep)
        
        rot_matrices_sheep = np.empty((len(sim.sheeps), 2, 2))
        rot_matrices_sheep[:, 0, 0] = cos_a_sheep
        rot_matrices_sheep[:, 0, 1] = -sin_a_sheep
        rot_matrices_sheep[:, 1, 0] = sin_a_sheep
        rot_matrices_sheep[:, 1, 1] = cos_a_sheep
        
        rotated_shapes_sheep = np.matmul(base_shape, rot_matrices_sheep.transpose(0, 2, 1))
        all_sheep_points = rotated_shapes_sheep + sheep_positions[:, np.newaxis, :]
        
        angles_wolf = np.arctan2(wolf_velocities[:, 1], wolf_velocities[:, 0])
        cos_a_wolf = np.cos(angles_wolf)
        sin_a_wolf = np.sin(angles_wolf)
        
        rot_matrices_wolf = np.empty((len(sim.wolves), 2, 2))
        rot_matrices_wolf[:, 0, 0] = cos_a_wolf
        rot_matrices_wolf[:, 0, 1] = -sin_a_wolf
        rot_matrices_wolf[:, 1, 0] = sin_a_wolf
        rot_matrices_wolf[:, 1, 1] = cos_a_wolf
        
        rotated_shapes_wolf = np.matmul(base_shape, rot_matrices_wolf.transpose(0, 2, 1))
        all_wolf_points = rotated_shapes_wolf + wolf_positions[:, np.newaxis, :]
        
        for points in all_sheep_points:
            pygame.draw.polygon(screen, (0, 255, 0), points)
        for points in all_wolf_points:
            if sim.wolves[all_wolf_points.tolist().index(points.tolist())].alpha:
                pygame.draw.polygon(screen, (255, 0, 0), points)  # Red for alpha wolf
            else:
                pygame.draw.polygon(screen, (255, 165, 0), points)  # Orange for other wolves
        
        pygame.display.set_caption(f"Sheep and Wolf - FPS: {int(clock.get_fps())}")
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    main()
