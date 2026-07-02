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
    ) # shape of the triangle representing the animals
    
    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        sim.update()
        screen.fill((0, 0, 0))
        board = pygame.draw.rect(screen, (255, 255, 255), [BOARD_X, BOARD_Y, BOARD_WIDTH, BOARD_HEIGHT], 1) 
               
        sheeps_positions = np.array([sheep.position for sheep in sim.sheeps])
        sheeps_velocities = np.array([sheep.velocity for sheep in sim.sheeps])
        
        wolfs_positions = np.array([wolf.position for wolf in sim.wolves])
        wolfs_velocities = np.array([wolf.velocity for wolf in sim.wolves])
        
        angles_sheep = np.arctan2(sheeps_velocities[:, 1], sheeps_velocities[:, 0])
        cos_a_sheep = np.cos(angles_sheep)
        sin_a_sheep = np.sin(angles_sheep)
        
        rot_matrices_sheep = np.empty((len(sim.sheeps), 2, 2))
        rot_matrices_sheep[:, 0, 0] = cos_a_sheep
        rot_matrices_sheep[:, 0, 1] = -sin_a_sheep
        rot_matrices_sheep[:, 1, 0] = sin_a_sheep
        rot_matrices_sheep[:, 1, 1] = cos_a_sheep
        
        rotated_shapes_sheep = np.matmul(base_shape, rot_matrices_sheep.transpose(0, 2, 1))
        all_sheep_points = rotated_shapes_sheep + sheeps_positions[:, np.newaxis, :]
        
        angles_wolf = np.arctan2(wolfs_velocities[:, 1], wolfs_velocities[:, 0])
        cos_a_wolf = np.cos(angles_wolf)
        sin_a_wolf = np.sin(angles_wolf)
        
        rot_matrices_wolf = np.empty((len(sim.wolves), 2, 2))
        rot_matrices_wolf[:, 0, 0] = cos_a_wolf
        rot_matrices_wolf[:, 0, 1] = -sin_a_wolf
        rot_matrices_wolf[:, 1, 0] = sin_a_wolf
        rot_matrices_wolf[:, 1, 1] = cos_a_wolf
        
        rotated_shapes_wolf = np.matmul(base_shape, rot_matrices_wolf.transpose(0, 2, 1))
        all_wolf_points = rotated_shapes_wolf + wolfs_positions[:, np.newaxis, :]
        
        # Remplacer la boucle des moutons par :
        for i, points in enumerate(all_sheep_points):
            sheep = sim.sheeps[i]
            if sheep.alive:
                pygame.draw.polygon(screen, (0, 255, 0), points)
            else:
                pygame.draw.polygon(screen, (128, 128, 128), points)
        
        # Remplacer la boucle des loups par :
        for i, points in enumerate(all_wolf_points):
            wolf = sim.wolves[i]
            if wolf.alpha:
                pygame.draw.polygon(screen, (255, 0, 0), points)
            else:
                pygame.draw.polygon(screen, (255, 165, 0), points)
        
        pygame.display.set_caption(f"Sheep and Wolf - FPS: {int(clock.get_fps())}")
        pygame.display.flip()
    
    pygame.quit()


if __name__ == "__main__":
    main()
