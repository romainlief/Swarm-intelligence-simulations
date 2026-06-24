import pygame
import math
import random
from simulation import Simulation
from const import *


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(TITLE)
clock = pygame.time.Clock()


# DRAW BOID (triangle)
def draw_boid(screen, position, velocity, color=(255, 255, 255), size=6):
    # Explicitly convert to float/int in case they are Vector2 objects
    x, y = float(position[0]), float(position[1])
    vx, vy = float(velocity[0]), float(velocity[1])

    angle = math.atan2(vy, vx)

    # triangle base shape
    points = [
        (size, 0),
        (-size * 0.6, size * 0.5),
        (-size * 0.6, -size * 0.5),
    ]

    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    rotated_points = []
    for px, py in points:
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a
        # Ensure we are passing standard numeric pairs to Pygame
        rotated_points.append((float(x + rx), float(y + ry)))

    pygame.draw.polygon(screen, color, rotated_points)

# -------------------------
# MAIN
# -------------------------
def main():
    sim = Simulation(num_boids=NUM_BOIDS)
    running = True
    while running:
        dt = clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sim.update()

        screen.fill((0, 0, 0))

        for boid in sim.boids:
            draw_boid(
                screen,
                boid.position,
                boid.velocity,
                color=(255, 255, 255),
                size=6
            )

        # FPS display (optionnel)
        pygame.display.set_caption(
            f"Boids - FPS: {int(clock.get_fps())}"
        )
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
