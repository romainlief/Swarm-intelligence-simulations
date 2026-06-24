import numpy as np
from boid import Boid
from const import *
import torch

class Utils:
    @staticmethod
    def getBoidsWithin(boid, min_radius, max_radius, simulation):
        boids_within = []
        for other_boid in simulation.boids:
            if other_boid != boid:
                distance = Utils.distance(boid, other_boid)
                if min_radius < distance <= max_radius:
                    boids_within.append(other_boid)
        return boids_within
    
    @staticmethod
    def distance(boid1, boid2):
        return ((boid1.position[0] - boid2.position[0]) ** 2 + (boid1.position[1] - boid2.position[1]) ** 2) ** 0.5

    @staticmethod
    def createRandomBoids(num) -> list[Boid]:
        boids = []
        for _ in range(num):
            x = np.random.rand() * SIZE[0]
            y = np.random.rand() * SIZE[1]
            angle = np.random.rand() * 2 * np.pi

            position = torch.tensor([x, y], dtype=torch.float32)
            velocity = torch.tensor([np.cos(angle), np.sin(angle)], dtype=torch.float32)

            new_boid = Boid(position, velocity)
            boids.append(new_boid)
        return boids

