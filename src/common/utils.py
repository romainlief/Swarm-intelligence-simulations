import numpy as np
from src.boids.srcV1.boid import Boid
from src.boids.srcV1.const import *
import torch
from src.common.animal import Animal

class Utils:
    @staticmethod
    def getAnimalsWithin(animal, animals_list, min_radius, max_radius):
        animals_within = []
        for other_animal in animals_list:
            if other_animal != animal:
                distance = Utils.distance(animal, other_animal)
                if min_radius < distance <= max_radius:
                    animals_within.append(other_animal)
        return animals_within

    @staticmethod
    def distance(animal1, animal2):
        return ((animal1.position[0] - animal2.position[0]) ** 2 + (animal1.position[1] - animal2.position[1]) ** 2) ** 0.5

    @staticmethod
    def createRandomSpecies(num, species: Animal) -> list[Animal]:
        species_list = []
        for _ in range(num):
            x = np.random.rand() * SIZE[0]
            y = np.random.rand() * SIZE[1]
            angle = np.random.rand() * 2 * np.pi

            position = torch.tensor([x, y], dtype=torch.float32)
            velocity = torch.tensor([np.cos(angle), np.sin(angle)], dtype=torch.float32)

            new_animal = species(position, velocity)
            species_list.append(new_animal)
        return species_list

    @staticmethod
    def normalize(vector: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vector)
        if norm > 1e-8:
            return vector / norm
        return torch.zeros_like(vector)
