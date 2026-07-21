import numpy as np
from src.boids.srcV1.boid import Boid
from src.boids.srcV1.const import *
import torch
from src.common.animal import Animal

class Utils:
    @staticmethod
    def getAnimalsWithin(animal, animals_list, min_radius, max_radius) -> list[Animal]:
        animals_within = []
        for other_animal in animals_list:
            if other_animal != animal:
                distance = Utils.distance(animal, other_animal)
                if distance is not None:
                    if min_radius < distance <= max_radius:
                        animals_within.append(other_animal)
        return animals_within

    @staticmethod
    def distance(animal1, animal2):
        if animal1 is not None and animal2 is not None:
            return ((animal1.position[0] - animal2.position[0]) ** 2 + (animal1.position[1] - animal2.position[1]) ** 2) ** 0.5
        return None

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

    @staticmethod
    def filter_by_view_angle(neighbors, animal, animal_angle):
        visible_neighbors = []
        for other in neighbors:
            diff = other.position - animal.position
            angle_to_neighbor = torch.atan2(diff[1], diff[0])                    
            angle_diff = angle_to_neighbor - animal_angle
            # Normalisation de l'angle entre -PI et PI (évite les bugs de saut de 0 à 2PI)
            angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
            if torch.abs(angle_diff) <= (VIEW_ANGLE / 2):
                visible_neighbors.append(other)
        return visible_neighbors
