from boid import Boid
from const import *
from utils import Utils
import torch


class Simulation:
    def __init__(self, num_boids: int):
        self.boids = Utils.createRandomBoids(num_boids)

    @staticmethod
    def normalize(vector: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vector)
        if norm > 1e-8:
            return vector / norm
        return torch.zeros_like(vector)

    def update(self):
        new_velocities = []
        for boid in self.boids:
            repelling_boids = Utils.getBoidsWithin(
                boid, 0, REPULSION_RADIUS * CHARACTERISTIC_LENGTH, self
            )

            # REPULSION
            if len(repelling_boids) > 0:
                repelling_vector_sum = torch.zeros(2)
                for repelling_boid in repelling_boids:
                    repelling_vector = repelling_boid.position - boid.position
                    norm = torch.norm(repelling_vector)
                    if norm > 1e-8:
                        repelling_vector_sum += repelling_vector / norm
                repelling_vector_sum = -repelling_vector_sum
                repelling_vector_sum = self.normalize(repelling_vector_sum)
                new_velocities.append(repelling_vector_sum)
            else:
                orienting_boids = Utils.getBoidsWithin(
                    boid,
                    REPULSION_RADIUS * CHARACTERISTIC_LENGTH,
                    ORIENTATION_RADIUS * CHARACTERISTIC_LENGTH,
                    self,
                )
                attracting_boids = Utils.getBoidsWithin(
                    boid,
                    ORIENTATION_RADIUS * CHARACTERISTIC_LENGTH,
                    ATTRACTION_RADIUS * CHARACTERISTIC_LENGTH,
                    self,
                )
                # ORIENTATION
                orienting_vector_sum = torch.zeros(2)
                for orienting_boid in orienting_boids:
                    orienting_vector_sum += orienting_boid.velocity
                orienting_vector_sum = self.normalize(orienting_vector_sum)
                # ATTRACTION
                attracting_vector_sum = torch.zeros(2)
                for attracting_boid in attracting_boids:
                    attracting_vector = attracting_boid.position - boid.position
                    norm = torch.norm(attracting_vector)
                    if norm > 1e-8:
                        attracting_vector_sum += attracting_vector / norm
                attracting_vector_sum = self.normalize(attracting_vector_sum)
                # DECISION 
                if len(orienting_boids) > 0 and len(attracting_boids) == 0:
                    new_velocities.append(orienting_vector_sum)
                elif len(orienting_boids) == 0 and len(attracting_boids) > 0:
                    new_velocities.append(attracting_vector_sum)
                elif len(orienting_boids) > 0 and len(attracting_boids) > 0:
                    required_velocity = (
                        orienting_vector_sum * 0.5 + attracting_vector_sum * 0.5
                    )
                    required_velocity = self.normalize(required_velocity)
                    new_velocities.append(required_velocity)
                else:
                    # Aucun voisin
                    new_velocities.append(boid.velocity.clone())

        for boid, velocity in zip(self.boids, new_velocities):
            boid.set_velocity(velocity)

        for boid in self.boids:
            boid.update()
