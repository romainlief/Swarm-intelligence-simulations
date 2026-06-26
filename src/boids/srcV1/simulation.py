from .const import *
from src.common.utils import Utils
import torch
from .boid import Boid
class Simulation:
    def __init__(self, num_boids: int = NUM_BOIDS):
        self.boids = Utils.createRandomSpecies(num_boids, Boid)

    def update(self):
        new_velocities = []
        
        for boid in self.boids:
            repelling_boids = Utils.getBoidsWithin(boid, 0, REPULSION_RADIUS * CHARACTERISTIC_LENGTH, self)
            orienting_boids = Utils.getBoidsWithin(boid, REPULSION_RADIUS * CHARACTERISTIC_LENGTH, ORIENTATION_RADIUS * CHARACTERISTIC_LENGTH, self)
            attracting_boids = Utils.getBoidsWithin(boid, ORIENTATION_RADIUS * CHARACTERISTIC_LENGTH, ATTRACTION_RADIUS * CHARACTERISTIC_LENGTH, self)

            # FORCE DE RÉPULSION 
            repulsion_force = torch.zeros(2)
            if len(repelling_boids) > 0:
                for other in repelling_boids:
                    diff = boid.position - other.position
                    dist = torch.norm(diff)
                    if dist > 1e-8:
                        # Plus ils sont proches, plus la force de répulsion est forte (1/dist)
                        repulsion_force += (diff / dist) / dist
                repulsion_force = Utils.normalize(repulsion_force)

            # FORCE D'ALIGNEMENT (ORIENTATION)
            alignment_force = torch.zeros(2)
            if len(orienting_boids) > 0:
                for other in orienting_boids:
                    alignment_force += other.velocity
                alignment_force = Utils.normalize(alignment_force)

            # FORCE D'ATTRACTION (COHÉSION)
            cohesion_force = torch.zeros(2)
            if len(attracting_boids) > 0:
                center_of_mass = torch.zeros(2)
                for other in attracting_boids:
                    center_of_mass += other.position
                center_of_mass /= len(attracting_boids)
                cohesion_force = center_of_mass - boid.position
                cohesion_force = Utils.normalize(cohesion_force)

            steering = boid.velocity.clone()
            
            if len(repelling_boids) > 0:
                steering = steering * 0.2 + repulsion_force * 0.8
            else:
                steering = steering * 0.5 + alignment_force * 0.3 + cohesion_force * 0.2
            
            new_velocities.append(Utils.normalize(steering))

        for boid, velocity in zip(self.boids, new_velocities):
            boid.set_velocity(velocity)

        for boid in self.boids:
            boid.update()
