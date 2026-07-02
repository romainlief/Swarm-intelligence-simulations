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
            raw_repelling = Utils.getAnimalsWithin(
                boid, self.boids, 0, REPULSION_RADIUS * CHARACTERISTIC_LENGTH
            )
            raw_orienting = Utils.getAnimalsWithin(
                boid,
                self.boids,
                REPULSION_RADIUS * CHARACTERISTIC_LENGTH,
                ORIENTATION_RADIUS * CHARACTERISTIC_LENGTH,
            )
            raw_attracting = Utils.getAnimalsWithin(
                boid,
                self.boids,
                ORIENTATION_RADIUS * CHARACTERISTIC_LENGTH,
                ATTRACTION_RADIUS * CHARACTERISTIC_LENGTH,
            )

            boid_angle = torch.atan2(boid.velocity[1], boid.velocity[0])
            
            repelling_boids = Utils.filter_by_view_angle(raw_repelling, boid, boid_angle)
            orienting_boids = Utils.filter_by_view_angle(raw_orienting, boid, boid_angle)
            attracting_boids = Utils.filter_by_view_angle(raw_attracting, boid, boid_angle)

            # FORCE DE RÉPULSION
            repulsion_force = torch.zeros(2)
            if len(repelling_boids) > 0:
                for other in repelling_boids:
                    diff = boid.position - other.position
                    dist = torch.norm(diff)
                    if dist > 1e-8:
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
