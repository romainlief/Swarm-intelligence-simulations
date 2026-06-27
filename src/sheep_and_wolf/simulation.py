from .const import *
from src.common.utils import Utils
from .wolf import Wolf
from .sheep import Sheep
import torch


class Simulation:
    def __init__(self, num_wolves: int = NUM_WOLVES, num_sheeps: int = NUM_SHEEPS):
        self.wolves = Utils.createRandomSpecies(num_wolves, Wolf)
        self.sheeps = Utils.createRandomSpecies(num_sheeps, Sheep)

    def update(self):
        new_sheeps_velocities = []

        for sheep in self.sheeps:
            repelling_sheeps = Utils.getAnimalsWithin(
                sheep,
                self.sheeps,
                0,
                S_REPULSION_RADIUS * S_CHARACTERISTIC_LENGTH,
            )
            orienting_sheeps = Utils.getAnimalsWithin(
                sheep,
                self.sheeps,
                S_REPULSION_RADIUS * S_CHARACTERISTIC_LENGTH,
                S_ORIENTATION_RADIUS * S_CHARACTERISTIC_LENGTH,
            )
            attracting_sheeps = Utils.getAnimalsWithin(
                sheep,
                self.sheeps,
                S_ORIENTATION_RADIUS * S_CHARACTERISTIC_LENGTH,
                S_ATTRACTION_RADIUS * S_CHARACTERISTIC_LENGTH,
            )
            repelling_wolves = Utils.getAnimalsWithin(
                sheep,
                self.wolves,
                0,
                S_WOLF_REPULSION_RADIUS * W_CHARACTERISTIC_LENGTH,
            )

            # FORCE DE RÉPULSION
            repulsion_force = torch.zeros(2)
            if len(repelling_sheeps) > 0:
                for other in repelling_sheeps:
                    diff = sheep.position - other.position
                    dist = torch.norm(diff)
                    if dist > 1e-8:
                        # Plus ils sont proches, plus la force de répulsion est forte (1/dist)
                        repulsion_force += (diff / dist) / dist
                repulsion_force = Utils.normalize(repulsion_force)

            # FORCE DE PEUR
            fear_force = torch.zeros(2)
            if len(repelling_wolves) > 0:
                for wolf in repelling_wolves:
                    diff = sheep.position - wolf.position
                    dist = torch.norm(diff)
                    if dist > 1e-8:
                        # Plus le loup est proche, plus la fuite est violente
                        fear_force += (diff / dist) / dist
                fear_force = Utils.normalize(fear_force)

            # FORCE D'ALIGNEMENT (ORIENTATION)
            alignment_force = torch.zeros(2)
            if len(orienting_sheeps) > 0:
                for other in orienting_sheeps:
                    alignment_force += other.velocity
                alignment_force = Utils.normalize(alignment_force)

            # FORCE D'ATTRACTION (COHÉSION)
            cohesion_force = torch.zeros(2)
            if len(attracting_sheeps) > 0:
                center_of_mass = torch.zeros(2)
                for other in attracting_sheeps:
                    center_of_mass += other.position
                center_of_mass /= len(attracting_sheeps)
                cohesion_force = center_of_mass - sheep.position
                cohesion_force = Utils.normalize(cohesion_force)

            steering = sheep.velocity.clone()

            if len(repelling_wolves) > 0:
                steering = steering * 0.1 + repulsion_force * 0.9
            elif len(repelling_sheeps) > 0:
                steering = steering * 0.2 + repulsion_force * 0.8
            else:
                steering = steering * 0.5 + alignment_force * 0.3 + cohesion_force * 0.2
            new_sheeps_velocities.append(Utils.normalize(steering))

        for sheep, velocity in zip(self.sheeps, new_sheeps_velocities):
            sheep.set_velocity(velocity)

        for sheep in self.sheeps:
            sheep.update()

        new_wolves_velocities = []
        for wolf in self.wolves:
            repelling_wolves = Utils.getAnimalsWithin(
                wolf,
                self.wolves,
                0,
                W_REPULSION_RADIUS * W_CHARACTERISTIC_LENGTH,
            )
            orienting_wolves = Utils.getAnimalsWithin(
                wolf,
                self.wolves,
                W_REPULSION_RADIUS * W_CHARACTERISTIC_LENGTH,
                W_ORIENTATION_RADIUS * W_CHARACTERISTIC_LENGTH,
            )
            attracting_wolves = Utils.getAnimalsWithin(
                wolf,
                self.wolves,
                W_ORIENTATION_RADIUS * W_CHARACTERISTIC_LENGTH,
                W_ATTRACTION_RADIUS * W_CHARACTERISTIC_LENGTH,
            )
            hunting_sheeps = Utils.getAnimalsWithin(
                wolf, self.sheeps, 0, W_WOLF_HUNTING_RADIUS * W_CHARACTERISTIC_LENGTH
            )

            # FORCE DE RÉPULSION
            repulsion_force = torch.zeros(2)
            if len(repelling_wolves) > 0:
                for other in repelling_wolves:
                    diff = wolf.position - other.position
                    dist = torch.norm(diff)
                    if dist > 1e-8:
                        repulsion_force += (diff / dist) / dist
                repulsion_force = Utils.normalize(repulsion_force)

            # FORCE D'ALIGNEMENT (ORIENTATION)
            alignment_force = torch.zeros(2)
            if len(orienting_wolves) > 0:
                for other in orienting_wolves:
                    alignment_force += other.velocity
                alignment_force = Utils.normalize(alignment_force)

            # FORCE D'ATTRACTION (COHÉSION)
            cohesion_force = torch.zeros(2)
            if len(attracting_wolves) > 0:
                center_of_mass = torch.zeros(2)
                for other in attracting_wolves:
                    center_of_mass += other.position
                center_of_mass /= len(attracting_wolves)
                cohesion_force = center_of_mass - wolf.position
                cohesion_force = Utils.normalize(cohesion_force)

            hunting_force = torch.zeros(2)
            if len(hunting_sheeps) > 0:
                closest_sheep = min(
                    hunting_sheeps, key=lambda s: Utils.distance(wolf, s)
                )
                hunting_force = (
                    closest_sheep.position - wolf.position
                )  # Direction : Vers le mouton
                hunting_force = Utils.normalize(hunting_force)
            steering = wolf.velocity.clone()

            if len(repelling_wolves) > 0:
                # Priorité absolue: ne pas se foncer dessus entre loups
                steering = steering * 0.3 + repulsion_force * 0.7
            elif len(hunting_sheeps) > 0:
                # Si pas de collision imminente entre loups, on chasse!
                # On garde un léger alignement (0.1) pour un effet de meute coordonnée pendant la chasse
                steering = steering * 0.2 + hunting_force * 0.7 + alignment_force * 0.1
            else:
                # Comportement de patrouille classique (Flocking)
                steering = steering * 0.5 + alignment_force * 0.3 + cohesion_force * 0.2
            new_wolves_velocities.append(Utils.normalize(steering))

        for wolf, velocity in zip(self.wolves, new_wolves_velocities):
            wolf.set_velocity(velocity)

        for wolf in self.wolves:
            wolf.update()
