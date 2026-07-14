from .const import *
from src.common.utils import Utils
from .wolf import Wolf
from .sheep import Sheep
import torch


class Simulation:
    def __init__(self, num_wolves: int = NUM_WOLVES, num_sheeps: int = NUM_SHEEPS):
        self.repulsion_force = None
        self.fear_force = None
        self.alignment_force = None
        self.cohesion_force = None
        self.steering = None

        # FOR SHEEPS
        self.sheeps: list[Sheep] = Utils.createRandomSpecies(num_sheeps, Sheep)  # type: ignore
        self.sheeps_alive = [sheep for sheep in self.sheeps if sheep.alive]

        self.repelling_sheeps = None
        self.orienting_sheeps = None
        self.attracting_sheeps = None
        self.repelling_wolves_for_sheep = None
        self.new_sheeps_velocities = []

        # FOR WOLFS
        self.wolves: list[Wolf] = Utils.createRandomSpecies(num_wolves, Wolf)  # type: ignore
        self.wolves[0].alpha = True

        self.repelling_wolves_for_wolf = None
        self.orienting_wolves = None
        self.attracting_wolves = None
        self.hunting_sheeps = []
        self.new_wolves_velocities = []

    def update(self):
        self.sheeps_alive = [sheep for sheep in self.sheeps if sheep.alive]
        self.new_sheeps_velocities = []

        for sheep in self.sheeps:
            self.compute_sheep_env(sheep)
            if (
                self.repelling_wolves_for_sheep is not None
                and len(self.repelling_wolves_for_sheep) > 0
            ):
                sheep.frightened = True
            else:
                sheep.frightened = False

            # FORCE DE RÉPULSION
            self.repulsion_force = torch.zeros(2)
            if self.repelling_sheeps is not None and len(self.repelling_sheeps) > 0:
                for other in self.repelling_sheeps:
                    diff = sheep.position - other.position
                    dist = torch.norm(diff)
                    if dist > 1e-8:
                        # Plus ils sont proches, plus la force de répulsion est forte (1/dist)
                        self.repulsion_force += (diff / dist) / dist
                self.repulsion_force = Utils.normalize(self.repulsion_force)

            # FORCE DE PEUR
            self.fear_force = torch.zeros(2)
            if (
                self.repelling_wolves_for_sheep is not None
                and len(self.repelling_wolves_for_sheep) > 0
            ):
                for wolf in self.repelling_wolves_for_sheep:
                    diff = sheep.position - wolf.position
                    dist = torch.norm(diff)
                    if dist > 1e-8:
                        # Plus le loup est proche, plus la fuite est violente
                        self.fear_force += (diff / dist) / dist
                self.fear_force = Utils.normalize(self.fear_force)

            # FORCE D'ALIGNEMENT (ORIENTATION)
            self.alignment_force = torch.zeros(2)
            if self.orienting_sheeps is not None and len(self.orienting_sheeps) > 0:
                for other in self.orienting_sheeps:
                    self.alignment_force += other.velocity
                self.alignment_force = Utils.normalize(self.alignment_force)

            # FORCE D'ATTRACTION (COHÉSION)
            self.cohesion_force = torch.zeros(2)
            if self.attracting_sheeps is not None and len(self.attracting_sheeps) > 0:
                center_of_mass = torch.zeros(2)
                for other in self.attracting_sheeps:
                    center_of_mass += other.position
                center_of_mass /= len(self.attracting_sheeps)
                self.cohesion_force = center_of_mass - sheep.position
                self.cohesion_force = Utils.normalize(self.cohesion_force)

            self.steering = sheep.velocity.clone()
            if (
                self.repelling_wolves_for_sheep is not None
                and len(self.repelling_wolves_for_sheep) > 0
            ):
                self.steering = self.steering * 0.1 + self.fear_force * 0.9
            elif self.repelling_sheeps is not None and len(self.repelling_sheeps) > 0:
                self.steering = self.steering * 0.2 + self.repulsion_force * 0.8
            else:
                self.steering = (
                    self.steering * 0.5
                    + self.alignment_force * 0.3
                    + self.cohesion_force * 0.2
                )
            self.new_sheeps_velocities.append(Utils.normalize(self.steering))

        self.update_animal(self.sheeps, self.new_sheeps_velocities)

        self.new_wolves_velocities = []
        for wolf in self.wolves:
            self.compute_wolf_env(wolf)

            # FORCE DE RÉPULSION
            self.repulsion_force = torch.zeros(2)
            if (
                self.repelling_wolves_for_wolf is not None
                and len(self.repelling_wolves_for_wolf) > 0
            ):
                for other in self.repelling_wolves_for_wolf:
                    diff = wolf.position - other.position
                    dist = torch.norm(diff)
                    if dist > 1e-8:
                        self.repulsion_force += (diff / dist) / dist
                self.repulsion_force = Utils.normalize(self.repulsion_force)

            # FORCE D'ALIGNEMENT (ORIENTATION)
            self.alignment_force = torch.zeros(2)
            if self.orienting_wolves is not None and len(self.orienting_wolves) > 0:
                for other in self.orienting_wolves:
                    self.alignment_force += other.velocity
                self.alignment_force = Utils.normalize(self.alignment_force)

            # FORCE D'ATTRACTION (COHÉSION)
            self.cohesion_force = torch.zeros(2)
            if self.attracting_wolves is not None and len(self.attracting_wolves) > 0:
                center_of_mass = torch.zeros(2)
                for other in self.attracting_wolves:
                    center_of_mass += other.position
                center_of_mass /= len(self.attracting_wolves)
                self.cohesion_force = center_of_mass - wolf.position
                self.cohesion_force = Utils.normalize(self.cohesion_force)

            self.hunting_force = torch.zeros(2)
            if len(self.hunting_sheeps) > 0:
                closest_sheep = min(
                    self.hunting_sheeps, key=lambda s: Utils.distance(wolf, s)
                )
                self.hunting_force = (
                    closest_sheep.position - wolf.position
                )  # Direction: Vers le mouton
                self.hunting_force = Utils.normalize(self.hunting_force)
            self.steering = wolf.velocity.clone()

            if (
                self.repelling_wolves_for_wolf is not None
                and len(self.repelling_wolves_for_wolf) > 0
            ):  # Ne pas se foncer dessus entre loups
                self.steering = self.steering * 0.3 + self.repulsion_force * 0.7
            elif len(self.hunting_sheeps) > 0:  # On chasse
                self.steering = (
                    self.steering * 0.4
                    + self.hunting_force * 0.4
                    + self.alignment_force * 0.2
                )
            else:  # Comportement de patrouille classique (Flocking)
                self.steering = (
                    self.steering * 0.5
                    + self.alignment_force * 0.3
                    + self.cohesion_force * 0.2
                )
            self.new_wolves_velocities.append(Utils.normalize(self.steering))

        self.update_animal(self.wolves, self.new_wolves_velocities)
        self.update_life_sheep()

    def update_life_sheep(self):
        for sheep in self.sheeps:
            if sheep.alive:
                distance = np.linalg.norm(self.wolves[0].position - sheep.position)
                if distance < SEUIL_COLLISION:
                    sheep.alive = False

    def update_animal(self, animals, new_velocities):
        for animal, velocity in zip(animals, new_velocities):
            animal.set_velocity(velocity)
        for animal in animals:
            if animals == self.sheeps:
                animal.update(animal.frightened)
            else:
                animal.update()

    def compute_sheep_env(self, sheep):
        self.repelling_sheeps = Utils.getAnimalsWithin(
            sheep,
            self.sheeps_alive,
            0,
            S_REPULSION_RADIUS * S_CHARACTERISTIC_LENGTH,
        )
        self.orienting_sheeps = Utils.getAnimalsWithin(
            sheep,
            self.sheeps_alive,
            S_REPULSION_RADIUS * S_CHARACTERISTIC_LENGTH,
            S_ORIENTATION_RADIUS * S_CHARACTERISTIC_LENGTH,
        )
        self.attracting_sheeps = Utils.getAnimalsWithin(
            sheep,
            self.sheeps_alive,
            S_ORIENTATION_RADIUS * S_CHARACTERISTIC_LENGTH,
            S_ATTRACTION_RADIUS * S_CHARACTERISTIC_LENGTH,
        )
        self.repelling_wolves_for_sheep = Utils.getAnimalsWithin(
            sheep,
            self.wolves,
            0,
            S_WOLF_REPULSION_RADIUS * W_CHARACTERISTIC_LENGTH,
        )

    def compute_wolf_env(self, wolf):
        self.repelling_wolves_for_wolf = Utils.getAnimalsWithin(
            wolf,
            self.wolves,
            0,
            W_REPULSION_RADIUS * W_CHARACTERISTIC_LENGTH,
        )
        self.orienting_wolves = Utils.getAnimalsWithin(
            wolf,
            self.wolves,
            W_REPULSION_RADIUS * W_CHARACTERISTIC_LENGTH,
            W_ORIENTATION_RADIUS * W_CHARACTERISTIC_LENGTH,
        )
        self.attracting_wolves = Utils.getAnimalsWithin(
            wolf,
            [self.wolves[0]],
            W_ORIENTATION_RADIUS * W_CHARACTERISTIC_LENGTH,
            W_ATTRACTION_RADIUS * W_CHARACTERISTIC_LENGTH,
        )
        if wolf.alpha:
            self.hunting_sheeps = Utils.getAnimalsWithin(
                wolf,
                self.sheeps_alive,
                0,
                W_WOLF_HUNTING_RADIUS * W_CHARACTERISTIC_LENGTH,
            )
        else:
            self.hunting_sheeps = []
