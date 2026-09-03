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

        self.closest_sheep = None

        # FOR WOLFS
        self.wolves: list[Wolf] = Utils.createRandomSpecies(num_wolves, Wolf)  # type: ignore
        self.wolves[0].alpha = True

        self.repelling_wolves_for_wolf = None
        self.orienting_wolves = None
        self.attracting_wolves = None
        self.hunting_sheeps = []
        self.new_wolves_velocities = []
        self.hunting_force = None

    def update(self):
        self.sheeps_alive = [sheep for sheep in self.sheeps if sheep.alive]
        self.new_sheeps_velocities = []

        for sheep in self.sheeps:
            self.compute_sheep_env(sheep)
            sheep.frightened = bool(self.repelling_wolves_for_sheep)
            self.compute_repulsion_force(self.repelling_sheeps, sheep)
            self.compute_fear_force_for(self.repelling_wolves_for_sheep, sheep)
            self.compute_alignement_force(self.orienting_sheeps, sheep)
            self.compute_cohesion_force(self.attracting_sheeps, sheep)
            
            self.steering = sheep.velocity.clone()
            if self.repelling_wolves_for_sheep and self.fear_force is not None:
                self.steering = self.steering * 0.1 + self.fear_force * 0.9
            elif self.repelling_sheeps and self.repulsion_force is not None:
                self.steering = self.steering * 0.2 + self.repulsion_force * 0.8
            elif self.alignment_force is not None and self.cohesion_force is not None:
                self.steering = (
                    self.steering * 0.5
                    + self.alignment_force * 0.3
                    + self.cohesion_force * 0.2
                )
            self.new_sheeps_velocities.append(Utils.normalize(self.steering))

        self.update_animal(self.sheeps, self.new_sheeps_velocities)

        alpha_wolf = next((w for w in self.wolves if w.alpha), self.wolves[0])

        sheeps_near_alpha = Utils.getAnimalsWithin(
            alpha_wolf,
            self.sheeps_alive,
            0,
            W_WOLF_HUNTING_RADIUS * W_CHARACTERISTIC_LENGTH,
        )

        # Si l'Alpha voit des moutons, il verrouille le plus proche pour toute la meute
        pack_target_sheep = None
        if sheeps_near_alpha:
            pack_target_sheep = min(
                sheeps_near_alpha, key=lambda s: Utils.distance(alpha_wolf, s)  # type: ignore
            )  # type: ignore

        # maj loup chasse en meute
        self.new_wolves_velocities = []
        for wolf in self.wolves:
            self.compute_wolf_env(wolf)
            self.compute_repulsion_force(self.repelling_wolves_for_wolf, wolf)
            self.compute_alignement_force(self.orienting_wolves, wolf)
            self.compute_cohesion_force(self.attracting_wolves, wolf)
            self.compute_hunting_force(pack_target_sheep, wolf)
            self.steering = wolf.velocity.clone()

            # Prise de décision du loup
            if self.repelling_wolves_for_wolf and self.repulsion_force is not None:
                # Éviter de se rentrer dedans entre loups
                self.steering = self.steering * 0.3 + self.repulsion_force * 0.7
            elif pack_target_sheep is not None and self.cohesion_force is not None and self.hunting_force is not None:
                # Si l'Alpha a donné une cible -> Tout le monde chasse
                self.steering = (
                    self.steering * 0.3
                    + self.hunting_force * 0.5
                    + self.cohesion_force * 0.2
                )
            else:
                # Pas de cible -> Patrouille/Flocking classique
                if self.alignment_force is not None and self.cohesion_force is not None:
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
                for wolf in self.wolves:
                    distance = np.linalg.norm(wolf.position - sheep.position)
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
            self.wolves,
            W_ORIENTATION_RADIUS * W_CHARACTERISTIC_LENGTH,
            W_ATTRACTION_RADIUS * W_CHARACTERISTIC_LENGTH,
        )

    def compute_repulsion_force(self, repelling_animals, animal):
        self.repulsion_force = torch.zeros(2)
        if repelling_animals is not None and len(repelling_animals) > 0:
            for other in repelling_animals:
                diff = animal.position - other.position
                dist = torch.norm(diff)
                if dist > 1e-8:
                    # Plus ils sont proches, plus la force de répulsion est forte (1/dist)
                    self.repulsion_force += (diff / dist) / dist
            self.repulsion_force = Utils.normalize(self.repulsion_force)

    def compute_fear_force_for(self, repelling_animals_for, animal_for):
        self.fear_force = torch.zeros(2)
        if repelling_animals_for is not None and len(repelling_animals_for) > 0:
            for wolf in repelling_animals_for:
                diff = animal_for.position - wolf.position
                dist = torch.norm(diff)
                if dist > 1e-8:
                    # Plus le loup est proche, plus la fuite est violente
                    self.fear_force += (diff / dist) / dist
            self.fear_force = Utils.normalize(self.fear_force)

    def compute_alignement_force(self, orienting_animals, animal):
        self.alignment_force = torch.zeros(2)
        if orienting_animals is not None and len(orienting_animals) > 0:
            for animal in orienting_animals:
                self.alignment_force += animal.velocity
            self.alignment_force = Utils.normalize(self.alignment_force)

    def compute_cohesion_force(self, attracting_animals, animal):
        self.cohesion_force = torch.zeros(2)
        if attracting_animals:
            center_of_mass = sum(s.position for s in attracting_animals) / len(
                attracting_animals
            )
            self.cohesion_force = Utils.normalize(center_of_mass - animal.position)

    def compute_hunting_force(self, target_sheep, wolf):
        self.hunting_force = torch.zeros(2)
        if target_sheep is not None:
            self.hunting_force = Utils.normalize(target_sheep.position - wolf.position)
