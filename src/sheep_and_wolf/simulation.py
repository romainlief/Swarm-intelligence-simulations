from .const import *
from src.common.utils import Utils
from .wolf import Wolf
from .sheep import Sheep
import torch

class Simulation:
    def __init__(self, num_wolves: int = NUM_WOLVES, num_sheep: int = NUM_SHEEPS):
        self.wolves = Utils.createRandomSpecies(num_wolves, Wolf)
        self.sheep = Utils.createRandomSpecies(num_sheep, Sheep)

    def update(self):
        pass
