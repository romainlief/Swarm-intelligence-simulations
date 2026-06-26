import torch
import numpy as np
from .const import *
from ..common.animal import Animal

class Sheep(Animal):
    def __init__(self, position: torch.Tensor, velocity: torch.Tensor):
        super().__init__(position, velocity)

    def set_velocity(self, target_velocity: torch.Tensor):
        pass

    def update(self):
        pass
