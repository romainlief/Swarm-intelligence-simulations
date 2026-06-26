from abc import ABC, abstractmethod
import torch
import numpy as np
from ..sheep_and_wolf.const import *

class Animal(ABC):
    def __init__(self, position: torch.Tensor, velocity: torch.Tensor):
        self.position = position
        self.velocity = velocity
        self.direction_angle = np.arctan2(self.velocity[1].item(), self.velocity[0].item())

    @abstractmethod
    def set_velocity(self, target_velocity: torch.Tensor):
        pass

    @abstractmethod
    def update(self):
        pass
