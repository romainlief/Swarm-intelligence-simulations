import torch
import numpy as np
from .const import *
from ..common.animal import Animal

class Sheep(Animal):
    def __init__(self, position: torch.Tensor, velocity: torch.Tensor):
        super().__init__(position, velocity)
        self.frightened: bool = False

    def set_velocity(self, target_velocity: torch.Tensor):
        if torch.norm(target_velocity) < 1e-6:
            return

        target_angle = np.arctan2(target_velocity[1].item(), target_velocity[0].item())
        
        angle_diff = target_angle - self.direction_angle
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        if abs(angle_diff) < S_TURNING_SPEED:
            self.direction_angle = target_angle
        else:
            self.direction_angle += np.sign(angle_diff) * S_TURNING_SPEED

    def update(self, frightened: bool = False):
        if not self.alive:
            return
        self.direction_angle += S_NOISE_ANGLE * (2 * np.random.rand() - 1)
        if frightened:
            self.velocity[0] = np.cos(self.direction_angle) * S_MOVING_SPEED * 4
            self.velocity[1] = np.sin(self.direction_angle) * S_MOVING_SPEED * 4
        else:
            self.velocity[0] = np.cos(self.direction_angle) * S_MOVING_SPEED
            self.velocity[1] = np.sin(self.direction_angle) * S_MOVING_SPEED
        
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

        if self.position[0] < BOARD_X + 2:
            self.position[0] = BOARD_X + 2
            self.velocity[0] *= -1
            self.direction_angle = np.arctan2(self.velocity[1].item(), self.velocity[0].item())
        elif self.position[0] > BOARD_X + BOARD_WIDTH - 2:
            self.position[0] = BOARD_X + BOARD_WIDTH - 2
            self.velocity[0] *= -1
            self.direction_angle = np.arctan2(self.velocity[1].item(), self.velocity[0].item())
            
        if self.position[1] < BOARD_Y + 2:
            self.position[1] = BOARD_Y + 2
            self.velocity[1] *= -1
            self.direction_angle = np.arctan2(self.velocity[1].item(), self.velocity[0].item())
        elif self.position[1] > BOARD_Y + BOARD_HEIGHT - 2:
            self.position[1] = BOARD_Y + BOARD_HEIGHT - 2
            self.velocity[1] *= -1
            self.direction_angle = np.arctan2(self.velocity[1].item(), self.velocity[0].item())
