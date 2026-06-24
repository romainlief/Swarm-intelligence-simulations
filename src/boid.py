import torch
from const import SIZE, TURNING_SPEED, NOISE_ANGLE, MOVING_SPEED

class Boid:
    def __init__(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
    ):
        self.position = position.float()
        self.velocity = velocity.float()
        self.direction = self.velocity / torch.norm(self.velocity) 
        
    def set_velocity(self, velocity: torch.Tensor):
        new_direction = velocity / torch.norm(velocity)
        
        if torch.all(torch.abs(new_direction - self.direction) < TURNING_SPEED):
            self.direction = new_direction;
        else:
            if (new_direction < self.direction).all():
                if (torch.abs(new_direction - self.direction) < torch.pi).all():
                    self.direction -= TURNING_SPEED
                else:
                    self.direction += TURNING_SPEED
            else:
                if (torch.abs(new_direction - self.direction) < torch.pi).all():
                    self.direction += TURNING_SPEED
                else:
                    self.direction -= TURNING_SPEED

        self.velocity = self.direction * torch.norm(velocity)
        
    def update(self):
        perturbated_direction = self.direction + 2 * NOISE_ANGLE * (torch.rand(1) - 0.5)
        perturbated_velocity = perturbated_direction * torch.norm(self.velocity)
        
        self.position[0] += MOVING_SPEED * perturbated_velocity[0]
        self.position[1] += MOVING_SPEED * perturbated_velocity[1]

        if (self.position[0] < 0):
                self.position[0] = 0
                self.velocity[0] *= -1
                self.direction = self.velocity / torch.norm(self.velocity) 
        elif (self.position[0] > SIZE[0]):
                self.position[0] = SIZE[0]
                self.velocity[0] *= -1
                self.direction = self.velocity / torch.norm(self.velocity)
        if (self.position[1] < 0):
                self.position[1] = 0
                self.velocity[1] *= -1
                self.direction = self.velocity / torch.norm(self.velocity)
        elif (self.position[1] > SIZE[1]):
                self.position[1] = SIZE[1]
                self.velocity[1] *= -1
                self.direction = self.velocity / torch.norm(self.velocity)
        
        self.constrain_direction();
    
    def constrain_direction(self):
        while (self.direction < -torch.pi).all():
            self.direction += 2 * torch.pi
        while (self.direction > torch.pi).all():
            self.direction -= 2 * torch.pi
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    # def move(self, dt: float = 1.0):
    #     self.position += self.velocity * dt

    # def limit_speed(self):
    #     speed = torch.norm(self.velocity)

    #     if speed > self.max_speed:
    #         self.velocity *= self.max_speed / speed

    # #def update(self, dt: float = 1.0):
    # #    self.limit_speed()
    # #    self.move(dt)
        
    # def flock(self, world):
    #     neighbors = self.__get_neighbors(world)
    #     sep = self.__separation(neighbors) * 3.0
    #     ali = self.__alignment(neighbors) * 5.0
    #     coh = self.__cohesion(neighbors) * 1.0
    #     self.velocity += sep + ali + coh

    # def __repr__(self):
    #     return (
    #         f"Boid("
    #         f"pos={self.position.tolist()}, "
    #         f"vel={self.velocity.tolist()}"
    #         f")"
    #     )
    
    # def __get_neighbors(self, world):
    #     neighbors = []
    #     for other in world.boids:
    #         if other is self:
    #             continue
    #         distance = torch.norm(
    #             other.position - self.position
    #         )
    #         if distance < self.vision_radius:
    #             neighbors.append(other)
    #     return neighbors

    # def __separation(self, neighbors):
    #     force = torch.zeros(2)
    #     for neighbor in neighbors:
    #         diff = self.position - neighbor.position
    #         dist = torch.norm(diff)
    #         if dist > 0:
    #             force += diff / dist
    #     return force
    
    # def __alignment(self, neighbors):
    #     if not neighbors:
    #         return torch.zeros(2)
    #     avg_velocity = torch.stack(
    #         [n.velocity for n in neighbors]
    #     ).mean(dim=0)
    #     return avg_velocity - self.velocity

    # def __cohesion(self, neighbors):
    #     if not neighbors:
    #         return torch.zeros(2)
    #     center = torch.stack(
    #         [n.position for n in neighbors]
    #     ).mean(dim=0)
    #     return center - self.position
