import torch
from .multi_discrete_space import MultiDiscreteSpace

class MultiDiscreteSpaceGrid(MultiDiscreteSpace):
    """
    A multi-discrete space that represents a grid of discrete spaces, where each cell in the grid can take values from a discrete space with the same number of possible instances.

    Args:
        shape (tuple): The shape of the grid (e.g., (rows, cols)).
        n (int): The number of possible instances for each discrete space in the grid.
    """
    def __init__(self, shape, n):
        nvec = torch.full(shape, n, dtype=torch.int64)
        super().__init__(nvec)
