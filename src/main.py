from space.multi_discrete_space_grid import MultiDiscreteSpaceGrid
from const import SIZE

def main():
    grid_space = MultiDiscreteSpaceGrid((SIZE, SIZE), 2)
    print(grid_space.nvec)
    print(grid_space.sample())
    
if __name__ == "__main__":
    main()
