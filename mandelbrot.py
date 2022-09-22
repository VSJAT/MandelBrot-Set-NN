#Functions for creating the Mandelbrot Dataset

import torch 
import random
from torch.utils.data.dataset import Dataset


# function helper, don't directly call
def _m(a, max_depth):
  z = 0
  for n in range(1, max_depth):
    z = z**2 + a
    if abs(z) > 2:
      return min((n-1)/50, 1)
      # return -(math.cos((n-1)*math.pi/50)/2) + (1/2)
  return 1.0


def mandelbrot_calc(x, y, max_depth=50):
  """ 
  Calculates whether the given point is in the mandelbrot set.
  Parameters: 
  x (float): real part of the number
  y (float): complex part of the number
  max_depth (int): Maximum number of recursive steps before deciding\
  whether the value is in the mandelbrot set
  Returns: 
  float: Number between 1 and 0 where 1.0 is in the mandelbrot set and\
  values closer to 1.0 required more steps to determine this
  """
  return _m(x + 1j * y, max_depth)


class MandelbrotDataSet(Dataset):
    """ 
    Creates a dataset of randomized points and their calculated mandelbrot values.
  
    Parameters: 
    size (int): number of randomized points to generate
    max_depth (int): Maximum number of recursive steps before deciding\
      whether the value is in the mandelbrot set
    xmin (float): minimum x value for points
    xmax (float): maximum x value for points
    ymin (float): minimum y value for points
    ymax (float): maximum y value for points
    """
    def __init__(self, size=1000, max_depth=50, xmin=-2.5, xmax=1.0, ymin=-1.1, ymax=1.1):
        self.input = []
        self.output = []
        
        for _ in range(size):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            self.input.append(torch.tensor([x, y]))
            self.output.append(torch.tensor(mandelbrot_calc(x, y, max_depth)))

    def __getitem__(self, idx):
      return self.input[idx], self.output[idx]

    def __len__(self):
        return len(self.input)