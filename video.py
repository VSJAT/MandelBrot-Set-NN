#Functions for creating Image

import torch
from mandelbrot import *

def renderModel(model, resx, resy, xmin=-2.4, xmax=1, yoffset=0, linspace=None, max_gpu=False):

    with torch.no_grad():
        if linspace is None:
            linspace = generateLinspace(resx, resy, xmin, xmax, yoffset)
        
        if not max_gpu:
            # slices each row of the image into batches to be fed into the nn.
            im_slices = []
            for points in linspace:
                
                im_slices.append(model(points))
            im = torch.stack(im_slices, 0)
        else:
            # otherwise cram the entire image in one batch
            if linspace.shape != (resx*resy, 2):
                linspace = torch.reshape(linspace, (resx*resy, 2))
            im = model(linspace).squeeze()
            im = torch.reshape(im, (resy, resx))


        im = torch.clamp(im, 0, 1) # doesn't add weird pure white artifacts
        linspace = linspace.cpu()
        return im.squeeze().cpu().numpy()


def generateLinspace(resx, resy, xmin=-2.4, xmax=1, yoffset=0):
    iteration = (xmax-xmin)/resx
    X = torch.arange(xmin, xmax, iteration)
    y_max = iteration * resy/2
    Y = torch.arange(-y_max-yoffset,  y_max-yoffset, iteration)
    linspace = []
    for y in Y:
        ys = torch.ones(len(X)) * y
        points = torch.stack([X, ys], 1)
        linspace.append(points)
    return torch.stack(linspace, 0)
