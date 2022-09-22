import argparse
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.utils.data import DataLoader


from network import DeepMLP
from mandelbrot import MandelbrotDataSet
from video import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MandelBrot Training")
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--hiddensize", type=int, default=50)
    parser.add_argument("--learningrate", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--image", type=bool, default=False)
    parser.add_argument("--resx", type=int, default=1920)
    parser.add_argument("--resy", type=int, default=1080)
    args, _ = parser.parse_known_args()

    if args.train == True:
        dataset = MandelbrotDataSet(1000000)

        BATCH_SIZE = 128

        data_loader_train = DataLoader(dataset, BATCH_SIZE, shuffle=True)
        data_loader_test = DataLoader(dataset, BATCH_SIZE, shuffle=False)
        
        mandel_model = DeepMLP(2, args.hiddensize, args.layers, 1, args.learningrate, args.optimizer, data_loader_train, data_loader_test, data_loader_test)

        # Train model for 20 epochs
        mandelbrot_trainer = Trainer(
            accelerator = "auto",
            devices = 1 if torch.cuda.is_available() else None,  # limiting got iPython runs
            max_epochs= args.epochs,
            callbacks = [TQDMProgressBar(refresh_rate=20)]
        )
        mandelbrot_trainer.fit(mandel_model)

        torch.save(mandel_model, 'best.pt')

    elif args.image == True:
        mandel_model = torch.load('best.pt')
        
        if not os.path.exists('./captures/images/'):
            os.makedirs('./captures/images/')
        plt.imsave("./captures/images/render.png", renderModel(mandel_model, args.resx, args.resy), vmin=0, vmax=1, cmap='plasma')
    