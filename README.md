# MandelBrot Set Neural Network
A Neural Network that can generate images of the Mandelbrot Set

The code for generating the image is taken from [Emergent Garden](https://github.com/MaxRobinsonTheGreat). I have mearly updated his code to use pytorch lightning. Check out his video on the topic.

https://www.youtube.com/watch?v=QmPBLroyHB0


## To Train the Network run

To run the code, open up a terminal in the same directory as the python files

```
python3 main.py --train True 
```
You can add the following additional parameters

### Additional Parameters
```
--epochs (Default=50)
--layers (Default=5)
--hiddensize (Default=50)
--learningrate (Default=1e-3)
--optimizer (Default=SGD)
```

For example
```
python3 main.py --train True --epochs 100 --layers 7 --hiddensize 50 --learningrate 0.01 --optimizer SGD
```

## Making an Image

You can call the following command to make an image from the network

```
python3 main.py --image True
```
### Additional Parameters
```
--resx (Default=1920)
--resy (Default=1080)
```

For example
```
python3 main.py --image True --resx 3840 --resy 2160
```

## Images

Here are some of the images I have generated using this
