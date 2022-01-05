import numpy as np
import matplotlib.pyplot as plt

def showNumpyAsImg(array):
    """
    Visualizes a single channel numpy array as a matplotlib plot.
    
    Keyword Arguments:
    array: A 2D numpy array representing an image.
    
    Returns None
    """
    array = np.transpose(array, (1,2,0)) # Converting 3 x 224 x 224 img to 224 x 224 x 3 img
    
    figure = plt.figure
    plt.imshow(array)
    plt.show()