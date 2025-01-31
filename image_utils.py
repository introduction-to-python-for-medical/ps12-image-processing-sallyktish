from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    image = Image.open(file_path)
    image_array = np.array(image)
    return image_array

def edge_detection(image_array):
    grayscale_image = np.mean(image_array, axis=2)
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) # Define kernelX here
    
    from scipy.signal import convolve2d # Import convolve2d here if needed
    
    edgeX = convolve2d(grayscale_image, kernelX, mode='constant', cval=0) # Using kernelX
    edgeY = convolve2d(grayscale_image, kernelY, mode='constant', cval=0) # Using kernelY
    
    return np.sqrt(edgeX**2 + edgeY**2)

