from PIL import Image
import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt

# פונקציה לטעינת התמונה
def load_image(file_path):
    image = Image.open(file_path)  # טוען את התמונה
    image_array = np.array(image)  # ממיר אותה למערך numpy
    return image_array

# פונקציה לזיהוי קצוות (Edge Detection)
def edge_detection(image_array):
    # המרה לגריסקייל (אם מדובר בתמונה צבעונית)
    grayscale_image = np.mean(image_array, axis=2)
    
    # הגדרת פילטר Sobel לזיהוי קצוות בכיוון X ובכיוון Y
    kernelY = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])  # פילטר בגרסא של Sobel בכיוון Y
    kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # פילטר בגרסא של Sobel בכיוון X
    
    # החלת הפילטרים על התמונה (בהתאם לכיוונים X ו-Y)
    edgeX = convolve2d(grayscale_image, kernelX, mode='constant', cval=0)
    edgeY = convolve2d(grayscale_image, kernelY, mode='constant', cval=0)
    
    # חישוב magnitude של הקצוות, כדי לקבל עוצמת קצה כוללת
    edge_magnitude = np.sqrt(edgeX**2 + edgeY**2)
    
    # ביצוע סף (thresholding) - קובעים ערך סף (למשל 50) כדי להפוך את הקצוות לבינאריים
    edge_binary = edge_magnitude > 50  # הערך כאן הוא הסף (אפשר לשנות לפי הצורך)
    
    return edge_binary

