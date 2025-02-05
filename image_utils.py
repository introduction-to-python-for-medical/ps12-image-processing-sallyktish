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

# פונקציה להערכת תוצאה (משתמשים בה לצורך חישוב נקודת האיכות מול תוצאה מצופה)
def evaluate_edge_detection(predicted_edge, true_edge):
    area = true_edge.shape[0] * true_edge.shape[1]
    score = np.sum(true_edge == predicted_edge) / area
    return score

# טעינת תמונת הקלט והפלט (ההגדרה היא בשביל מבחן)
image_path = '.tests/lena.jpg'  # נתיב לתמונה המקורית
true_edge_path = '.tests/lena_edges.png'  # נתיב לתמונה עם הקצוות המצופים

image = load_image(image_path)  # טוען את התמונה
edge = edge_detection(image)  # מבצע Edge Detection

# טוען את התמונה הצפויה של הקצוות
true_edge = load_image(true_edge_path)
true_edge_binary = np.mean(true_edge, axis=2) > 50  # מבצע סף גם כאן כדי להשוות

# הערכת התוצאה
score = evaluate_edge_detection(edge, true_edge_binary)

# הדפסת הציון
print(f"Edge detection accuracy score: {score}")

# הצגת התמונה עם הקצוות
plt.imshow(edge, cmap='gray')
plt.title("Edge Detection Result")
plt.show()

# ביצוע בדיקה אם הציון מספיק טוב (לפי סף 90%)
assert score > 0.9, f"Score is below the threshold: {score}"
