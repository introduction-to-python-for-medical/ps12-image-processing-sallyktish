from image_utils import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
from PIL import Image
lena = load_image('lena.png')
clean_lena = median(lena, ball(3))
edge_lena = edge_detection(clean_lena)
edge_image = Image.fromarray(edge_lena)
edge_image.save('my_edges.png')
