from PIL import Image
import numpy as np


class image_classifier():
    def __init__(self, filepath):
        self.filepath = filepath
        self.image = None

    def load_image(self):
        img = Image.open(self.filepath)
        self.image = np.array(img)
        return self.image
    
    def grayscale (self): # independent on the self.image
        gray = 0.299 * self.image[:, :, 0] + 0.587 * self.image[:, :, 1] + 0.114 * self.image[:, :, 2] 
        return gray.astype(np.uint8)
    
    def invert (self): # independent on the self.image
        inverted = 255 -self.image
        return inverted.astype(np.unit8)
    
    def threshold(self, value):
        pass


 