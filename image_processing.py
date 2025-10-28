from PIL import Image
import numpy as np
import cv2


class image_classifier():
    def __init__(self, filepath):
        self.filepath = filepath
        self.image = None

    def load_image(self):
        img = Image.open(self.filepath)
        self.image = np.array(img)
        return self.image
    
    def grayscale (self): # independent on the self.image
        """Converts the image into gray scale extracting specific channels"""
        gray = 0.299 * self.image[:, :, 0] + 0.587 * self.image[:, :, 1] + 0.114 * self.image[:, :, 2] 
        return gray.astype(np.uint8)
    
    def invert (self): # independent on the self.image
        """Converts the image into values"""
        inverted = 255 -self.image # get the counter value of the existing 
        return inverted.astype(np.uint8)
    
    def threshold(self, value):
        """Converts the pixels into black and white depending on the threshold values"""
        binary = np.where(self.image > value, 255, 0)
        return binary.astype(np.uint8)
    
    def apply_blur(self, kernel_size =3):
        blurred = cv2.blur(self.image, (kernel_size, kernel_size))
        return blurred.astype(np.uint8)
    
    def apply_sharpen(self):
        kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
        sharpened = cv2.filter2D(self.image, -1, kernel) # -1 is the convention of saying match the input with the output
        return sharpened.astype(np.uint8) # returns the value in the integer from any floats/decimals
        
    def save_image(self, filepath, image_array=None):
        if image_array is None:
            image_array = self.image
        img = Image.fromarray(image_array)
        img.save(filepath)
        
processor = image_classifier("TestImage.jpg")
processor.load_image()
processor.save_image('gray.jpg', processor.grayscale())