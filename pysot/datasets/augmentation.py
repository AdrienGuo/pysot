import cv2
import numpy as np


class Augmentation:
    def __init__(self):
        pass
    
    def _gray(self, img):
        grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return img

    def _contrast(self, img):
        alpha = 192
        img = alpha * (img / alpha) ** 0.5
        return img
