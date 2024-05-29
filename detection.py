import numpy as np
import cv2
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import os

def load_image(i):
    lien_image = f"data/validation/{i}"  # Fixing the string formatting
    img = None
    if os.path.exists(lien_image + ".jpg"):
        img = mplimg.imread(lien_image + ".jpg")

    elif os.path.exists(lien_image + ".png"):
        img = mplimg.imread(lien_image + ".png")

    elif os.path.exists(lien_image + ".jpeg"):
        img = mplimg.imread(lien_image + ".jpeg")

    elif os.path.exists(lien_image + ".JPG"):
        img = mplimg.imread(lien_image + ".JPG")

    elif os.path.exists(lien_image + ".PNG"):
        img = mplimg.imread(lien_image + ".PNG")

    if img is None:
        print(f"Error: Image {lien_image} not found or couldn't be loaded.")

    return img


