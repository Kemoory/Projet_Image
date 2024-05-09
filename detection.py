import numpy as np
import cv2
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import os

def load_image(i):
    lien_image = f"data/Images/{i}"  # Fixing the string formatting
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

img = load_image(42)
if img is not None:
    print("Image loaded successfully.")
    img_copy=cv2.Canny(img,0,100)
    circles = cv2.HoughCircles(img_copy, cv2.HOUGH_GRADIENT, dp=1, minDist=500,param1=100, param2=30, minRadius=10, maxRadius=100)

    plt.imshow(circles,cmap='gray')
    plt.show()
else:
    print("No image loaded. Exiting.")


