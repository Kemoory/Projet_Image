import numpy as np
import cv2
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import os

def load_image(i):
    lien_image = f"..\\..\\Images/{i}"  # Fixing the string formatting
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

"""
if img is not None:
    print("Image loaded successfully.")
    #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img_copy = cv2.GaussianBlur(img_gray, (5, 5), 3, 4)
    #img_copy = cv2.Sobel(src=img_copy, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
    #img_copy=cv2.Canny(img,0,100)
    #circles = cv2.HoughCircles(img_copy, cv2.HOUGH_GRADIENT, dp=1, minDist=500,param1=100, param2=30, minRadius=10, maxRadius=100)

    plt.imshow(img,cmap='gray')
    plt.show()
else:
    print("No image loaded. Exiting.")

for i in range(1, 286):
    print("charge",i)
    img = load_image(i)
    """
