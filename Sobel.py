import detection
import numpy as np
import matplotlib.image as mplimg
import cv2
import matplotlib.pyplot as plt
import math
def convolutions(img, kernel):
    h, w = img.shape
    
    taileK, _ = kernel.shape
    img_pad = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            somme = 0
            for ki in range(-taileK//2, taileK//2):
                for kj in range(-taileK//2, taileK//2):
                    if 0 < i+ki < h and 0 <j+kj < w:
                        somme += img[i+ki][j+kj]*kernel[ki+taileK//2][kj+taileK//2]
            img_pad[i][j] = somme
    return img_pad

def Sobel(img):
    img_horizontal = convolutions(img, np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]))
    img_vertical = convolutions(img, np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))
    h, w = img.shape
    img_sobel = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            img_sobel[i][j] = math.sqrt(img_horizontal[i][j]**2 + img_vertical[i][j]**2)
    return img_sobel

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)


"""kernel_gaussian = np.array([[0, 0, 1, 2, 1, 0, 0],
                           [0, 3, 13, 22, 13, 3, 0],
                           [1, 13, 59, 97, 59, 13, 1],
                           [2, 22, 97, 159, 97, 22, 2],
                           [1, 13, 59, 97, 59, 13, 1],
                           [0, 3, 13, 22, 13, 3, 0],
                           [0, 0, 1, 2, 1, 0, 0]])
"""
def seuilGray(img, seuil):
    h, w = img.shape
    img_seuil = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            if img[i][j] < seuil:
                img_seuil[i][j] = 0
            else:
                img_seuil[i][j] = 255
    return img_seuil

def find_threshold_kmeans(image):
    # Redimensionner l'image en un vecteur 1D
    data = image.reshape((-1, 1))
    data = np.float32(data)

    # Définir le nombre de clusters (seuils) que vous voulez obtenir
    num_clusters = 2

    # Appliquer l'algorithme K-Means pour regrouper les données en 'num_clusters' clusters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Trier les centres des clusters
    centers = np.sort(centers, axis=0)

    # Le seuil est la valeur du centre entre les deux clusters
    threshold_value = int((centers[0][0] + centers[1][0]) / 2)

    return threshold_value    

def chainDeTraitement(img):            
    kernel_gaussian = gaussian_kernel(15, 1.5)
    h, w, _ = img.shape
    h_min, w_min = h, w
    while h_min> 1000 or w_min>1000:
        h_min //= 2
        w_min //= 2
    img_min = cv2.resize(img, (w_min, h_min))
    img_gray1 = cv2.cvtColor(img_min, cv2.COLOR_BGR2GRAY)
    img_gaussian = convolutions(img_gray1, kernel_gaussian)
    imgSobel = Sobel(img_gaussian)
    imgSeuil = seuilGray(imgSobel, 30)
    img_max = cv2.resize(imgSeuil, (w, h))
    
    maxTaieux = h if h>w else w
    
    if img_max.dtype != np.uint8:  # Vérifier le type de données de l'image
        print("converti")
        img_max = img_max.astype(np.uint8)  # Convertir en CV_8UC1 si nécessaire

    img_copy = img.copy()

    circles=cv2.HoughCircles(img_max,cv2.HOUGH_GRADIENT,1, minDist = maxTaieux//10, param1=200,param2=60,minRadius=maxTaieux//15,maxRadius=maxTaieux//8)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            print("circullll")
            # draw the outer circle
            cv2.circle(img_copy, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(img_copy, (i[0], i[1]), 2, (0, 0, 255), 3)
    plt.imshow(imgSobel)
    plt.show() 
    plt.imshow(img_max)
    plt.show() 
    plt.imshow(img_copy)
    plt.show()   

for i in range(0, 286):
    print("charge",i)
    img = detection.load_image(i)
    if  img is not None and img.size != 0:
        chainDeTraitement(img)
                             
