import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.image as mplimg

def kMeans(img_name, nb_seuil):
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
    img = cv2.resize(img, (640, 800))
    h, w = img.shape

    # Utilisation de KMeans de scikit-learn pour initialiser les centres de cluster de manière plus efficace
    kmeans = KMeans(n_clusters=nb_seuil, init='k-means++', random_state=15)
    flattened_img = img.reshape((-1, 1))  # Redimensionner pour KMeans
    kmeans.fit(flattened_img)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # Reconstruction de l'image à partir des clusters
    img_eq = (centers[labels].reshape((h, w))).astype(np.uint8)

    return img_eq

lien_image = "data/Images/182.jpg"
img = cv2.imread(lien_image)
img = cv2.resize(img , (640 , 800))
image_copy = img.copy()
img_eq = kMeans(lien_image, 2)  # Utilisez kMeans pour obtenir une image égalisée

# Vous pouvez choisir un seuil basé sur la valeur moyenne ou médiane de l'image égalisée
seuil = np.mean(img_eq)  # Par exemple, utilisez la valeur moyenne
print("Seuil:", seuil)

ret , thresh = cv2.threshold(img_eq , seuil, 255 , cv2.THRESH_BINARY)


contours , _ = cv2.findContours(thresh , cv2.RETR_TREE , cv2.CHAIN_APPROX_NONE)
area = {}
for i in range(len(contours)):
    cnt = contours[i]
    ar = cv2.contourArea(cnt)
    area[i] = ar
srt = sorted(area.items() , key = lambda x : x[1] , reverse = True)
results = np.array(srt).astype("int")
num = np.argwhere(results[: , 1] > 500).shape[0]

for i in range(1 , num):
    image_copy = cv2.drawContours(image_copy , contours , results[i , 0] ,
                                  (0 , 255 , 0) , 3)
print("Number of coins is " , num - 1)
cv2.imshow("Objects Detected", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()