import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import Sobel
import detection
import math
def read_annotations(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
        
    circles = []
    for shape in data['shapes']:
        if shape['shape_type'] == 'circle':
            #on prend deux points de la forme , le premier c'est le centre
            x1, y1 = shape['points'][0]
            x2, y2 = shape['points'][1]
            #on calcule la distance entre eux
            rayon = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            centre = (int(x1), int(y1))
            circles.append((centre, rayon))
    return circles



def calculate_iou(circle1, circle2):
    centre1, rayon1 = circle1
    centre2, rayon2 = circle2
    
    # Distance entre les centres des cercles
    d = np.sqrt((centre1[0] - centre2[0])**2 + (centre1[1] - centre2[1])**2)
    
    if d >= (rayon1 + rayon2):
        # Les cercles ne se chevauchent pas
        return 0.0
    
    if d <= abs(rayon1 - rayon2):
        # L'un des cercles est complètement à l'intérieur de l'autre
        intersection_area = np.pi * min(rayon1, rayon2)**2

    else:
        # Calcule de l'aire d'intersection

        #on calcule le carre des rayons
        r1_sq = rayon1**2
        r2_sq = rayon2**2

        #arc de cercle dans le premier cercle (cosinus)
        angle1 = np.arccos((r1_sq + d**2 - r2_sq) / (2 * rayon1 * d))

        #arc de cercle dans le deuxieme cercle (cosinus)
        angle2 = np.arccos((r2_sq + d**2 - r1_sq) / (2 * rayon2 * d))


        intersection_area = (
            r1_sq * angle1 + 
            r2_sq * angle2 - 
            0.5 * np.sqrt(
                (-d + rayon1 + rayon2) * 
                (d + rayon1 - rayon2) * 
                (d - rayon1 + rayon2) * 
                (d + rayon1 + rayon2)
            )    
        )
    union_area = np.pi * (rayon1**2 + rayon2**2) - intersection_area

    return intersection_area / union_area

def calculate_pixel(circle):
    centre, r = circle
    h, k = centre
    ensembre = set()
    for x in range(h - r, h + r + 1):
        for y in range(k - r, k + r + 1):
            if (x - h) ** 2 + (y - k) ** 2 <= r ** 2:
                ensembre.add((x,y))
    return ensembre




def evaluer_image (img, json_file):
    annotated_circles = read_annotations(json_file)
    detected_circles = Sobel.chainDeTraitement(img)

    #ON va faire la moyenne de toutes les iou pour chaque cercle
    iou_scores = []

    for detected_circle in detected_circles:
        best_iou = 0
        for annotated_circle in annotated_circles:
            iou = calculate_iou(annotated_circle, detected_circle)
            if iou > best_iou:
                best_iou = iou
        iou_scores.append(best_iou)
    
    
    moyenne = np.mean(iou_scores) if iou_scores else 0
    return moyenne

def evaluer_image_pixel (img, json_file):
    annotated_circles = read_annotations(json_file)
    detected_circles = Sobel.chainDeTraitement(img)
    
    annotated_set = set()
    detected_set = set()
    for detected_circle in detected_circles:
        detected_set |= calculate_pixel(detected_circle)
    for annotated_circle in annotated_circles:
        annotated_set |= calculate_pixel(annotated_circle)
    intersection_area = annotated_set & detected_set
    union_area = annotated_set | detected_set
    return float(len(intersection_area))/float(len(union_area)), len(annotated_circles), len(detected_circles)

def execution(nb_images):
    #on va stocker le num de l'image et son score
    scores=[]
    maes = []
    mses = []
    for i in range(0, nb_images):
        print("charge",i)
        img = detection.load_image(i)
        json_file=f"data/entrainement/json/{i}.json"

        if  img is not None and img.size != 0:
            score, nbAnnotated, nbDetected=evaluer_image_pixel(img,json_file)
            mae = abs(nbAnnotated-nbDetected)
            maes.append(mae)
            mse = math.pow(nbAnnotated-nbDetected, 2)
            mses.append(mse) 
            scores.append(score)
            with open("result.txt", 'a') as file:
                file.write(str(score)+", "+ str(mae) +", " + str(mse) +"\n")
            print(score)
    with open("result.txt", 'a') as file:
        file.write("mean"+ str(np.mean(scores))+", " + str(np.mean(maes))+ ", " + str(np.mean(mses)))



def draw_detected_circles():

    image=detection.load_image(0)
    detected_circles=Sobel.chainDeTraitement(image)
    # Afficher l'image
    plt.imshow(image)
    
    # Dessiner les cercles détectés
    for center, radius in detected_circles:
        circle = plt.Circle(center, radius, color='r', fill=False)
        plt.gca().add_patch(circle)
    
    # Fixer les limites des axes pour s'assurer que les cercles sont entièrement visibles
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    
    # Afficher l'image avec les cercles dessinés
    plt.show()

draw_detected_circles()


execution(173)
        

