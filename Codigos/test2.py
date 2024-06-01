import cv2
import os
import imutils
from PIL import Image
import time
import numpy as np


#### ENTRENAMIENTO
  
dataPath2 = '../Data/Videos Procesados'
peopleList = os.listdir(dataPath2)
print('Lista de personas: ', peopleList)

labels = []
facesData = []
label = 0

for nameDir in peopleList:
    personPath = dataPath2 + '/' + nameDir
    print('Leyendo las imágenes')

    for fileName in os.listdir(personPath):
        print('Rostros: ', nameDir + '/' + fileName)
        labels.append(label)
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))
        image = cv2.imread(personPath + '/' + fileName, 0)
        cv2.imshow('image', image)
        cv2.waitKey(10)
    label += 1

print('labels= ', labels)  # etiquetado

cv2.destroyAllWindows()

# Entrenamiento:

metodos = ['FisherFace']
inicio_tiempo = time.time()

for metodo in metodos:
    if metodo == 'EigenFaces':
        face_recognizer = cv2.face.EigenFaceRecognizer_create()
    elif metodo == 'FisherFace':
        face_recognizer = cv2.face.FisherFaceRecognizer_create()
    elif metodo == 'BPHFace':
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    print(f"Entrenando con el método {metodo}...")
    
    # Añadir barra de progreso
    #with tqdm(total=len(facesData), desc=f"Entrenando {metodo}") as pbar:
    #  for i in range(len(facesData)):
    face_recognizer.train(facesData, np.array(labels))
      #      pbar.update(1)
    
    face_recognizer.write(f'{metodo}.xml')
    print(f"Modelo {metodo} almacenado...")

fin_tiempo = time.time()
tiempo_entrenamiento = fin_tiempo - inicio_tiempo
print("Tiempo transcurrido:", tiempo_entrenamiento, "segundos")
