import cv2
import os
import imutils
from PIL import Image
import time
import numpy as np
#from tqdm import tqdm

dataPath = '../Data' 

# Obtener la lista de archivos en la carpeta de videos
archivos_videos = os.listdir(dataPath + '/CENTRO EDUCATIVO/post')
archivos_mov = [os.path.splitext(archivo)[0] for archivo in archivos_videos if archivo.endswith(".MOV")]

for archivo in archivos_mov:
    personPath = dataPath + '/Videos Procesados/' + archivo
    
    if not os.path.exists(personPath):
        print('Carpeta creada: ', personPath)
        os.makedirs(personPath)

    path_video = os.path.join(dataPath, 'CENTRO EDUCATIVO/post', archivo + '.MOV')
    cap = cv2.VideoCapture(path_video)

    faceClassif = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    count = 100 ##ahi anado mas imagenes

    while True:
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if ret == False:
            break
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        auxFrame = frame.copy()

        faces = faceClassif.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            rostro = auxFrame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(personPath + '/imagen_{}.jpg'.format(count), rostro)
            count += 1
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27 or count >= 150:   #50 mas
            break

    cap.release()
    cv2.destroyAllWindows()
