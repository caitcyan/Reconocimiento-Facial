import cv2
import os
import imutils
from PIL import Image

personName = 'Giovanna'
dataPath = '../Data' 
personPath = dataPath + '/' + personName

# Obtener la lista de archivos en la carpeta de videos
archivos_videos = os.listdir(dataPath +'/CENTRO EDUCATIVO/')
archivos_mov = [archivo for archivo in archivos_videos if archivo.endswith(".MOV")]

for archivo in archivos_mov:
    print(archivos_mov)