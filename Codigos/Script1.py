import cv2
import os
import imutils
from PIL import Image
import time
import numpy as np

dataPath = '../Data' 

# Obtener la lista de archivos en la carpeta de videos
archivos_videos = os.listdir(dataPath +'/CENTRO EDUCATIVO/PRE')

archivos_mov = [os.path.splitext(archivo)[0] for archivo in archivos_videos if archivo.endswith(".MOV")]


for archivo in archivos_mov:
	
	personPath = dataPath + '/Videos Procesados/' + archivo
	
	if not os.path.exists(personPath):
		print('Carpeta creada: ',personPath)
		os.makedirs(personPath)

	path_video = os.path.join(dataPath, 'CENTRO EDUCATIVO/PRE', archivo + '.MOV')
	cap = cv2.VideoCapture(path_video)

	#faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'lbpcascade_frontalface.xml')
	faceClassif =cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
	count = 0

	while True:

		ret, frame = cap.read()
		frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
		if ret == False: break
		frame =  imutils.resize(frame, width=640)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = frame.copy()

		faces = faceClassif.detectMultiScale(gray,1.3,5)

		for (x,y,w,h) in faces:
			cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
			cv2.imwrite(personPath + '/imagen_{}.jpg'.format(count),rostro)
			count = count + 1
		cv2.imshow('frame',frame)

		k =  cv2.waitKey(1)
		if k == 27 or count >= 150:
			break

	cap.release()
	cv2.destroyAllWindows()

####ENTRENAMIENTO
	
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
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
		image = cv2.imread(personPath+'/'+fileName,0)
		cv2.imshow('image',image)
		cv2.waitKey(10)
	label = label + 1

print('labels= ',labels) #etiquetado

cv2.destroyAllWindows()

#Entrenamiento:

#metodos = ['EigenFaces','FisherFace','BPHFace']
metodos = ['FisherFace']
inicio_tiempo = time.time()
# Métodos:
# Guardamos el modelo
for metodo in metodos:
	if metodo == 'EigenFaces':
		face_recognizer = cv2.face.EigenFaceRecognizer_create()
		print("Entrenando...")
		face_recognizer.train(facesData, np.array(labels))
		face_recognizer.write('EigenFace.xml')
		print("Modelo almacenado...")
	elif metodo == 'FisherFace' :
		face_recognizer = cv2.face.FisherFaceRecognizer_create()
		print("Entrenando...")
		face_recognizer.train(facesData, np.array(labels))
		face_recognizer.write('FisherFace.xml')
		print("Modelo almacenado...")
	elif metodo  == 'BPHFace':
		face_recognizer = cv2.face.LBPHFaceRecognizer_create()
		print("Entrenando...")
		face_recognizer.train(facesData, np.array(labels))
		face_recognizer.write('BPHFace.xml')
		print("Modelo almacenado...")

fin_tiempo = time.time()
tiempo_entrenamiento = fin_tiempo - inicio_tiempo
print("Tiempo transcurrido:", tiempo_entrenamiento, "segundos")
