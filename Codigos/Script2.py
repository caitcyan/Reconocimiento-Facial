import cv2
import os

dataPath = '../Data' 

archivos_intrusos = os.listdir(dataPath +'/intrusos')  # usa esta seccion para reconocer a los intrusos 
archivos_mov = [os.path.splitext(archivo)[0] for archivo in archivos_intrusos if archivo.endswith(".MOV")]
''' 
archivos_videos = os.listdir(dataPath +'/CENTRO EDUCATIVO/POST')  # usa esta seccion para reconocer post o pre
archivos_mov = [os.path.splitext(archivo)[0] for archivo in archivos_videos if archivo.endswith(".MOV")]
''' 
archivos_nombre = os.listdir(dataPath +'/CENTRO EDUCATIVO/POST')
archivos_nombre = [os.path.splitext(archivo)[0] for archivo in archivos_nombre if archivo.endswith(".MOV")]

print('imagePaths=',archivos_mov)

face_recognizer = cv2.face.FisherFaceRecognizer_create()

# Leyendo el modelo
face_recognizer.read('modelos_entrenados/FisherFace2daiteracion2.xml')
proms = []
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

for archivo in archivos_mov:
	#path_video = os.path.join(dataPath, 'CENTRO EDUCATIVO/PRE', archivo + '.MOV') #los archivos de pre
	#path_video = os.path.join(dataPath, 'CENTRO EDUCATIVO/Post', archivo + '.MOV') #los archivos de post
	path_video = os.path.join(dataPath, 'intrusos', archivo + '.MOV') #los archivos de intrusos
	cap = cv2.VideoCapture(path_video)
	faceClassif = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
	resultados = []
	while True:
		ret,frame = cap.read()
		frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
		if ret == False: break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		auxFrame = gray.copy()

		faces = faceClassif.detectMultiScale(gray,1.3,5)
		for (x,y,w,h) in faces:
			rostro = auxFrame[y:y+h,x:x+w]
			rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
			result = face_recognizer.predict(rostro)
			resultados.append(result[1])
			#print(resultados)
			cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
			# FisherFace
			if result[1] < 753: #753 es el umbral
				cv2.putText(frame,'{}'.format(archivos_nombre[result[0]]),(x,y-25),2,1.1,(255,255,0),1,cv2.LINE_AA)
				#cv2.putText(frame,'test',(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
				#cv2.putText(frame,'Etiqueta no valida',(x,y-25),2,1.1,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(255,255,0),2)
			else:
				cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
				cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
			
		cv2.imshow('frame',frame)
		k = cv2.waitKey(1)
		if k == 27:
			break
	print(sum(resultados) / len(resultados))
	#print(resultados)
	proms.append(sum(resultados) / len(resultados))
	cap.release()
	cv2.destroyAllWindows()

print(proms)