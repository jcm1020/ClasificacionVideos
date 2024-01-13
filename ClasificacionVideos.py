# Importar las librerías necesarias
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions 
from matplotlib import pyplot as plt
import ultralytics
import subprocess
import os
from ultralytics import YOLO
import cv2
import glob

ultralytics.checks()

# Mostramos el directorio de trabajo actual
print("Directorio actual:")
print(os.getcwd())
# Cambiamos el directorio de trabajo a '/data'
os.chdir("./data/")
print('Directorio de trabajo:')
print(os.getcwd())

# Ruta del video de entrada
video_in = 'parque.mp4' 

# Ruta del video de salida 
video_out = 'video_processed.mp4'

# Directorio donde guardar frames extraídos. No cambiar.
frames_dir = 'frames/'

#El directorio frames almacena cada frame que luego sera procesado y clasificado
os.chdir("./frames/")
for f in os.listdir(os.getcwd()):
    os.remove(os.path.join(os.getcwd(), f))

#Extraccion de frames
os.chdir("./..")
# Extraer frames del video de entrada
video = cv2.VideoCapture(video_in)
count = 0
success = True
while success:
  success, frame = video.read()
  if success:
    cv2.imwrite(frames_dir + 'frame%s.jpg'%str(count).zfill(4), frame)
    count += 1
video.release()



# Cargar el modelo MobileNetV2 pre-entrenado 
model_mobilenetv2 = MobileNetV2(weights='imagenet')
model_yolo = YOLO('yolov8n.pt') # Cargar el modelo para segmentacion




os.chdir("./frames/")
# Procesar cada frame almacenado
for file in os.listdir(os.getcwd()):

    # Cargar la imagen y preprocesarla. Para primeras pruebas
    #imagen='imagen4.jpg'
    imagen=file
    img = load_img(imagen, target_size=(224, 224)) 

    #plt.imshow(img) #No activar si existen muchos frames
    #plt.show()

    #Adecuacion de la imagen para el formato que admite MobileNetv2
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) 

    # Predecir la clase de la imagen
    preds = model_mobilenetv2.predict(x)
    predicted_class = decode_predictions(preds)
    print('Clase predicha con MobileNETV2:', predicted_class)
     
    #Leer imagen de nuevo
    img = cv2.imread(imagen)
    height, width, layers = img.shape
 
    #Características del texto
    texto = "MobileNETV2:"+str(predicted_class[0][0][1])
    ubicacion = (10,50)
    font = cv2.FONT_HERSHEY_TRIPLEX
    tamañoLetra = 1.8
    colorLetra = (150,0,0)
    grosorLetra = 2
 
    #Escribir texto. DOs textos en cabez y pie nos indican resultado de clasificacion
    cv2.putText(img, texto, ubicacion, font, tamañoLetra, colorLetra, grosorLetra)
    cv2.putText(img, texto, (10,height-70), font, tamañoLetra, colorLetra, grosorLetra)
    imagen=imagen.replace('imagen','imagen_procesada')
    #Guardar imagen
    cv2.imwrite(imagen, img)
 
    #Mostrar imagen procesada. No activar si existen muchos frames.
    #cv2.imshow('imagen',img)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()


# Unir frames procesados en un nuevo video 
frame_array = []
files = glob.glob('*.jpg')
print("Cantidad de frames: {0}".format(len(files)))
indice=1
for file in files:
  if file.title=='video_processed.mp4':continue
  img = cv2.imread(file)  
  frame_array.append(img)
  print("Creando video: frame{0}".format(indice))
  indice+=indice

height, width, layers = img.shape
size = (width,height)
#out = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc(*'MPEG'), 75, size)
out = cv2.VideoWriter(video_out,cv2.VideoWriter_fourcc(*'mp4v'), 90, size)

for i in range(len(frame_array)):
  out.write(frame_array[i])
out.release()

# Ejecutar inference en un video con YOLOv8n
instruccion="yolo predict model=yolov8n.pt source={0}".format(video_out)
#instruccion="yolo predict model=sam_b.pt source={0}".format(video_out)


# Llamada externa para creacion final del video con segmentacion
subprocess.call(instruccion, shell=True)

#Pulsar intro para terminar
print("Gracias por usar nuestro programa.")
inputs = input()
