# Librerias necesarias
import numpy as np
from os import listdir
from os.path import join
import cv2
from tqdm import tqdm


### Funcion para cargar imágenes y etiquetas
def load_images(path, width=120):
    rawImgs = []  # lista para almacenar imágenes cargadas
    labels = []   # lista para almacenar etiquetas (1 para benign, 0 para malignant)

    files_list = listdir(path)  # obtener lista de archivos en el directorio
    for item in tqdm(files_list):  # iterar con barra de progreso
        file = join(path, item)  # construir ruta completa del archivo
        if file[-1] == 'g':  # verificar que sea un archivo de imagen (termina en 'g', e.g. jpg, png)
            img = cv2.imread(file)  # leer imagen
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convertir BGR a RGB
            img = cv2.resize(img, (width, width))  # redimensionar imagen
            rawImgs.append(img)  # agregar imagen a la lista
            l = path.split('/')[-1]  # obtener nombre de carpeta para etiqueta
            if l == 'benign':  # si es benign, etiqueta 1
                labels.append([1])
            else:  # si no, etiqueta 0
                labels.append([0])
    return rawImgs, labels, files_list  # retornar imágenes, etiquetas y lista de archivos

