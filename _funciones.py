# Librerias necesarias
import numpy as np
from os import listdir
from os.path import join
import cv2
from tqdm import tqdm
import keras_tuner as kt
import tensorflow as tf

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



# Funcion para afinar hiperparámetros de la red neuronal convolucional
hp = kt.HyperParameters()  # Objeto para definir espacio de búsqueda

def build_model(hp):
    # Hiperparámetros a optimizar
    dropout_rate = hp.Float('DO', 0.05, 0.2, step=0.05) # Tasa de dropout: 5% a 20%
    reg_strength = hp.Float("rs", 0.0001, 0.0005, step=0.0001) # Fuerza de regularización L2
    optimizer = hp.Choice('optimizer', ['adam', 'sgd']) # Optimizador a usar

    hp_units_1 = hp.Int('units_1', 32, 256, step=16) # Numero de neuronas en la capa densa
    hp_activation_1 = hp.Choice('activation_1', ['relu', 'tanh']) # Función de activación

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=x_train.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(reg_strength)), # Capa convolucional
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(reg_strength)), # Capa convolucional
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hp_units_1, activation=hp_activation_1, kernel_regularizer=tf.keras.regularizers.l2(reg_strength)), # Capa convolucional
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid') # Capa de salida para clasificación binaria
    ])

    # Selección del optimizador
    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=0.001)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.001)


    # Compilación del modelo con función de pérdida y métrica    
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["Recall"])

    return model # Retorna el modelo construido con los hiperparámetros definidos
