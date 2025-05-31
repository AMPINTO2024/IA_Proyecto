import os
import numpy as np
import tensorflow as tf
import tqdm
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image  # <- esta es la forma correcta
from tensorflow.keras.models import Model
from tqdm import tqdm

# Ruta a la carpeta donde tienes las imágenes
carpeta_imagenes = 'imagenes_emociones'
# Cargar modelo VGG16 con capa 'fc2' para extraer vector 4096-d
base_model = VGG16(weights='imagenet', include_top=True)
model_fc2 = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
def extraer_vector(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # tamaño esperado
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # preprocesar para VGG16
    features = model_fc2.predict(x)
    return features.flatten()  # vector 4096
# Listas para guardar vectores y nombres de archivo
vectores = []
nombres = []



# Recorrer las imágenes y extraer vectores

for carpeta_emocion in os.listdir(carpeta_imagenes):
    subcarpeta = os.path.join(carpeta_imagenes, carpeta_emocion)

    if os.path.isdir(subcarpeta):
        for archivo in os.listdir(subcarpeta):
            ruta_imagen = os.path.join(subcarpeta, archivo)
            if archivo.lower().endswith(('.jpg', '.png', '.jpeg')):
                try:
                    vec = extraer_vector(ruta_imagen)
                    vectores.append(vec)
                    nombres.append(f"{carpeta_emocion}/{archivo}")  # guardas la etiqueta también si quieres
                except Exception as e:
                    print(f"Error procesando {archivo}: {e}")


# Convertir a numpy arrays
vectores = np.array(vectores)

# Guardar en archivo .npz para luego cargar fácil
np.savez('vectores_imagenes_emociones.npz', vectores=vectores, nombres=nombres)

print(f"Guardados {len(vectores)} vectores de imágenes.")










import numpy as np
from collections import Counter

# Cargar el archivo .npz
data = np.load('vectores_imagenes_emociones.npz')
nombres = data['nombres']

# Extraer etiquetas desde el nombre (antes del '/')
etiquetas = [nombre.split('/')[0] for nombre in nombres]

# Contar frecuencia por etiqueta
conteo = Counter(etiquetas)

# Mostrar resultados
print("Cantidad de imágenes por etiqueta:")
for etiqueta, cantidad in conteo.items():
    print(f"- {etiqueta}: {cantidad}")
