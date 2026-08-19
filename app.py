import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

# Título y subtítulo de la aplicación
st.title("Identificación de imagen")
st.write("### Nahun Alberto Flores Cruz")

# Cargar el modelo entrenado
# Asegúrate de que 'cifar10_image_classifier.keras' esté en el mismo directorio que app.py
@st.cache_resource
def load_my_model():
    model = load_model('cifar10_image_classifier.keras')
    return model

model = load_my_model()

# Definir los nombres de las clases de CIFAR-10
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
]

# Cargador de archivos
uploaded_file = st.file_uploader("Sube una imagen para identificación", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mostrar la imagen subida
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen subida', use_column_width=True)
    st.write("")
    st.write("Clasificando...")

    # Preprocesar la imagen para el modelo
    image_resized = image.resize((32, 32)) # CIFAR-10 images are 32x32
    img_array = np.array(image_resized).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Añadir dimensión de lote

    # Realizar la predicción
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index] * 100

    # Mostrar el resultado de la predicción
    st.success(f"Predicción: **{class_names[predicted_class_index]}**")
    st.info(f"Confianza: **{confidence:.2f}%**")
