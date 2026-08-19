import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Identificación de imagen", layout="centered")

# Título y Subtítulo
st.title("Identificación de imagen")
st.subheader("Nahun Alberto Flores Cruz")

# Definir nombres de clases en español
class_names = ['avión', 'auto', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# Cargar el modelo
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('modelo_cifar10.keras')

model = load_model()

# Cargador de archivos
uploaded_file = st.file_uploader("Elige una imagen...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Mostrar imagen
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida.", use_container_width=True)
    
    # Preprocesamiento
    img = image.resize((32, 32))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predicción
    if st.button('Clasificar'):
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx] * 100

        st.write(f"### Resultado: {class_names[class_idx]}")
        st.write(f"**Confianza:** {confidence:.2f}%")
