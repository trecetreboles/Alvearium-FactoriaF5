import os
import time
import requests
from google.cloud import storage

# Definir la URL del servidor
URL_SERVIDOR = "http://localhost:8000"

# Nombre del bucket de Google Cloud Storage
NOMBRE_BUCKET = "alvea_example"

# Inicializar el cliente de Google Cloud Storage
cliente_storage = storage.Client()

# Función para subir un archivo a Google Cloud Storage
def upload_to_gcs(file_path, gcs_file_name):
    bucket = cliente_storage.bucket(NOMBRE_BUCKET)
    blob = bucket.blob(gcs_file_name)
    blob.upload_from_filename(file_path)

# Función para descargar un archivo desde Google Cloud Storage
def download_from_gcs(gcs_file_name, local_file_path):
    bucket = cliente_storage.bucket(NOMBRE_BUCKET)
    blob = bucket.blob(gcs_file_name)
    blob.download_to_filename(local_file_path)

try:
    # Eliminar el archivo WAV existente, si lo hay
    if os.path.exists(RUTA_ARCHIVO_AUDIO_WAV):
        os.remove(RUTA_ARCHIVO_AUDIO_WAV)

    # Subir el archivo MP3 original a Google Cloud Storage
    upload_to_gcs(RUTA_ARCHIVO_AUDIO_MP3_ORIGINAL, "audio_files/original.mp3")

    # Realizar la solicitud al servidor para convertir el MP3 a WAV
    response_mp3_to_wav = requests.post(f"{URL_SERVIDOR}/ask_audio", files={"file": open(RUTA_ARCHIVO_AUDIO_MP3_ORIGINAL, "rb")})
    response_mp3_to_wav.raise_for_status()

    # Realizar la solicitud al servidor para obtener la respuesta del chatbot
    response_answer = requests.post(f"{URL_SERVIDOR}/answer", json={"chat_history": [], "question": "What's your question?"})
    response_answer.raise_for_status()

    # Obtener la respuesta del chatbot
    answer = response_answer.content

    # Guardar la respuesta del chatbot en un archivo de audio MP3 en Google Cloud Storage
    with open(RUTA_ARCHIVO_AUDIO_MP3_RESPUESTA, "wb") as audio_file:
        audio_file.write(answer)

    upload_to_gcs(RUTA_ARCHIVO_AUDIO_MP3_RESPUESTA, "audio_files/respuesta.mp3")

    print("Respuesta del chatbot generada correctamente")

except requests.exceptions.RequestException as e:
    print(f"Error en la solicitud: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")
