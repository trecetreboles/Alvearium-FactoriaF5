import os
import time
import requests

# Definir la URL del servidor
url_servidor = "http://localhost:8000"

# Definir la ruta del archivo de audio MP3
ruta_archivo_audio_mp3 = r"C:\Users\ferna\OneDrive\Escritorio\GitHub\Alvearium-VR\ChatBot\audio_files\Record (online-voice-recorder.com).mp3"

try:
    # Realizar la solicitud al servidor para convertir el MP3 a WAV
    response_mp3_to_wav = requests.post(f"{url_servidor}/ask_audio", files={"file": open(ruta_archivo_audio_mp3, "rb")})
    response_mp3_to_wav.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    # Esperar un tiempo para que el servidor tenga tiempo de crear el archivo WAV
    time.sleep(1)  # Ajusta el tiempo de espera según sea necesario

    # Obtener la ruta del archivo WAV generado por el servidor
    response_data = response_mp3_to_wav.json()
    if "wav_file_path" not in response_data:
        print("El servidor no devolvió la ruta del archivo WAV.")
        print("Respuesta del servidor:", response_data)
        exit()

    ruta_archivo_audio_wav = response_data["wav_file_path"]

    # Realizar la solicitud al servidor para convertir el audio WAV a texto (STT)
    with open(ruta_archivo_audio_wav, "rb") as file:
        audio_data = {"file": file}
        response_stt = requests.post(f"{url_servidor}/speech_to_text", files=audio_data)
    response_stt.raise_for_status()  # Lanzar una excepción en caso de error de solicitud

    # Obtener la transcripción de audio a texto
    transcription = response_stt.json().get("text")

    print(f"Transcripción de audio a texto exitosa: {transcription}")

except requests.exceptions.RequestException as e:
    print(f"Error en la solicitud: {e}")
except Exception as e:
    print(f"Error inesperado: {e}")