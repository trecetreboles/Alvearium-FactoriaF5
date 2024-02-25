import streamlit as st
from google.cloud import texttospeech
from google.cloud import speech_v1 as speech

# Función para convertir texto a voz (TTS)
def text_to_speech(text):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="es-ES", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content

# Función para convertir voz a texto (STT)
def speech_to_text(audio_content):
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="es-ES",
    )
    response = client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript
    else:
        return "No se pudo transcribir el audio"

# Título de la interfaz
st.title("Prueba de Conexión a Google Cloud")

# Botón para probar la conexión de TTS
if st.button("Probar TTS (Texto a Voz)"):
    # Texto de prueba
    test_text = "Hola, esto es una prueba de conexión a Google Cloud Text-to-Speech."

    # Intentar convertir texto a voz
    try:
        audio_content = text_to_speech(test_text)
        st.audio(audio_content, format="audio/wav")
        st.success("La conexión a Google Cloud Text-to-Speech fue exitosa.")
    except Exception as e:
        st.error(f"Error al conectar a Google Cloud Text-to-Speech: {e}")

# Botón para probar la conexión de STT
if st.button("Probar STT (Voz a Texto)"):
    # Archivo de audio de prueba (reemplace con su propio archivo de audio si es necesario)
    test_audio_file = "audio_sample.wav"

    # Intentar convertir voz a texto
    try:
        with open(test_audio_file, "rb") as audio_file:
            audio_content = audio_file.read()
        transcript = speech_to_text(audio_content)
        st.write("Texto transcrito:", transcript)
        st.success("La conexión a Google Cloud Speech-to-Text fue exitosa.")
    except Exception as e:
        st.error(f"Error al conectar a Google Cloud Speech-to-Text: {e}")
