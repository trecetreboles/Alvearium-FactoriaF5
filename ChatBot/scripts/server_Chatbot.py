import os
import pyaudio
import wave
from typing import List, Tuple
from operator import itemgetter
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydub import AudioSegment
from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain_community.callbacks import get_openai_callback
from google.cloud import texttospeech_v1 as texttospeech
from google.cloud import speech_v1 as speech
from extract_apis_keys import load

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple API server using Langchlistain's Runnable interfaces",
)

UPLOAD_DIRECTORY = "./audio_files"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Configura las credenciales de autenticación de Google Cloud
GOOGLE_APPLICATION_CREDENTIALS = load()

# Carga de la clave de la API OpenAI
OPENAI_API_KEY = load()[1]
openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# Plantillas de conversación y respuesta
_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

# Función para combinar documentos
def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

# Función para formatear el historial del chat
def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

# Carga del índice de vectores
index_directory = "./faiss_index"
persisted_vectorstore = FAISS.load_local(index_directory, openai_embeddings)
retriever = persisted_vectorstore.as_retriever()

# Definición del mapeo de entrada y contexto
_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}

# Definición del modelo de entrada del historial de chat
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str

# Cadena de procesamiento de la conversación
conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI() | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

# Variable global para almacenar el historial del chat
global_chat_history = []

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
def speech_to_text(file_path: str):
    client = speech.SpeechClient()
    with open(file_path, "rb") as audio_file:
        content = audio_file.read()
    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="es-ES",
    )
    response = client.recognize(config=config, audio=audio)
    # Verificar si hay resultados antes de acceder al primer elemento
    if response.results:
        return response.results[0].alternatives[0].transcript
    else:
        # Manejar el caso donde no hay resultados
        return "No se pudo transcribir el audio"

# Esta es la función que manejará las solicitudes de los usuarios en formato de audio
@app.post("/ask_audio")
async def ask_question_audio(file: UploadFile = File(...)):
    # Guardamos el archivo de audio enviado por el usuario
    file_path_mp3 = os.path.join(UPLOAD_DIRECTORY, file.filename)
    file_path_wav = os.path.splitext(file_path_mp3)[0] + ".wav"
    
    # Normalizar la ruta del archivo
    file_path_mp3 = os.path.normpath(file_path_mp3)
    file_path_wav = os.path.normpath(file_path_wav)
    
    # Imprime la ruta de archivo para depurar
    print("MP3 file path:", file_path_mp3)
    print("WAV file path:", file_path_wav)
    
    with open(file_path_mp3, "wb") as audio_file:
        audio_file.write(await file.read())
    
    # Convertir el archivo MP3 a LINEAR16
    convert_mp3_to_linear16(file_path_mp3, file_path_wav)
    
    # Intentamos convertir el audio a texto utilizando la función speech_to_text
    try:
        question = speech_to_text(file_path_wav)
    except Exception as e:
        # Si ocurre un error al transcribir el audio, devolvemos un mensaje de error
        return {"error": "No se pudo transcribir el audio"}
    
    # Aquí es donde enviarías la pregunta al bot y obtendrías una respuesta
    # Por ahora, solo devolvemos un mensaje de prueba
    answer = "¡Hola! Has preguntado: " + question
    
    # Convertir la respuesta del asistente a audio utilizando la función text_to_speech
    audio_content = text_to_speech(answer)
    
    # Enviar respuesta en formato de audio
    return FileResponse(audio_content, media_type="audio/mp3")

# Función para convertir archivos MP3 a LINEAR16
def convert_mp3_to_linear16(mp3_file_path: str, wav_file_path: str):
    # Cargar el archivo MP3
    audio = AudioSegment.from_mp3(mp3_file_path)
    
    # Convertir a formato LINEAR16 (16-bit PCM)
    audio = audio.set_frame_rate(16000).set_sample_width(2).set_channels(1)
    
    # Guardar el archivo convertido
    audio.export(wav_file_path, format="wav")

# Define la ruta y la función controladora para manejar las solicitudes POST
@app.post("/answer")
async def get_answer(request_body: dict):
    global global_chat_history
    
    # Extraer la pregunta del cuerpo de la solicitud
    question = request_body.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="Pregunta no proporcionada en el cuerpo de la solicitud.")
    
    # Extraer el historial de chat del cuerpo de la solicitud, o usar una lista vacía si no está presente
    chat_history = request_body.get("chat_history", [])
    
    # Llama a tu lógica existente para obtener la respuesta
    with get_openai_callback() as cb:
        respuesta = chain.invoke({"chat_history": chat_history, "question": question})
        print(cb)
        # Si ocurrió algún error al obtener la respuesta, lanza una excepción HTTP
        if not respuesta:
            raise HTTPException(status_code=500, detail="Error al procesar la pregunta")
    
    # Actualizar el historial de chat global con la nueva conversación
    global_chat_history.append(("Usuario", question))
    global_chat_history.append(("Asistente", respuesta))
    
    # Convertir la respuesta del chatbot a audio utilizando la función text_to_speech
    audio_content = text_to_speech(respuesta)
    
    # Enviar respuesta en formato de audio
    return FileResponse(audio_content, media_type="audio/mp3")

# Función para grabar audio
def record_audio(file_path: str, duration: int = 5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = duration
    
    audio = pyaudio.PyAudio()
    
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    print("Recording...")
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Finished recording.")
    
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    with wave.open(file_path, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# Ruta para la grabación de audio
@app.post("/record_audio")
async def record_audio_endpoint(duration: int = 5):
    # Guardamos el archivo de audio grabado
    file_path_wav = os.path.join(UPLOAD_DIRECTORY, "recorded_audio.wav")
    
    # Normalizar la ruta del archivo
    file_path_wav = os.path.normpath(file_path_wav)
    
    # Llama a la función de grabación de audio
    record_audio(file_path_wav, duration)
    
    return {"message": "Audio grabado exitosamente."}

# Ruta para la conversión de voz a texto (STT)
@app.post("/speech_to_text")
async def stt(audio_content: bytes):
    text = speech_to_text(audio_content)
    return {"text": text}

# Ruta para la conversión de texto a voz (TTS)
@app.post("/text_to_speech")
async def tts(text: str):
    audio_content = text_to_speech(text)
    return {"audio_content": audio_content}

# Ruta para ver el historial del chat
@app.get("/chat_history")
async def view_chat_history():
    global global_chat_history
    # La función `view_chat_history` devuelve el historial global del chat
    return {"chat_history": global_chat_history}

# Manejar solicitudes para el ícono de favicon
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Puedes devolver una imagen de ícono si tienes una
    return

# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
