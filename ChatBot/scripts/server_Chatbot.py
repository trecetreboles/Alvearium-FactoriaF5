from operator import itemgetter
from typing import List, Tuple

from extract_apis_keys import load

from fastapi import FastAPI, HTTPException
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.vectorstores.faiss import FAISS
from langchain_community.callbacks import get_openai_callback

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

OPENAI_API_KEY = load()

openai_embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)


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


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


index_directory = "./faiss_index"
persisted_vectorstore = FAISS.load_local(index_directory, openai_embeddings)
retriever = persisted_vectorstore.as_retriever()

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


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatOpenAI(model="gpt-4-0125-preview") | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

# Variable global para almacenar el historial del chat
global_chat_history = []

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
        salida = chain.invoke({"chat_history": chat_history, "question": question})
        print(cb)
        # Si ocurrió algún error al obtener la respuesta, lanza una excepción HTTP
        if not salida:
            raise HTTPException(status_code=500, detail="Error al procesar la pregunta")
    
    # Actualizar el historial de chat global con la nueva conversación
    global_chat_history.append(("Usuario", question))
    global_chat_history.append(("Asistente", salida))
    
    # Retorna la respuesta obtenida
    return {"answer": salida}

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

add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
