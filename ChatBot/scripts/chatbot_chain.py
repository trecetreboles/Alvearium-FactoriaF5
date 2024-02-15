from operator import itemgetter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.schema import format_document
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_community.callbacks import get_openai_callback
from extract_apis_keys import load

# Cargar la clave de la API de OpenAI
OPENAI_API_KEY = load()

class Chatbot:
    def __init__(self):
        if OPENAI_API_KEY is None:
            raise ValueError("OPENAI_API_KEY no puede ser None")
        self._embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
        self._persisted_vectorstore = None
        self._retriever = None
        self._ANSWER_PROMPT = None
        self._CONDENSE_QUESTION_PROMPT = None
        self._model = ChatOpenAI(model="gpt-4-0125-preview", temperature=0.5, max_tokens=3000)
        self._chain = None
        self._salida = None
        self._callback = None
        self._inputs = None
        self._context = None

    def load_database(self, _embeddings):
        index_directory = "./faiss_index"
        persisted_vectorstore = FAISS.load_local(index_directory, _embeddings)
        retriever = persisted_vectorstore.as_retriever()
        return retriever

    def chain_template(self, template, _template):
        template = """Answer the question based only on the following context:
                    {context}

                    Question: {question}

                    System: Answer according to the language you are typing in the prompt {question}. 
                    """
        
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

                    Chat History:
                    {chat_history}
                    Follow Up Input: {question}
                    Standalone question:"""
        
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
        ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

        return ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT

    def _combine_documents(self, docs, document_separator="\n\n"):
        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
        doc_strings = [format_document(doc, DEFAULT_DOCUMENT_PROMPT) for doc in docs]
        return document_separator.join(doc_strings)

    def invoke_chain(self):
        retriever = self.load_database(self._embeddings)
        ANSWER_PROMPT, CONDENSE_QUESTION_PROMPT = self.chain_template(template=None, _template=None)

        _inputs = RunnableParallel(
            standalone_question=RunnablePassthrough.assign(
                chat_history=lambda x: get_buffer_string(x["chat_history"])
            )
            | CONDENSE_QUESTION_PROMPT
            | self._model
            | StrOutputParser(),
        )
        
        
        _context = {
            "context": itemgetter("standalone_question") | retriever | self._combine_documents,
            "question": lambda x: get_buffer_string(x.get("chat_history", []))
        }

        conversational_chain = _inputs | _context | ANSWER_PROMPT | self._model

        while True:
            question = input("Haz tu pregunta:")
            with get_openai_callback() as cb:
                salida = conversational_chain.invoke(
                    {
                        "question": question,
                        "chat_history": [
                            HumanMessage(content="Who wrote this notebook?"),
                            AIMessage(content="Harrison"),
                        ],
                    }
                )
                print(f"Total Cost (USD): ${format(cb.total_cost, '.20f')}")
                print(cb)
                print(salida)

if __name__ == "__main__":
    # Crear una instancia de la clase Chatbot
    my_chatbot = Chatbot()

    # Llamar al método invoke_chain()
    my_chatbot.invoke_chain()

    input("Presiona Enter para salir...")
