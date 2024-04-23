import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_ai21 import AI21Embeddings
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PROMPT: str = ""


def text_to_chunks(data: WebBaseLoader) -> list[str]:
    """Splits text data into chunks of a specified size with optional overlap."""
    text_splitter: RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0
    )
    all_splits: list[str] = text_splitter.split_documents(data)
    return all_splits


def generate_embeddings(chunks: list[str]) -> FAISS:
    """Generates embeddings for a list of text chunks using an AI21Embeddings model."""
    embedding_model: AI21Embeddings = AI21Embeddings()
    vector_db: FAISS = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    return vector_db

PROMPT:str = ""
async def get_prompt() -> str:  # Specify return type as string
    """Retrieves a prompt from the langchain hub."""
    global PROMPT
    PROMPT = hub.pull("rlm/rag-prompt")


def get_answer_from_AI(
    vector_db: FAISS, question: str
) -> str:  # Specify input and output types
    """Gets an answer to a question using a RetrievalQA chain with the provided vector store and question."""

    
    llm = ChatGoogleGenerativeAI(model="gemini-pro")

    qa_chain: RetrievalQA = RetrievalQA.from_chain_type(
        llm, retriever=vector_db.as_retriever(), chain_type_kwargs={"prompt": PROMPT}
    )

    result: dict[str, str] = qa_chain.invoke({"query": question})
    return result["result"]


asyncio.run(get_prompt())

loader = WebBaseLoader(input("Enter the url: "))
data: str = loader.load()  # Type hint for data variable after loading

chunks: list[str] = text_to_chunks(data=data)
document_vector_db: FAISS = generate_embeddings(chunks=chunks)

ask: bool = True
while ask:
    query: str = input("Enter your questions: ")  # Type hint for query input
    answer: str = get_answer_from_AI(vector_db=document_vector_db, question=query)
    print(answer)

    ask = int(input("Do you want to ask more?\n 0. No\n 1. Yes\n  "))

