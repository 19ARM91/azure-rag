import os
import openai
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv(".env")

app = FastAPI()

openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "openai"
openai.api_version = "v1" 

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


qdrant_client = QdrantClient(":memory:")

vector_store = Qdrant(
    client=qdrant_client,
    collection_name="wine_collection",
    embeddings=embeddings
)

class Body(BaseModel):
    query: str


# -----------------------------
# Load embeddings helper
# -----------------------------

def load_embeddings():

    print("Loading dataset...")

    loader = CSVLoader("wine-ratings.csv")
    documents = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    docs = splitter.split_documents(documents)

    print("Creating embeddings and storing in Qdrant...")

    vector_store.add_documents(docs)

    print("Embeddings loaded successfully.")


# -----------------------------
# Run on application startup
# -----------------------------

@app.on_event("startup")
def startup_event():
    load_embeddings()


# -----------------------------
# Root redirect
# -----------------------------

@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)


# -----------------------------
# Ask endpoint
# -----------------------------

@app.post('/ask')
def ask(body: Body):

    search_result = search(body.query)

    response = assistant(body.query, search_result)

    return {"response": response}


# -----------------------------
# Vector search
# -----------------------------

def search(query):

    docs = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=5
    )

    result = docs[0][0].page_content

    print("Retrieved context:")
    print(result)

    return result


# -----------------------------
# LLM assistant
# -----------------------------

def assistant(query, context):

    messages = [

        {
            "role": "system",
            "content": "You are a helpful assistant that recommends wines based on provided context."
        },

        {
            "role": "system",
            "content": f"Context: {context}"
        },

        {
            "role": "user",
            "content": query
        }

    ]

    response = openai.ChatCompletion.create(
        model="phi-2",
        messages=messages
    )

    return response['choices'][0]['message']['content']
