
from fastapi import FastAPI
import chromadb
from sentence_transformers import SentenceTransformer
import ollama


app = FastAPI()


embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "mobile_phones"
)


def retrieve(query):

    query_embedding = embedding_model.encode(query)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )

    return results["documents"][0]


@app.get("/")
def home():
    return {
        "message": "Mobile Phone Recommendation RAG API",
        "models": [
            "qwen2.5:0.5b",
            "tinyllama",
            "smollm2:135m"
        ]
    }


@app.get("/qwen")
def qwen(query: str):

    context = retrieve(query)

    prompt = f"""
    You are a mobile phone recommendation expert.

    Mobile Data:
    {context}

    User Query:
    {query}

    Recommend the best phone and explain why.
    """

    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "model": "Qwen 2.5",
        "query": query,
        "response": response["message"]["content"]
    }


@app.get("/tinyllama")
def tinyllama(query: str):

    context = retrieve(query)

    prompt = f"""
    Mobile Data:
    {context}

    User Query:
    {query}

    Recommend the best phone and explain why.
    """

    response = ollama.chat(
        model="tinyllama",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "model": "TinyLlama",
        "query": query,
        "response": response["message"]["content"]
    }


@app.get("/smollm")
def smollm(query: str):

    context = retrieve(query)

    prompt = f"""
    Mobile Data:
    {context}

    User Query:
    {query}

    Recommend the best phone and explain why.
    """

    response = ollama.chat(
        model="smollm2:135m",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return {
        "model": "SmolLM2",
        "query": query,
        "response": response["message"]["content"]
    }