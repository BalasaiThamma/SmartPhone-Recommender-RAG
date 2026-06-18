# Mobile Phone Recommendation System using RAG

A Retrieval-Augmented Generation (RAG) based Mobile Phone Recommendation System that uses semantic search and local LLMs to recommend mobile phones based on user queries.

## Tech Stack

- Python
- FastAPI
- ChromaDB
- Ollama
- Pandas
- Sentence Transformers

## LLM Models Used

- Qwen2.5:0.5B
- TinyLlama
- SmolLM2:135M

## Embedding Model

- BAAI/bge-small-en-v1.5

## RAG Workflow

Dataset → Chunking → Embeddings → ChromaDB → Retrieval → LLM → Recommendation

## Dataset

- Kaggle Mobile Phones Dataset
- 930 Mobile Phone Records


uvicorn app:app --reload
