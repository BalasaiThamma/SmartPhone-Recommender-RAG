# Smartphone Recommendation System

A simple RAG-based smartphone recommendation system that uses semantic search to recommend mobile phones based on user queries.

## Flow

Dataset (CSV)
→ Embeddings (BGE)
→ ChromaDB (Vector Store)
→ User Query
→ Similarity Search
→ Mobile Recommendations

## Tech Stack

- Python
- ChromaDB
- Sentence Transformers
- BAAI/bge-small-en-v1.5

## Files

- `ingest.py` – Creates embeddings and stores them in ChromaDB
- `query.py` – Accepts user queries and retrieves relevant phones
- `mobile_phones.csv` – Mobile phone dataset

## Run

```bash
python ingest.py
python query.py
Example Query
Best Samsung phone under 30000
