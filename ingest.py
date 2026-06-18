import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


df = pd.read_csv("Data/Mobiles Dataset.csv", encoding="latin1")

print(df.head())
print(df.columns)

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="mobile_phones"
)

for index, row in df.iterrows():

    chunk = f"""
    Company: {row['Company Name']}
    Model: {row['Model Name']}
    RAM: {row['RAM']}
    Processor: {row['Processor']}
    Battery: {row['Battery Capacity']}
    Screen Size: {row['Screen Size']}
    Front Camera: {row['Front Camera']}
    Back Camera: {row['Back Camera']}
    India Price: {row['Launched Price (India)']}
    Launch Year: {row['Launched Year']}
    """

    embedding = embedding_model.encode(chunk)

    collection.add(
        ids=[str(index)],
        documents=[chunk],
        embeddings=[embedding.tolist()]
    )

    if index % 100 == 0:
        print(f"Processed {index} records")

print(f"Total Records Stored: {collection.count()}")
print("Data stored successfully")