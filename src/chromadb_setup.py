import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize a persistent client
client = chromadb.PersistentClient(path="./mind_chroma_db")

# Create or retrieve a collection (similar to a table in SQL)
collection = client.create_collection(name="mind_news")

# Add documents (ChromaDB automatically generates the embeddings)
df = pd.read_csv(
    '/Users/jessicakahn/Documents/repos/MIND/data/MINDsmall_train/news.tsv', 
    sep="\t", 
    header=None,
    names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
)
df["abstract"] = df["abstract"].fillna("")
df["text_to_embed"] = "Title: " + df["title"] + " | Abstract: " + df["abstract"]

# embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = embed_model.encode(df["text_to_embed"].to_list(), batch_size=64, normalize_embeddings=True)
# embeddings.shape == (n_articles, d)
# np.save("data/mind_abstract_embeddings.npy", embeddings)

BATCH_SIZE = 1000
total_rows = len(df)

for i in range(0, total_rows, BATCH_SIZE):
    batch = df.iloc[i:i + BATCH_SIZE]
    
    # Prepare ChromaDB inputs
    ids = batch["news_id"].tolist()
    documents = batch["text_to_embed"].tolist()
    
    # Construct structured metadata dictionaries for advanced filtering
    metadatas = [
        {"category": row["category"], "subcategory": row["subcategory"], "title": row["title"]}
        for _, row in batch.iterrows()
    ]
    
    # Add batch to collection
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print(f"Ingested {min(i + BATCH_SIZE, total_rows)} / {total_rows} articles.")

print("MIND Dataset successfully added to ChromaDB!")