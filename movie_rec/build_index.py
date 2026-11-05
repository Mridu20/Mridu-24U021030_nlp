# Step 2: turn every movie into a vector and store in FAISS
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

catalog = pd.read_csv("data/catalog.csv")
texts = (catalog.title + ". " + catalog.genres + ". " + catalog.description).tolist()

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")

index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine (vectors are normalized)
index.add(emb)
faiss.write_index(index, "data/movies.faiss")
print("Index built with", index.ntotal, "movies")
