# The main pipeline: retrieve -> constrained LLM rerank -> validate -> ground explanations
import json
import faiss
import ollama
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder

LLM = "llama3.2"      # any model you pulled with ollama
MIN_SCORE = 0.25      # below this, we abstain instead of guessing

catalog = pd.read_csv("data/catalog.csv")
by_id = catalog.set_index("movie_id")
index = faiss.read_index("data/movies.faiss")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
_nli = None


def retrieve(query, k=30):
    q = embedder.encode([query], normalize_embeddings=True).astype("float32")
    scores, idx = index.search(q, k)
    rows = catalog.iloc[idx[0]].copy()
    rows["score"] = scores[0]
    return rows


def reason_ok(movie_id, reason):
    """Reject a reason only if an NLI model says it CONTRADICTS the stored facts."""
    global _nli
    if _nli is None:
        _nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")
    m = by_id.loc[movie_id]
    premise = f"{m.title} ({m.year}). Genres: {m.genres}. {m.description}"
    label = _nli.predict([(premise, reason)])[0].argmax()  # 0=contradiction 1=entailment 2=neutral
    return label != 0


def recommend(query, n=5, check_reasons=True):
    cands = retrieve(query)
    if cands.score.iloc[0] < MIN_SCORE:
        return {"abstained": True, "results": []}

    ids = [int(i) for i in cands.movie_id]
    listing = "\n".join(
        f"{r.movie_id} | {r.title} ({r.year}) | {r.genres} | {r.description[:200]}"
        for r in cands.itertuples()
    )
    # JSON schema: movie_id can ONLY be one of the candidate ids -> no invented movies
    schema = {
        "type": "object",
        "properties": {"recommendations": {
            "type": "array", "maxItems": n,
            "items": {"type": "object",
                      "properties": {"movie_id": {"type": "integer", "enum": ids},
                                     "reason": {"type": "string"}},
                      "required": ["movie_id", "reason"]}}},
        "required": ["recommendations"],
    }
    prompt = (f"User request: {query}\n\nCandidate movies (id | title | genres | plot):\n{listing}\n\n"
              f"Pick the best {n} movies for the request. For each give a one-sentence reason "
              f"using ONLY facts from the candidate list.")
    resp = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}],
                       format=schema, options={"temperature": 0})
    data = json.loads(resp["message"]["content"])

    results, seen = [], set()
    for rec in data["recommendations"]:
        mid = rec["movie_id"]
        if mid not in ids or mid in seen:       # final validator
            continue
        seen.add(mid)
        m = by_id.loc[mid]                      # facts come from the DB, never from the LLM
        ok = reason_ok(mid, rec["reason"]) if check_reasons else True
        results.append({"movie_id": mid, "title": m.title, "year": int(m.year),
                        "genres": m.genres,
                        "reason": rec["reason"] if ok else m.description[:200] + "...",
                        "reason_verified": ok})
    return {"abstained": False, "results": results}
