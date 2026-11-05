# The "bad" system: plain LLM prompting, no catalog, no constraints
import json
import ollama
from core import LLM


def baseline(query, n=5):
    prompt = f"Recommend {n} movies for this request: {query}. Return JSON with a list of titles."
    schema = {"type": "object",
              "properties": {"titles": {"type": "array", "items": {"type": "string"}}},
              "required": ["titles"]}
    resp = ollama.chat(model=LLM, messages=[{"role": "user", "content": prompt}],
                       format=schema, options={"temperature": 0})
    return json.loads(resp["message"]["content"])["titles"][:n]


if __name__ == "__main__":
    print(baseline("I like Inception and The Matrix"))
