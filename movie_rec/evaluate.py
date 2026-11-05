# Compare baseline vs our system. Add more queries (aim for 50+) for a stronger result.
import pandas as pd
import matplotlib.pyplot as plt
from core import recommend, catalog
from baseline import baseline

QUERIES = [  # (query, genre we expect in good results)
    ("space adventure with an emotional story", "Science Fiction"),
    ("scary haunted house", "Horror"),
    ("funny romantic story", "Romance"),
    ("animated family movie for kids", "Animation"),
    ("gritty crime thriller with detectives", "Crime"),
    ("epic war drama", "War"),
    ("fantasy quest with magic and dragons", "Fantasy"),
    ("heist with clever twists", "Crime"),
    ("superhero team saves the world", "Action"),
    ("true story historical drama", "History"),
]

titles = {t.lower().strip(): g for t, g in zip(catalog.title, catalog.genres)}
stat = {"baseline": {"total": 0, "bad": 0, "good": 0, "found": 0},
        "ours": {"total": 0, "bad": 0, "good": 0, "found": 0}}
abstain = 0

for q, genre in QUERIES:
    for t in baseline(q):
        s = stat["baseline"]; s["total"] += 1
        g = titles.get(t.lower().strip())
        if g is None: s["bad"] += 1
        else:
            s["found"] += 1; s["good"] += genre in g
    out = recommend(q, check_reasons=False)
    abstain += out["abstained"]
    for r in out["results"]:
        s = stat["ours"]; s["total"] += 1
        if r["title"].lower().strip() not in titles: s["bad"] += 1
        else:
            s["found"] += 1; s["good"] += genre in r["genres"]

rows = []
for name, s in stat.items():
    rows.append({"system": name,
                 "hallucination_rate_%": round(100 * s["bad"] / max(s["total"], 1), 1),
                 "genre_precision_%": round(100 * s["good"] / max(s["found"], 1), 1)})
res = pd.DataFrame(rows)
print(res.to_string(index=False)); print("Abstained queries (ours):", abstain)
res.to_csv("results.csv", index=False)
res.set_index("system").plot(kind="bar", rot=0); plt.title("Baseline vs ours")
plt.tight_layout(); plt.savefig("results.png"); print("Saved results.csv, results.png")
