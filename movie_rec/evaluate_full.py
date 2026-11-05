# Full evaluation: baseline vs retrieval-only vs our system. Run: python evaluate_full.py
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from core import recommend, retrieve, catalog
from baseline import baseline

IN_SCOPE = [  # (query, genre we expect in good results) - add more for a stronger result
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
NONSENSE = ["asdfgh qwerty zxcv", "xzqv lkjh mnbv", "blorp fnord quux", "mmmmm kkkkk jjjjj", "qpwoei alskdj"]

titles = {t.lower().strip(): g for t, g in zip(catalog.title, catalog.genres)}
pct = lambda a, b: round(100 * a / b, 1) if b else np.nan

B = dict(total=0, bad=0, found=0, good=0, time=0.0)
O = dict(total=0, bad=0, found=0, good=0, time=0.0, ver=0, abst=0)
R = dict(found=0, good=0)

for q, genre in IN_SCOPE:
    t0 = time.time(); names = baseline(q); B["time"] += time.time() - t0
    for t in names:
        B["total"] += 1
        g = titles.get(t.lower().strip())
        if g is None: B["bad"] += 1
        else: B["found"] += 1; B["good"] += genre in g

    t0 = time.time(); out = recommend(q); O["time"] += time.time() - t0
    O["abst"] += out["abstained"]
    for r in out["results"]:
        O["total"] += 1; O["ver"] += r["reason_verified"]
        if r["title"].lower().strip() not in titles: O["bad"] += 1
        else: O["found"] += 1; O["good"] += genre in r["genres"]

    top = retrieve(q).head(5)                      # retrieval only (no LLM)
    R["found"] += len(top); R["good"] += sum(genre in g for g in top.genres)

n = len(IN_SCOPE)
base_answered = sum(len(baseline(q)) > 0 for q in NONSENSE)
ours_abstained = sum(recommend(q, check_reasons=False)["abstained"] for q in NONSENSE)

nan = np.nan
table = pd.DataFrame({
    "metric": ["Hallucination rate % (lower=better)", "Genre precision@5 %",
               "Explanation faithfulness %", "Correct abstention on nonsense %",
               "False abstention on valid queries %", "Avg latency (s/query)"],
    "baseline LLM": [pct(B["bad"], B["total"]), pct(B["good"], B["found"]), nan,
                     pct(len(NONSENSE) - base_answered, len(NONSENSE)), 0.0, round(B["time"] / n, 1)],
    "retrieval only": [0.0, pct(R["good"], R["found"]), nan, nan, nan, nan],
    "our system": [pct(O["bad"], O["total"]), pct(O["good"], O["found"]),
                   pct(O["ver"], O["total"]), pct(ours_abstained, len(NONSENSE)),
                   pct(O["abst"], n), round(O["time"] / n, 1)],
})
print(table.to_string(index=False))
table.to_csv("results_full.csv", index=False)

chart = table.iloc[[0, 1, 2, 3]].set_index("metric")
chart.plot(kind="bar", rot=15, figsize=(9, 5)); plt.ylabel("%"); plt.title("Baseline vs ours")
plt.tight_layout(); plt.savefig("results_full.png")
print("Saved results_full.csv and results_full.png")
