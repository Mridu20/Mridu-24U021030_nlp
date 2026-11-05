# Step 1: build the clean catalog from ANY movie CSV in the data/ folder.
# Usage:  python prepare_data.py            (auto-finds the csv)
#         python prepare_data.py data/top_rated_movies.csv
import ast, glob, json, sys
import pandas as pd

GENRES = {28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
          99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
          27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
          878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"}

if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    files = [f for f in glob.glob("data/*.csv") if not f.endswith("catalog.csv")]
    if not files:
        sys.exit("No csv found in the data/ folder. Put your movie csv there.")
    path = files[0]
print("Reading:", path)

df = pd.read_csv(path)
print("Columns found:", list(df.columns))
low = {c.lower(): c for c in df.columns}
pick = lambda *names: next((low[n] for n in names if n in low), None)

c_id = pick("id", "movie_id")
c_title = pick("title", "original_title", "name")
c_desc = pick("overview", "description", "plot")
c_gen = pick("genres", "genre_ids", "genre")
c_date = pick("release_date", "release_year", "year")
if not c_title or not c_desc:
    sys.exit("Could not find a title and a description/overview column. Send me the 'Columns found' line.")


def parse_genres(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    try:
        obj = json.loads(s)
    except Exception:
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return s  # already plain text like "Action, Drama"
    if isinstance(obj, (int, str)):
        obj = [obj]
    names = []
    for g in obj:
        if isinstance(g, dict):
            names.append(g.get("name") or GENRES.get(g.get("id"), ""))
        elif isinstance(g, int):
            names.append(GENRES.get(g, ""))
        else:
            names.append(str(g))
    return ", ".join(n for n in names if n)


out = pd.DataFrame()
out["movie_id"] = df[c_id] if c_id else range(len(df))
out["title"] = df[c_title]
out["description"] = df[c_desc]
out["genres"] = df[c_gen].apply(parse_genres) if c_gen else ""
out["genres"] = out["genres"].replace("", "Unknown")
if c_date and "year" in c_date.lower():
    out["year"] = pd.to_numeric(df[c_date], errors="coerce")
elif c_date:
    out["year"] = pd.to_datetime(df[c_date], errors="coerce").dt.year
else:
    out["year"] = 0

out = out.dropna(subset=["title", "description", "year"])
out = out.drop_duplicates("movie_id").drop_duplicates("title")
out["year"] = out["year"].astype(int)
out = out[["movie_id", "title", "year", "genres", "description"]]
out.to_csv("data/catalog.csv", index=False)
print("Saved data/catalog.csv with", len(out), "movies")
print(out.head(3).to_string())
