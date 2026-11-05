import streamlit as st
from core import recommend

st.title("Hallucination-free Movie Recommender")
query = st.text_input("What are you in the mood for?", "a mind-bending sci-fi thriller")
if st.button("Recommend"):
    with st.spinner("Thinking..."):
        out = recommend(query)
    if out["abstained"]:
        st.warning("No good match found in the catalog.")
    for r in out["results"]:
        st.subheader(f"{r['title']} ({r['year']})")
        st.caption(r["genres"])
        st.write(r["reason"] + ("" if r["reason_verified"] else "  \n*(LLM reason rejected; showing stored plot)*"))
