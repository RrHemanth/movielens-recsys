import mlflow
import pandas as pd
import streamlit as st

from recsys.config import load_config
from recsys.monitoring.drift import PSI_COLUMNS
from recsys.serving.app import RecommendationStore
from recsys.tracking import setup_mlflow

st.set_page_config(page_title="MovieLens recommender", layout="wide")
cfg = load_config()


@st.cache_resource
def load_store():
    return RecommendationStore(cfg["paths"]["serving"])


@st.cache_data
def load_runs():
    setup_mlflow(cfg["mlflow"]["experiment"])
    return mlflow.search_runs(experiment_names=[cfg["mlflow"]["experiment"]])


@st.cache_data
def load_drift():
    return pd.read_csv(cfg["paths"]["processed"] / "drift_report.csv")


st.title("MovieLens 25M recommender")
st.caption("Offline evaluation on historical MovieLens data. No live users, no A/B tests.")
results_tab, drift_tab, try_tab = st.tabs(["Results", "Drift", "Try it"])

with results_tab:
    runs = load_runs()
    final = runs[runs["tags.mlflow.runName"] == "FINAL TEST 2019"].iloc[0]
    systems = [("returning", "popularity", "returning_popularity"), ("returning", "ALS", "returning_als"),
               ("returning", "re-ranker", "returning_reranker"), ("new", "popularity", "new_popularity"),
               ("new", "fold-in", "new_foldin")]
    table = pd.DataFrame([
        {"users": seg, "system": name,
         **{m: final[f"metrics.{prefix}_{m}"] for m in ["ndcg", "recall", "coverage", "diversity"]}}
        for seg, name, prefix in systems
    ])
    st.subheader("Final test: 2019 held-out year (run once)")
    st.dataframe(table.style.format(precision=4), hide_index=True)
    st.bar_chart(table, x="system", y="ndcg", color="users", horizontal=True)

    st.subheader("Validation runs (2018), returning users")
    cols = {"tags.mlflow.runName": "run", "metrics.ndcg_returning": "ndcg",
            "metrics.recall_returning": "recall", "metrics.coverage_returning": "coverage"}
    validation = runs.dropna(subset=["metrics.ndcg_returning"])[list(cols)].rename(columns=cols)
    st.dataframe(validation.sort_values("ndcg", ascending=False).style.format(precision=4), hide_index=True)

with drift_tab:
    drift = load_drift().set_index("month")
    st.subheader("Monthly PSI vs the 2017 reference (model built 2018-01-01, never retrained)")
    st.line_chart(drift[PSI_COLUMNS])
    st.caption("Rule of thumb: below 0.1 stable, 0.1 to 0.25 watch, above 0.25 drift. Retraining triggers at 0.1.")
    st.subheader("Share of ratings the model can't handle")
    st.line_chart(drift[["unknown_movie_share", "new_user_share"]])
    st.dataframe(drift, use_container_width=True)

@st.cache_resource
def title_index():
    """{title: movie_id} and a sorted title list, built once."""
    store = load_store()
    index = {title: m for m, title in store.titles.items()}
    return index, sorted(index)


@st.fragment
def returning_user_section(store):
    st.subheader("Returning user")
    user_id = st.number_input("User ID", min_value=1, value=75, step=1)
    source, ids = store.for_user(int(user_id), 10)
    st.write(f"Source: **{source}**")
    st.table(pd.DataFrame({"title": [store.titles.get(int(m), "unknown") for m in ids]}))


@st.fragment
def new_user_section(store):
    st.subheader("New user")
    index, titles = title_index()
    if "picked" not in st.session_state:
        st.session_state.picked = ["Toy Story (1995)", "Finding Nemo (2003)"]

    query = st.text_input("Search for a movie you liked (press Enter)", placeholder="e.g. Up, Inception")
    if query:
        matches = [t for t in titles if query.lower() in t.lower()][:25]
        if matches:
            choice = st.selectbox("Matches", matches)
            if st.button("Add") and choice not in st.session_state.picked:
                st.session_state.picked.append(choice)
        else:
            st.write("No matches.")

    if st.button("Clear liked movies"):
        st.session_state.picked = []
    st.write("**Liked:** " + (", ".join(st.session_state.picked) or "nothing yet"))

    if st.session_state.picked:
        source, ids = store.for_new_user([index[t] for t in st.session_state.picked], 10)
        st.write(f"Source: **{source}**")
        st.table(pd.DataFrame({"title": [store.titles.get(int(m), "unknown") for m in ids]}))


with try_tab:
    store = load_store()
    st.caption(f"Serving snapshot as of {store.manifest.get('as_of')} | commit {store.manifest.get('git_commit')}")
    returning_user_section(store)
    new_user_section(store)