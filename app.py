"""Streamlit UI.  Run:  streamlit run app.py"""
import json

import pandas as pd
import streamlit as st

import config
from core import MovieNotFoundError, UserNotFoundError
import serving

st.set_page_config(page_title="CineMatch | ALS Recommender", page_icon="🎬", layout="wide")
st.markdown("""
<style>
.hero{padding:1.4rem 1.6rem;border-radius:14px;margin-bottom:1rem;color:#fff;
      background:linear-gradient(120deg,#1f2a44 0%,#5b3fd1 60%,#c2417b 100%);}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.3rem 0 0;opacity:.85}
div[data-testid="stMetric"]{background:rgba(127,127,127,.08);padding:.8rem 1rem;border-radius:10px}
</style>
<div class="hero"><h1>🎬 CineMatch</h1>
<p>Scalable movie recommendations · PySpark ALS matrix factorization · cosine item similarity</p></div>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ALS factors...")
def load_engine() -> serving.LocalRecommender:
    return serving.LocalRecommender()


try:
    eng = load_engine()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python download_data.py` then "
             "`SAVE_MODEL=1 python train.py` first.")
    st.stop()

RATING_COL = st.column_config.ProgressColumn("Predicted ⭐", min_value=0.5, max_value=5.0, format="%.2f")
SIM_COL = st.column_config.ProgressColumn("Cosine similarity", min_value=0.0, max_value=1.0, format="%.3f")

with st.sidebar:
    st.header("About")
    st.write(f"**{len(eng.user_ids):,}** users · **{len(eng.space.ids):,}** movies · "
             f"latent rank **{eng.space.V.shape[1]}**")
    st.caption("Predictions are clipped to [0.5, 5.0]. Already-rated movies are excluded.")

tab_rec, tab_sim, tab_perf = st.tabs(["🎯 For a user", "🧬 Similar movies", "📈 Model performance"])

with tab_rec:
    c1, c2 = st.columns([1, 1])
    uid = c1.number_input("User ID", min_value=1, value=int(eng.user_ids[0]), step=1)
    n = c2.slider("How many recommendations?", 1, 30, 10)
    if st.button("Recommend", type="primary", key="rec"):
        try:
            left, right = st.columns([3, 2])
            with left:
                st.subheader(f"Top {n} for user {uid}")
                st.dataframe(eng.recommend_top_n_for_user(int(uid), n), hide_index=True,
                             use_container_width=True, column_config={"predicted_rating": RATING_COL})
            with right:
                st.subheader("Their favourites")
                st.dataframe(eng.user_history(int(uid), 8)[["title", "rating"]],
                             hide_index=True, use_container_width=True)
        except UserNotFoundError as exc:
            st.warning(str(exc))

with tab_sim:
    title = st.selectbox("Pick a movie (type to search)", eng.titles,
                         index=eng.titles.index("Toy Story (1995)") if "Toy Story (1995)" in eng.titles else 0)
    k = st.slider("Number of similar movies", 1, 20, 5)
    if st.button("Find similar", type="primary", key="sim"):
        try:
            st.dataframe(eng.get_similar_movies(title, k), hide_index=True,
                         use_container_width=True, column_config={"similarity": SIM_COL})
        except MovieNotFoundError as exc:
            st.warning(str(exc))

with tab_perf:
    if not config.METRICS_PATH.exists():
        st.info("metrics.json not found. Re-run training with SAVE_MODEL=1.")
    else:
        m = json.loads(config.METRICS_PATH.read_text())
        a, b, c = st.columns(3)
        a.metric("Global mean RMSE", m["global_mean_rmse"])
        b.metric("ALS baseline RMSE", m["als_baseline_rmse"], f"{m['als_baseline_rmse']-m['global_mean_rmse']:+.4f}", delta_color="inverse")
        c.metric("Tuned ALS RMSE", m["als_tuned_rmse"], f"{m['als_tuned_rmse']-m['global_mean_rmse']:+.4f}", delta_color="inverse")
        st.bar_chart(pd.Series({"Global mean": m["global_mean_rmse"], "ALS baseline": m["als_baseline_rmse"],
                                "ALS tuned": m["als_tuned_rmse"]}, name="RMSE"))
        st.write("**Best params:**", m["best_params"])
        st.dataframe(pd.DataFrame(m["cv_results"]), hide_index=True, use_container_width=True)
