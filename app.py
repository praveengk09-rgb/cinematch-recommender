"""Streamlit UI. Run: streamlit run app.py"""
import json
import re
import urllib.parse
import pandas as pd
import requests
import streamlit as st

import config
from core import MovieNotFoundError, UserNotFoundError
import serving

st.set_page_config(page_title="CineMatch | ALS Recommender", page_icon="🎬", layout="wide")

# Custom CSS
st.markdown("""
<style>
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    margin-bottom: 1rem;
    color: #fff;
    background: linear-gradient(120deg, #1f2a44 0%, #5b3fd1 60%, #c2417b 100%);
}
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: .3rem 0 0; opacity: .85; }
div[data-testid="stMetric"] {
    background: rgba(127, 127, 127, .08);
    padding: .8rem 1rem;
    border-radius: 10px;
}
</style>
<div class="hero">
  <h1>🎬 CineMatch</h1>
  <p>Scalable movie recommendations · PySpark ALS matrix factorization · cosine item similarity</p>
</div>
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


# Helper: Extract Year from Movie Title
def extract_year(title: str) -> tuple[str, int | None]:
    match = re.search(r'\((\d{4})\)', title)
    year = int(match.group(1)) if match else None
    clean_title = re.sub(r'\s*\([^)]*\)', '', title).strip()
    return clean_title, year


# Robust TMDB Poster Fetcher
@st.cache_data(show_spinner=False)
def fetch_poster_url(movie_title: str) -> str:
    """Fetch real movie poster from TMDB API with browser-like headers."""
    clean_title, year = extract_year(movie_title)
    
    api_key = "15d2ea6d0dc1d476efbca3eba1e9bbfb"
    params = {
        "api_key": api_key,
        "query": clean_title,
    }
    if year:
        params["year"] = year

    url = f"https://api.themoviedb.org/3/search/movie?{urllib.parse.urlencode(params)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }

    try:
        response = requests.get(url, headers=headers, timeout=3)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            for res in results:
                if res.get("poster_path"):
                    return f"https://image.tmdb.org/t/p/w500{res['poster_path']}"
    except Exception:
        pass

    # Fallback to SVG placeholder
    text_encoded = urllib.parse.quote(clean_title[:25])
    return f"https://placehold.co/500x750/1f2a44/FFFFFF/png?text={text_encoded}"


# Column configurations for table views
RATING_COL = st.column_config.ProgressColumn("Predicted ⭐", min_value=0.5, max_value=5.0, format="%.2f")
SIM_COL = st.column_config.ProgressColumn("Cosine similarity", min_value=0.0, max_value=1.0, format="%.3f")

# Find the underlying dataframe safely across possible engine attributes
movies_data = None
for attr in ['movies_df', 'df', 'items_df']:
    if hasattr(eng, attr):
        movies_data = getattr(eng, attr)
        break
    elif hasattr(eng, 'space') and hasattr(eng.space, attr):
        movies_data = getattr(eng.space, attr)
        break

# Sidebar Controls & Filters
with st.sidebar:
    st.header("About")
    st.write(f"**{len(eng.user_ids):,}** users · **{len(eng.space.ids):,}** movies · "
             f"latent rank **{eng.space.V.shape[1]}**")
    st.caption("Predictions are clipped to [0.5, 5.0]. Already-rated movies are excluded.")
    
    st.divider()
    st.header("🔍 Filter Options")

    # Extract unique genres safely
    all_genres = set()
    if movies_data is not None and 'genres' in movies_data.columns:
        for g_str in movies_data['genres'].dropna():
            for g in str(g_str).split('|'):
                if g != '(no genres listed)':
                    all_genres.add(g)

    selected_genres = st.multiselect(
        "Filter by Genre",
        options=sorted(list(all_genres)),
        default=[]
    )

    year_range = st.slider(
        "Release Year Range",
        min_value=1900,
        max_value=2024,
        value=(1980, 2024)
    )

    view_mode = st.radio("Display Mode", options=["Poster Grid", "Data Table"], index=0)


# Main Tabs
tab_rec, tab_sim, tab_perf = st.tabs(["🎯 For a user", "🧬 Similar movies", "📈 Model performance"])

with tab_rec:
    c1, c2 = st.columns([1, 1])
    uid = c1.number_input("User ID", min_value=1, value=int(eng.user_ids[0]), step=1)
    n = c2.slider("How many recommendations?", 1, 30, 8)
    
    if st.button("Recommend", type="primary", key="rec"):
        try:
            # Fetch candidate pool to allow genre and year filtering
            recs = eng.recommend_top_n_for_user(int(uid), 100)

            # Apply Release Year Filter
            recs['year'] = recs['title'].apply(lambda x: extract_year(x)[1] or 2000)
            recs = recs[(recs['year'] >= year_range[0]) & (recs['year'] <= year_range[1])]

            # Apply Genre Filter
            if selected_genres and 'genres' in recs.columns:
                def genre_match(genres_str):
                    movie_g = set(str(genres_str).split('|'))
                    return any(g in movie_g for g in selected_genres)
                recs = recs[recs['genres'].apply(genre_match)]

            filtered_recs = recs.head(n)

            if filtered_recs.empty:
                st.warning("No recommendations matched your selected genre and year filters!")
            else:
                left, right = st.columns([3, 2])
                with left:
                    st.subheader(f"Top {len(filtered_recs)} for user {uid}")
                    if view_mode == "Poster Grid":
                        cols = st.columns(4)
                        for idx, (_, row) in enumerate(filtered_recs.iterrows()):
                            with cols[idx % 4]:
                                poster_url = fetch_poster_url(row['title'])
                                st.image(poster_url, use_container_width=True)
                                st.markdown(f"**{row['title']}**")
                                st.caption(f"⭐ **Predicted:** {row['predicted_rating']:.2f} / 5.0")
                                if 'genres' in row and pd.notna(row['genres']):
                                    st.caption(f"🏷️ {str(row['genres']).replace('|', ', ')}")
                    else:
                        cols_to_show = ["movieId", "title", "predicted_rating"]
                        if 'genres' in filtered_recs.columns:
                            cols_to_show.insert(2, "genres")
                        st.dataframe(filtered_recs[cols_to_show],
                                     hide_index=True, use_container_width=True,
                                     column_config={"predicted_rating": RATING_COL})

                with right:
                    st.subheader("Their favourites")
                    st.dataframe(eng.user_history(int(uid), 8)[["title", "rating"]],
                                 hide_index=True, use_container_width=True)
        except UserNotFoundError as exc:
            st.warning(str(exc))

with tab_sim:
    title = st.selectbox("Pick a movie (type to search)", eng.titles,
                         index=eng.titles.index("Toy Story (1995)") if "Toy Story (1995)" in eng.titles else 0)
    k = st.slider("Number of similar movies", 1, 20, 4)
    
    if st.button("Find similar", type="primary", key="sim"):
        try:
            sims = eng.get_similar_movies(title, k)
            st.subheader(f"Movies similar to '{title}'")
            
            if view_mode == "Poster Grid":
                cols = st.columns(4)
                for idx, (_, row) in enumerate(sims.iterrows()):
                    with cols[idx % 4]:
                        poster_url = fetch_poster_url(row['title'])
                        st.image(poster_url, use_container_width=True)
                        st.markdown(f"**{row['title']}**")
                        st.caption(f"🎯 **Similarity:** {row['similarity']:.1%}")
                        if 'genres' in row and pd.notna(row['genres']):
                            st.caption(f"🏷️ {str(row['genres']).replace('|', ', ')}")
            else:
                st.dataframe(sims, hide_index=True, use_container_width=True,
                             column_config={"similarity": SIM_COL})
        except MovieNotFoundError as exc:
            st.warning(str(exc))

with tab_perf:
    if not config.METRICS_PATH.exists():
        st.info("metrics.json not found. Re-run training with SAVE_MODEL=1.")
    else:
        m = json.loads(config.METRICS_PATH.read_text())
        a, b, c = st.columns(3)
        a.metric("Global mean RMSE", m["global_mean_rmse"])
        b.metric("ALS baseline RMSE", m["als_baseline_rmse"],
                 f"{m['als_baseline_rmse']-m['global_mean_rmse']:+.4f}", delta_color="inverse")
        c.metric("Tuned ALS RMSE", m["als_tuned_rmse"],
                 f"{m['als_tuned_rmse']-m['global_mean_rmse']:+.4f}", delta_color="inverse")
        st.bar_chart(pd.Series({
            "Global mean": m["global_mean_rmse"],
            "ALS baseline": m["als_baseline_rmse"],
            "ALS tuned": m["als_tuned_rmse"]
        }, name="RMSE"))
        st.write("**Best params:**", m["best_params"])
        st.dataframe(pd.DataFrame(m["cv_results"]), hide_index=True, use_container_width=True)