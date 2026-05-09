import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download
import os

# page config
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    /* hide streamlit defaults */
    header { visibility: hidden; }
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0e1117 50%, #1a0a0a 100%); }
    
    /* navbar */
    .navbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 2rem;
        border-bottom: 1px solid #222;
        margin-bottom: 0;
    }
    .navbar-logo {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e50914;
    }
    .navbar-links {
        display: flex;
        gap: 2rem;
    }
    .navbar-link {
        color: #aaaaaa;
        text-decoration: none;
        font-size: 0.95rem;
        cursor: pointer;
    }
    .navbar-link:hover { color: white; }
    .navbar-link.active { color: white; font-weight: 600; }

    /* hero */
    .hero {
        text-align: center;
        padding: 5rem 2rem 3rem;
    }
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        color: white;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    .hero-title span {
        background: linear-gradient(90deg, #e50914, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 3rem;
    }

    /* search bar */
    .search-container {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }

    /* buttons */
    .stButton > button {
        background: linear-gradient(90deg, #e50914, #ff4444);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 36px;
        font-size: 15px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #f40612, #ff5555);
        color: white;
    }

    /* selectbox */
    .stSelectbox > div > div {
        background-color: #1a1a2e;
        border: 1px solid #333;
        border-radius: 50px;
        color: white;
    }

    /* divider */
    hr { border-color: #222222; margin: 2rem 0; }

    /* movie card */
    .movie-card {
        position: relative;
        cursor: pointer;
        margin-bottom: 8px;
        border-radius: 8px;
    }
    .movie-card img {
        width: 100%;
        border-radius: 8px;
        display: block;
        aspect-ratio: 2/3;
        object-fit: cover;
    }
    .movie-tooltip {
        display: none;
        position: absolute;
        z-index: 999;
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 16px;
        width: 500px;
        left: 50%;
        transform: translateX(-50%);
        top: 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.9);
    }
    .movie-card:hover .movie-tooltip { display: block; }
</style>

<div class="navbar">
    <span class="navbar-logo">🎬 CineMatch</span>
    <div class="navbar-links">
        <span class="navbar-link active">Find Similar</span>
        <span class="navbar-link">Personal Recommendations</span>
    </div>
</div>

<div class="hero">
    <div class="hero-title">Discover Your Next<br><span>Favorite Movie</span></div>
    <div class="hero-subtitle">ML-powered recommendations based on what you love</div>
</div>
""", unsafe_allow_html=True)

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w300"

# download data from hugging face
def download_data():
    files = ['ratings_small.parquet', 'movies_enriched.parquet', 'svd_small.pkl', 'cosine_sim.pkl']
    for f in files:
        if not os.path.exists(f):
            print(f'Downloading {f}...')
            hf_hub_download(
                repo_id='krav0x/movie-recommender',
                filename=f,
                repo_type='dataset',
                local_dir='.'
            )

download_data()

# load data
@st.cache_data
def load_data():
    ratings = pd.read_parquet('ratings_small.parquet')
    movies = pd.read_parquet('movies_enriched.parquet')
    return ratings, movies

@st.cache_resource
def load_model():
    with open('svd_small.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_content_model(_movies):
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    return _movies, cosine_sim

@st.cache_data
def get_popular_movies(_ratings):
    return (_ratings.groupby('movieId')
            .size()
            .sort_values(ascending=False)
            .head(5000)
            .index.tolist())

def display_movies(movie_df):
    cols = st.columns(5)
    for i, (_, row) in enumerate(movie_df.iterrows()):
        with cols[i % 5]:
            poster = POSTER_BASE_URL + str(row['poster_path']) if pd.notna(row.get('poster_path')) and row.get('poster_path') else "https://via.placeholder.com/300x450?text=No+Poster"
            title = row['title']
            rating = f"{row['vote_average']:.1f}" if pd.notna(row.get('vote_average')) and row.get('vote_average') else "N/A"
            overview = str(row['overview'])[:400] + "..." if pd.notna(row.get('overview')) and row.get('overview') else "No description available."
            genres = row.get('genres', '')

            st.markdown(f"""
            <div class="movie-card">
                <img src="{poster}" style="width:100%; border-radius:8px;">
                <p style='text-align:center; color:white; font-size:12px; margin-top:6px'>{title}</p>
                <div class="movie-tooltip">
                    <div style="display:flex; gap:12px;">
                        <img src="{poster}" style="width:160px; border-radius:8px; flex-shrink:0;">
                        <div>
                            <p style="color:white; font-weight:bold; font-size:14px; margin:0 0 6px">{title}</p>
                            <p style="color:#e50914; font-size:12px; margin:0 0 6px">⭐ {rating} &nbsp;|&nbsp; {genres}</p>
                            <p style="color:#cccccc; font-size:12px; margin:0">{overview}</p>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

ratings, movies = load_data()
svd = load_model()
movies_cb, cosine_sim = load_content_model(movies)
popular_movies = get_popular_movies(ratings)

# --- header ---
st.title("🎬 Movie Recommender")
st.markdown("Discover movies you'll love using collaborative filtering and content-based recommendations.")
st.divider()

# --- section 1: content based ---
st.subheader("🎥 Find similar movies")
st.caption("Select a movie and we'll find similar ones based on genres and tags.")

movie_list = movies['title'].sort_values().tolist()
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button("Find Similar Movies"):
    idx = movies_cb[movies_cb['title'] == selected_movie].index
    if len(idx) == 0:
        st.error("Movie not found!")
    else:
        idx = idx[0]
        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:10]
        top_movies = movies_cb.iloc[[s[0] for s in scores]]
        display_movies(top_movies)

st.divider()

# --- section 2: personalized ---
st.subheader("👤 Personalized recommendations")
st.caption("Enter a user ID to get recommendations based on their rating history.")

user_id = st.number_input("Enter a user ID (1-162541)", min_value=1, max_value=162541, value=1)

if st.button("Get Personalized Recommendations"):
    with st.spinner("Finding recommendations..."):

        st.write("**Top rated movies by this user:**")
        top_rated = (ratings[ratings['userId'] == user_id]
                     .sort_values('rating', ascending=False)
                     .head(5)[['title', 'rating']])
        st.dataframe(top_rated, hide_index=True)

        st.write("**Recommended for you:**")
        rated_movies = set(ratings[ratings['userId'] == user_id]['movieId'].tolist())
        unrated = [m for m in popular_movies if m not in rated_movies]

        predictions = [svd.predict(user_id, m) for m in unrated]
        predictions.sort(key=lambda x: x.est, reverse=True)
        top_ids = [p.iid for p in predictions[:10]]

        top_movies = movies[movies['movieId'].isin(top_ids)]
        display_movies(top_movies)