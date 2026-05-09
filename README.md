---
title: Movie Recommender
emoji: 🎬
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# 🎬 CineMatch — Movie Recommender System

A production-ready movie recommender system built with collaborative filtering and content-based filtering, deployed as an interactive web app with a Netflix-inspired design.

**[🚀 Live Demo](https://movierecommender-xaxykeuayaad9nt4rqsoge.streamlit.app/)**

---

## Overview

CineMatch explores three recommendation approaches on the MovieLens 25M dataset, enriched with TMDb metadata (posters, descriptions, ratings).

| Model | Approach | Precision@10 |
|-------|----------|-------------|
| SVD (default) | Collaborative filtering | 0.168 |
| SVD (tuned) | Collaborative filtering | 0.168 |
| Content-based | TF-IDF + cosine similarity | 0.027 |
| Hybrid | 70% CF + 30% content-based | 0.168 |

> Evaluation used a temporal train/test split — training on ratings before March 2016, testing on after.

---

## Features

- **Content-based filtering** — find movies similar to one you select, based on genres and user tags
- **Collaborative filtering** — personalized recommendations using SVD matrix factorization
- **Hybrid model** — weighted blend of both approaches
- **TMDb enrichment** — movie posters, overviews, and ratings from The Movie Database API
- **Interactive UI** — dark theme with hover tooltips showing full movie details

---

## Project Structure
movie-recommender/
├── app/
│   └── app.py                  # Streamlit app
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_collaborative_filtering.ipynb
│   ├── 03_content_based.ipynb
│   ├── 04_hybrid.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_tmdb_enrichment.ipynb
├── requirements.txt
└── README.md

---

## Stack

| Layer | Tool |
|-------|------|
| Collaborative filtering | scikit-surprise (SVD) |
| Content-based | scikit-learn (TF-IDF, cosine similarity) |
| Data | MovieLens 25M + TMDb API |
| App | Streamlit |
| Hosting | Streamlit Community Cloud |
| Data storage | Hugging Face Datasets |

---

## How to Run Locally

```bash
git clone https://github.com/krav81nite/movie_recommender.git
cd movie_recommender
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

---

## Key Design Decisions

- **Temporal split** over random split — prevents data leakage from future ratings
- **Precomputed cosine similarity** — stored as a pickle file for fast startup
- **Parquet format** — 6x smaller than CSV for faster loading
- **Popular movies filter** — SVD predictions limited to top 5000 movies for speed
- **Hybrid model** — content-based adds explainability even when CF dominates metrics