# 🎬 Movie Recommender System

A production-ready movie recommender system built with collaborative filtering, content-based filtering, and a hybrid model. Deployed as an interactive web app with movie posters and hover tooltips.

**[🚀 Live Demo](https://movierecommender-xaxykeuayaad9nt4rqsoge.streamlit.app/)**

---

## Overview

This project explores three recommendation approaches on the MovieLens 25M dataset, enriched with TMDb metadata (posters, descriptions, ratings).

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
- **Interactive UI** — Netflix-style dark theme with hover tooltips showing full movie details

---

## Project Structure