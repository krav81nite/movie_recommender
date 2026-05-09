FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

RUN python -c "
from huggingface_hub import hf_hub_download
files = ['ratings_25m_clean.parquet', 'movies_enriched.parquet', 'svd_25m.pkl', 'tags.csv']
for f in files:
    print(f'Downloading {f}...')
    hf_hub_download(repo_id='krav0x/movie-recommender', filename=f, repo_type='dataset', local_dir='.')
    print(f'{f} done!')
"

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]