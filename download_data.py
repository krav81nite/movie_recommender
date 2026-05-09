from huggingface_hub import hf_hub_download

files = ['ratings_25m_clean.parquet', 'movies_enriched.parquet', 'svd_25m.pkl', 'tags.csv']
for f in files:
    print(f'Downloading {f}...')
    hf_hub_download(repo_id='krav0x/movie-recommender', filename=f, repo_type='dataset', local_dir='.')
    print(f'{f} done!')