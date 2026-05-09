from huggingface_hub import hf_hub_download

files = ['ratings_small.parquet', 'movies_enriched.parquet', 'svd_small.pkl', 'cosine_sim.pkl']
for f in files:
    print(f'Downloading {f}...')
    hf_hub_download(repo_id='krav0x/movie-recommender', filename=f, repo_type='dataset', local_dir='.')
    print(f'{f} done!')