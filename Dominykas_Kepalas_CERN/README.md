Spotify Song Recommender
========================

CLI tool to explore a Spotify song dataset: get similar-track recommendations based on audio features or list the most popular songs by genre, plus optional visualizations (radar plots, feature histograms, heatmaps, UMAP).

Table of contents
-----------------
- Features
- Project layout
- Setup
- Running the CLI
- Data and cleaning
- Visualizations
- How recommendations work
- Notes and tips

Features
--------
- Similarity search: enter `Artist - Title` to get top similar songs (cosine KNN on audio features), with optional same-genre preference and a radar plot comparison.
- Popularity by genre: list the most popular tracks within a chosen genre.
- Visual reports: histograms, correlation heatmap, and optional 3D UMAP projection of songs.

Project layout
--------------
- `main.py`: entrypoint that launches the CLI.
- `src/cli.py`: interactive loop for similarity search or genre popularity.
- `src/data_loader.py`: CSV path helpers and data loading.
- `src/features.py`: feature scaling and weighting for the model.
- `src/model.py`: builds a cosine KNN model.
- `src/similarity.py`: parsing, song lookup (exact + fuzzy), nearest-neighbor recommendations, and artist-specific picks.
- `src/popularity.py`: top-popular-by-genre helper.
- `visualization/plotting.py`: radar plots, histograms, correlation heatmap, and UMAP projections.
- `Showcase.py`: quick script to generate the visualizations.
- `data_cleaning.py`: one-off cleaner to produce `data/spotify_song_features_clean.csv` from the raw CSV.
- `data/spotify_song_features_clean.csv`: cleaned dataset consumed by the app.

Setup
-----
1) Create/activate a Python 3.10+ environment.
2) Install dependencies:
   ```
   pip install pandas scikit-learn rapidfuzz matplotlib umap-learn
   ```
3) Ensure `data/spotify_song_features_clean.csv` exists. If you have the raw CSV (`data/spotify_song_features.csv`), run:
   ```
   python data_cleaning.py
   ```

Running the CLI
---------------
```
python main.py
```
- Mode 1: enter a song as `Artist - Title`, choose whether to stick to the same genre, and optionally plot a radar comparison with the top match.
- Mode 2: pick a genre to see the most popular tracks in that genre.
- Type `exit` to quit at any prompt.

Data and cleaning
-----------------
- `data_cleaning.py` drops unused columns from the raw Spotify features CSV and writes `data/spotify_song_features_clean.csv`.
- If `instrumentalness` (or other columns) are missing in the cleaned file, visualizations now adapt by skipping absent columns.

Visualizations
--------------
- Radar plots: compare input vs. top rec (`visualization.plotting.plot_comparison_radar`), or single-song radar via `plot_song_radar`.
- Histograms and heatmap: run `python Showcase.py` to output `energy_histogram.png`, `valence_histogram.png`, and `feature_correlation_heatmap.png`.
- UMAP 3D: set `RUN_3D_UMAP = True` in `Showcase.py` to produce `umap_spotify_songs_3d.png`.

How recommendations work
------------------------
- Features: numeric audio features are standardized and weighted (`src/features.py`), excluding popularity (kept for display only).
- Model: cosine `NearestNeighbors` over the scaled/weighted features (`src/model.py`).
- Song lookup: exact match on artist/title first; otherwise a single fuzzy match on `artist - title` strings via `rapidfuzz` (`src/similarity.py`).
- Recommendations: query nearest neighbors, drop duplicates, optionally re-rank within the same genre using cosine similarity. Also returns up to three more tracks from the same artist (deduped).
- Popularity: filter songs by genre (case-insensitive), sort by Spotify popularity, drop duplicate artist/track (`src/popularity.py`).

Notes and tips
--------------
- Input format for similarity mode: `Artist - Song Title` (dash required).
- The radar plot scales popularity to 0–1 for visualization only.
- If you adjust feature weights or columns, keep `feature_cols` and `feature_weights` in sync (`src/features.py`).