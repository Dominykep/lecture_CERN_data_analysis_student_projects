from pathlib import Path
import pandas as pd


def default_data_path():
    base_dir = Path(__file__).resolve().parent.parent
    return base_dir / "data" / "spotify_song_features_clean.csv"


def load_songs(path=None):
    csv_path = Path(path) if path else default_data_path()
    return pd.read_csv(csv_path)


def top_genres(df, limit=12):
    return df["genre"].value_counts().head(limit).index.tolist()
