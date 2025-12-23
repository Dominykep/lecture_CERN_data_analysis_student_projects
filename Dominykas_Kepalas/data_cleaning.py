import pandas as pd

data = "data/spotify_song_features.csv"
data_cleaned = "data/spotify_song_features_clean.csv"
drop_columns = ["time_signature", "liveness", "mode", "duration_ms", "track_id", "instrumentalness"]

df = pd.read_csv(data, encoding="latin-1", low_memory=False)

print("=== BEFORE DROPPING COLUMNS ===")
print(df.head(5))
print("Columns:", list(df.columns))

cols_to_drop = []
for col in drop_columns:
    if col in df.columns:
        cols_to_drop.append(col)

df = df.drop(columns=cols_to_drop)

print("\n=== AFTER DROPPING COLUMNS ===")
print(df.head(5))
print("Columns:", list(df.columns))

df.to_csv(data_cleaned, index=False, encoding="latin-1")
print(f"\nsaved as {data_cleaned}")
