def get_top_popular_by_genre(df, genre, top_n=10):
    if genre is None or not str(genre).strip():
        raise ValueError("Genre cannot be empty.")

    genre_normalized = str(genre).strip().lower()
    genre_df = df.loc[df["genre"].str.lower() == genre_normalized, ["artist_name", "track_name", "popularity"]]

    if genre_df.empty:
        raise ValueError("Genre not found in dataset.")

    genre_df = genre_df.sort_values("popularity", ascending=False)
    genre_df = genre_df.drop_duplicates(subset=["artist_name", "track_name"])
    return genre_df.head(top_n)
