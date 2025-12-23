import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.metrics.pairwise import cosine_similarity


def parse_input(text):
    if "-" not in text:
        raise ValueError("Use 'Artist - Song title'")
    artist, title = text.split("-", 1)
    return artist.strip().lower(), title.strip().lower()


def get_song_index(df, artist, title):
    # return row index for the requested song using a simple exact or fuzzy match.
    exact = df[
        (df["artist_name"].str.lower() == artist)
        & (df["track_name"].str.lower() == title)
    ]
    if not exact.empty:
        return exact.index[0]

    combined = df["artist_name"].str.lower() + " - " + df["track_name"].str.lower()
    match = process.extractOne(
        f"{artist} - {title}",
        combined,
        scorer=fuzz.token_sort_ratio,
        score_cutoff=70,
    )
    if match:
        _, _, idx = match
        return idx

    raise ValueError("Song not found")


def _filter_recs(recs, exclude_key=None):
    if not exclude_key:
        return recs
    artist, track = (exclude_key[0].lower(), exclude_key[1].lower())
    return recs[
        ~(
            (recs["artist_name"].str.lower() == artist)
            & (recs["track_name"].str.lower() == track)
        )
    ]


def recommend_songs(
    song_idx,
    df,
    knn_model,
    scaled_features,
    n=5,
    exclude_key=None,
    prefer_same_genre=False,
):
    k = min(len(df), n * 4 + 5)
    dist, idxs = knn_model.kneighbors(
        scaled_features[song_idx].reshape(1, -1),
        n_neighbors=k,
    )

    idxs = idxs[0][idxs[0] != song_idx]
    dist = dist[0][: len(idxs)]

    recs = df.iloc[idxs][["artist_name", "track_name", "popularity"]].copy()
    recs["similarity"] = 1 - dist
    recs = recs.drop_duplicates(subset=["artist_name", "track_name"])
    recs = _filter_recs(recs, exclude_key)

    target_vec = scaled_features[song_idx].reshape(1, -1)
    if prefer_same_genre:
        genre = str(df.iloc[song_idx].get("genre", "")).lower()
        if genre:
            genre_idx = df.index[df["genre"].str.lower() == genre].tolist()
            genre_idx = [idx for idx in genre_idx if idx != song_idx]
            if genre_idx:
                sims = cosine_similarity(target_vec, scaled_features[genre_idx])[0]
                recs = pd.DataFrame(
                    {
                        "artist_name": df.loc[genre_idx, "artist_name"].values,
                        "track_name": df.loc[genre_idx, "track_name"].values,
                        "popularity": df.loc[genre_idx, "popularity"].values,
                        "similarity": sims,
                    }
                )
                recs = recs.drop_duplicates(subset=["artist_name", "track_name"])
                recs = _filter_recs(recs, exclude_key)

    recs = recs.sort_values("similarity", ascending=False)
    general_df = recs.head(n).reset_index(drop=True)

    target_artist = df.iloc[song_idx]["artist_name"].lower()
    artist_df = (
        recs[recs["artist_name"].str.lower() == target_artist]
        .drop_duplicates(subset=["artist_name", "track_name"])
        .head(3)
    )

    if artist_df.empty:
        artist_idx = df.index[df["artist_name"].str.lower() == target_artist].tolist()
        artist_idx = [idx for idx in artist_idx if idx != song_idx]
        if artist_idx:
            sims = cosine_similarity(target_vec, scaled_features[artist_idx])[0]
            fallback = pd.DataFrame(
                {
                    "artist_name": df.loc[artist_idx, "artist_name"].values,
                    "track_name": df.loc[artist_idx, "track_name"].values,
                    "popularity": df.loc[artist_idx, "popularity"].values,
                    "similarity": sims,
                }
            )
            fallback = _filter_recs(fallback, exclude_key)
            artist_df = (
                fallback.sort_values("similarity", ascending=False)
                .drop_duplicates(subset=["artist_name", "track_name"])
                .head(3)
            )

    return general_df, artist_df.reset_index(drop=True)


def recommend_from_text(user_input, df, knn_model, scaled_features, n=5, prefer_same_genre=False):
    artist, title = parse_input(user_input)
    song_idx = get_song_index(df, artist, title)
    resolved_artist = df.iloc[song_idx]["artist_name"].lower()
    resolved_title = df.iloc[song_idx]["track_name"].lower()
    general_df, artist_df = recommend_songs(
        song_idx,
        df,
        knn_model,
        scaled_features,
        n,
        exclude_key=(resolved_artist, resolved_title),
        prefer_same_genre=prefer_same_genre,
    )
    return general_df, artist_df
