from sklearn.preprocessing import StandardScaler

# audio feature setup (popularity intentionally excluded)
feature_cols = [
    "acousticness",
    "danceability",
    "energy",
    "valence",
    "speechiness",
    "loudness",
    "tempo",
]

feature_weights = {
    "acousticness": 0.8,
    "danceability": 1.0,
    "energy": 2.0,
    "valence": 1.3,
    "speechiness": 0.5,
    "loudness": 1.5,
    "tempo": 1.5,
}


def scale_features(df):
    feats = df[feature_cols]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feats)
    for i, col in enumerate(feature_cols):
        scaled[:, i] *= feature_weights.get(col, 1.0)
    return scaler, scaled
