import pandas as pd
from src.features import scale_features
from visualization.plotting import plot_umap_3d
from visualization.plotting import plot_feature_histograms
from visualization.plotting import plot_feature_correlation_heatmap

DATA_PATH = "data/spotify_song_features_clean.csv"

RUN_3D_UMAP = False  # Set to TRUE to enable 3D UMAP plotting


def main():
    df = pd.read_csv(DATA_PATH)

    # Reuse your existing feature pipeline
    _, X_scaled = scale_features(df)

    plot_feature_histograms(df)
    plot_feature_correlation_heatmap(df)

    if RUN_3D_UMAP:
        plot_umap_3d(X_scaled)


if __name__ == "__main__":
    main()
