import numpy as np
import matplotlib.pyplot as plt
import umap
from mpl_toolkits.mplot3d import Axes3D 

# five radar features (0-1); popularity is scaled to 0-1 for plotting
RADAR_FEATURES = [
    "acousticness",
    "danceability",
    "energy",
    "valence",
    "popularity",
]


def plot_song_radar(df, song_index):
    _plot_radar(df, primary_idx=song_index)


def plot_comparison_radar(df, base_idx, rec_idx):
    artist_a = df.loc[base_idx, "artist_name"]
    track_a = df.loc[base_idx, "track_name"]
    artist_b = df.loc[rec_idx, "artist_name"]
    track_b = df.loc[rec_idx, "track_name"]

    _plot_radar(
        df,
        primary_idx=base_idx,
        secondary_idx=rec_idx,
        title=rf"{artist_a} - {track_a}  $\mathit{{vs}}$  {artist_b} - {track_b}"
    )


def _prep_vals(df, idx):
    vals = df.loc[idx, RADAR_FEATURES].astype(float).values
    pop_idx = RADAR_FEATURES.index("popularity")
    vals[pop_idx] = min(max(vals[pop_idx] / 100.0, 0.0), 1.0)
    return vals


def _plot_radar(df, primary_idx, secondary_idx=None, title=None):
    vals_primary = _prep_vals(df, primary_idx)
    vals_secondary = _prep_vals(df, secondary_idx) if secondary_idx is not None else None

    angles = np.linspace(0, 2 * np.pi, len(RADAR_FEATURES), endpoint=False)
    plot_angles = np.concatenate([angles, [angles[0]]])

    def close(vals):
        return np.concatenate([vals, [vals[0]]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(plot_angles, close(vals_primary), linewidth=2, label="Input song")
    ax.fill(plot_angles, close(vals_primary), alpha=0.2)

    if vals_secondary is not None:
        ax.plot(plot_angles, close(vals_secondary), linewidth=2, label="Most similar song")
        ax.fill(plot_angles, close(vals_secondary), alpha=0.2)

    ax.set_ylim(0, 1)
    ax.set_xticks(angles)
    ax.set_xticklabels([])

    label_radius = 1.05
    for angle, label in zip(angles, RADAR_FEATURES):
        ha = "left" if np.cos(angle) >= 0 else "right"
        ax.text(angle, label_radius, label.upper(), ha=ha, va="center", rotation=0, rotation_mode="anchor")

    if title:
        ax.set_title(title, pad=35, fontsize=16, fontweight="bold")
    else:
        artist = df.loc[primary_idx, "artist_name"]
        track = df.loc[primary_idx, "track_name"]
        ax.set_title(f"{artist} - {track}", pad=30, fontsize=16, fontweight="bold")

    if vals_secondary is not None:
        ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0.1))

    plt.tight_layout()
    plt.show()

# UMAP PLOTTING
def plot_umap_3d(X_scaled):

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=3,
        random_state=42,
    )

    embedding = reducer.fit_transform(X_scaled)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        embedding[:, 2],
        s=6,
        alpha=0.6,
        c=embedding[:, 0],
        cmap="magma",
    )
    # To switch this to a 2D UMAP: change n_components from 3 to 2, remove (projection="3d") and plot only embedding[:, 0] and embedding[:, 1] with plt.scatter


    fig.colorbar(scatter, ax=ax, label="UMAP-1 value")

    ax.set_title("3D UMAP Projection of Spotify Songs")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")

    plt.tight_layout()
    plt.savefig("umap_spotify_songs_3d.png", dpi=300, bbox_inches="tight")
    plt.show()
    
# Histogram Plotting

def plot_feature_histograms(df):

    features = ["energy", "valence"]

    for feature in features:
        plt.figure(figsize=(6, 4))

        plt.hist(
            df[feature],
            bins=30,
            color='blue',
            edgecolor="black",    
            linewidth=0.8,
            alpha=0.85,
            zorder=3
        )

        plt.title(f"{feature.capitalize()} Distribution", fontsize=13, weight="bold")
        plt.xlabel(feature.capitalize(), fontsize=11)
        plt.ylabel("Number of Songs", fontsize=11)

        plt.grid(True, alpha=0.5, zorder=0)

        plt.tight_layout()
        plt.savefig(f"{feature}_histogram.png", dpi=300)
        plt.show()

# heatmap plotting

def plot_feature_correlation_heatmap(df):

    features = [
        "acousticness",
        "danceability",
        "energy",
        "valence",
        "instrumentalness",
        "speechiness",
        "loudness",
        "tempo",
    ]

    # only keep columns that exist to avoid KeyError on cleaned data
    used_features = [f for f in features if f in df.columns]
    if not used_features:
        raise ValueError("No matching audio feature columns found for correlation heatmap.")

    corr = df[used_features].corr()

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        corr,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        zorder=1,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation coefficient", fontsize=10)

    ax.set_xticks(range(len(used_features)))
    ax.set_yticks(range(len(used_features)))
    ax.set_xticklabels(used_features, rotation=90, ha="right")
    ax.set_yticklabels(used_features)

    ax.set_axisbelow(True)
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.5,
        alpha=0.4,
        zorder=0,
    )

    ax.set_title("Audio Feature Correlation Heatmap", fontsize=13, weight="bold")

    plt.tight_layout()
    plt.savefig("feature_correlation_heatmap.png", dpi=300)
    plt.show()
