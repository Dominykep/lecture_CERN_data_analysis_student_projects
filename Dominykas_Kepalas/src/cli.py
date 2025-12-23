from src.data_loader import load_songs, top_genres
from src.features import scale_features
from src.model import build_knn
from src.popularity import get_top_popular_by_genre
from src.similarity import get_song_index, parse_input, recommend_from_text
from visualization.plotting import plot_comparison_radar


def build_context():
    songs = load_songs()
    genres = top_genres(songs)
    _, scaled = scale_features(songs)
    knn = build_knn()
    knn.fit(scaled)
    return songs, genres, scaled, knn


def run_cli():
    songs, genres, scaled, knn = build_context()

    print("\nSong Recommender")
    print("PLease choose a mode:")
    print("  1) Similarity-based recommendations (enter as: Artist - Song title)")
    print("  2) Popular songs by genre (spotify popularity score)")
    print("Type 'exit' to quit.\n")

    while True:
        choice = input("Mode (1/2 or 'exit'): ").strip().lower()

        if choice == "exit":
            print("Goodbye")
            break

        if choice in {"1", "similarity", "s"}:
            user_input = input("Enter a song (Artist - Song title): ")
            if user_input.lower() == "exit":
                print("Goodbye")
                break

            try:
                artist, title = parse_input(user_input)
                song_idx = get_song_index(songs, artist, title)

                genre_pref = input("Limit to same genre? (y/n): ").strip().lower()
                prefer_same_genre = genre_pref == "y"

                general_recs, artist_recs = recommend_from_text(
                    user_input,
                    songs,
                    knn,
                    scaled,
                    n=5,
                    prefer_same_genre=prefer_same_genre,
                )

                print(f"\nRecommendations for: {user_input}\n")
                print("Top 5 similar songs:")
                for i, row in enumerate(general_recs.itertuples(), start=1):
                    print(f"{i}. {row.artist_name} - {row.track_name} (popularity: {row.popularity})")

                print("\nMore from this artist:")
                if artist_recs.empty:
                    print("No additional songs from this artist were found.")
                else:
                    for i, row in enumerate(artist_recs.itertuples(), start=1):
                        print(f"{i}. {row.artist_name} - {row.track_name} (popularity: {row.popularity})")

                print()

                if not general_recs.empty:
                    visualize_res = input("Visualize input vs. top similar song? (y/n): ").strip().lower()
                    if visualize_res == "y":
                        top_rec = general_recs.iloc[0]
                        mask = (songs["artist_name"] == top_rec.artist_name) & (songs["track_name"] == top_rec.track_name)
                        if mask.any():
                            rec_idx = songs.index[mask][0]
                            plot_comparison_radar(songs, song_idx, rec_idx)
                        else:
                            print("Could not locate top recommendation in dataset for plotting.")

            except ValueError as e:
                print(f"Error: {e}\n")

        elif choice in {"2", "popularity", "p"}:
            if genres:
                preview = ", ".join(genres)
                print(f"Popular genres (top {len(genres)}): {preview}")
            else:
                print("No genre list available.")
            genre_input = input("Enter genre: ")
            if genre_input.lower() == "exit":
                print("Goodbye")
                break

            try:
                top_popular = get_top_popular_by_genre(songs, genre_input, top_n=10)
                print(f"\nTop popular songs for genre '{genre_input.strip()}':")
                for i, row in enumerate(top_popular.itertuples(), start=1):
                    print(f"{i}. {row.artist_name} - {row.track_name} (popularity: {row.popularity})")
                print()
            except ValueError as e:
                print(f"Error: {e}\n")

        else:
            print("Invalid choice. Please select 1, 2, or 'exit'.\n")
