from sklearn.neighbors import NearestNeighbors


def build_knn():
    return NearestNeighbors(metric="cosine", n_neighbors=6) # Neighbors = 6 to ignore the first one (the song itself)
