# Embeddings
# FastText Aligned Embedding helper funcitons
# Download aligned vector files from: https://fasttext.cc/docs/en/aligned-vectors.html
from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Tuple


"""
load_vecs loads word vectors from the FastText aligned vector files

This function takes the vector file and parses a tuple that associates words
to their vectors

Parameters:
path: pathlib.Path
  Path to the FastText aligned vector file
Limit: Optional int (Default 50000)
  Maximum number of word vectors to load

Returns a tuple
tuple[0]: List of the vocab words in the embedding
tuple[1]: Numpy matrix, each column is a word embedding vector
"""

def load_vecs(path: Path, limit: int = 50000) -> Tuple[List[str], np.ndarray]:
    words = []
    vecs = []
    expected_dim = None
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for ix, line in enumerate(f):
            if ix == 0:
                # If header present, parse for dimensions
                header = line.strip().split()
                if len(header) >= 2 and header[0].isdigit() and header[1].isdigit():
                    expected_dim = int(header[1])
                continue

            # Split line into word and vector
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue

            word = parts[0]
            vals = parts[1:]

            # Use default dimensionality if no header
            if expected_dim is None:
                expected_dim = len(vals)

            # Skip all vectors that do not match the expected dimensionality
            if len(vals) != expected_dim:
                skipped += 1
                continue

            try:
                arr = np.array(vals, dtype=float)
            except Exception:
                skipped += 1
                continue

            words.append(word)
            vecs.append(arr)
            if len(words) >= limit:
                break

    # Debug, show amount of invalid lines
    if skipped:
        print(f"Skipped {skipped} invalid lines while reading {path.name}")

    if len(vecs) == 0:
        return [], np.zeros((0, 0), dtype=float)

    # Return list of words as well as all vectors as one matrix
    return words, np.vstack(vecs)

# Helper function to pull the top 5 similar words for a word
  """
  cross_lingual_neighbors finds k nearest-neighbor translations between two aligned FastText vector sets.

  Parameters
  src_path: Path
      Path to the embedding file of the language we want to translate (e.g., Spanish .vec)
  tgt_path: Path
      Path to the embedding file of the language we want to translate to (e.g., English .vec)
  src_words_to_query: list of str
      Words from the source language whose neighbors should be retrieved
  limit: optional int (Default 50000)
      Maximum number of vectors to load from each embedding file
  neighbors: optional int (Default 5)
      Number of neighbors for KNN
  debug: optional bool (Default true)
      Prints details of the KNN process

  Returns a dictionary mapping each source word to a list of (target_word, similarity)
    tuples sorted by highest similarity first.
      Unknown Words will map to an empty list.
  """
  def cross_lingual_neighbors(
    src_path: Path,
    tgt_path: Path,
    src_words_to_query: List[str],
    limit: int = 50000,
    neighbors: int = 5,
    debug: bool = True,
) -> Dict[str, List[Tuple[str, float]]]:

    if debug:
        print(f"Loading source embeddings: {src_path.name}...")
    src_words, src_vecs = load_vecs(src_path, limit=limit)

    if debug:
        print(f"Loading target embeddings: {tgt_path.name}...")
    tgt_words, tgt_vecs = load_vecs(tgt_path, limit=limit)

    # Build index for fast lookups
    src_index = {w: i for i, w in enumerate(src_words)}
    tgt_index = {w: i for i, w in enumerate(tgt_words)}

    # Fit K nearest-neighbor model on target embeddings
    nn = NearestNeighbors(n_neighbors=neighbors, metric="cosine", algorithm="brute")
    nn.fit(tgt_vecs)

    results: Dict[str, List[Tuple[str, float]]] = {}

    for word in src_words_to_query:
        if word not in src_index:
            if debug:
                print(f"Source word '{word}' not found in the embedding subset.")
            results[word] = []
            continue

        # Query vector
        vec = src_vecs[src_index[word]]

        # Nearest neighbors
        distances, indices = nn.kneighbors(vec.reshape(1, -1), n_neighbors=neighbors)
        distances = distances[0]
        indices = indices[0]

        # Convert cosine distance → cosine similarity
        sims = 1 - distances

        neighbors_list = [
            (tgt_words[int(idx)], float(sim)) for idx, sim in zip(indices, sims)
        ]
        results[word] = neighbors_list

        if debug:
            print(f"\nSource word: {word}")
            for rank, (tgt_word, sim) in enumerate(neighbors_list, 1):
                print(f"{rank:2}. {tgt_word:20s} sim={sim:.4f}")

    return results

# """
# examples = ["hola", "mundo", "gracias", "amor", "escuela"]

# results = cross_lingual_neighbors(
#     src_path=(spanish file path),
#     tgt_path=(english file path),
#     src_words_to_query=examples,
#     limit=50000,
#     neighbors=5,
#     debug=True,
# )
# """
