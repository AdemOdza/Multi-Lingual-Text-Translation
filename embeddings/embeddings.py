# FastText Aligned Embedding helper funcitons
# Download vector files from: https://fasttext.cc/docs/en/crawl-vectors.html

from pathlib import Path
import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple

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

            parts = line.rstrip().split()
            if len(parts) < 2:
                continue

            word = parts[0]
            vals = parts[1:]

            if expected_dim is None:
                expected_dim = len(vals)
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
    if skipped:
        print(f"Skipped {skipped} invalid lines while reading {path.name}")
    if len(vecs) == 0:
        return [], np.zeros((0, 0), dtype=float)
    return words, np.vstack(vecs)

# Example code, uses the Spanish and English vector files.
if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]

    es_path = repo_root / "embeddings" / "wiki.es.align.vec"
    en_path = repo_root / "embeddings" / "wiki.en.align.vec"

    if not es_path.exists() or not en_path.exists():
        print("Place wiki.es.align.vec and wiki.en.align.vec under embeddings/ to run this example.")
        exit()

    print("Loading Spanish vectors (subset)...")
    es_words, es_vecs = load_vecs(es_path, limit=5000)
    print("Loading English vectors (subset)...")
    en_words, en_vecs = load_vecs(en_path, limit=5000)

    # build a word to index map for English
    en_index = {w: i for i, w in enumerate(en_words)}

    # Train the nearest neighbors on english embeddings
    nn = NearestNeighbors(n_neighbors=20, metric="cosine", algorithm="brute")
    nn.fit(en_vecs)

    spanish_examples = ["hola", "mundo", "gracias", "amor", "escuela", "zapato", "felicidad", "mañana", "mujer", "niños"]

    # Go through each spanish example word and find the top 5 most similar words
    for s in spanish_examples:
        if s in es_words:
            # Get the index and embedding for the spanish word
            si = es_words.index(s)
            q = es_vecs[si]

            # Find the most similar vectors to q
            distances, indices = nn.kneighbors(q.reshape(1, -1), n_neighbors=5, return_distance=True)
            indices = indices[0]
            distances = distances[0]

            # Sklearn returns distance, subtract from 1 to get similarities
            sims = 1 - distances

            print(f"\nSpanish word: {s} (idx={si}) -> Top English neighbors:")
            for rank, (idx, sim) in enumerate(zip(indices, sims), start=1):
                print(f"{rank:2}. {en_words[int(idx)]:20s} sim={float(sim):.4f}")
        else:
            print(f"Spanish word '{s}' not found in the loaded Spanish subset")
