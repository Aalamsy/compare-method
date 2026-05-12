"""
Train the Word2Vec-LLS (Lexical Linguistic Similarity) model.

This script enhances Word2Vec with linguistic knowledge by:
1. Loading a pre-trained Word2Vec model (word2vecLL)
2. Loading a golden standard dataset (concept-word mappings)
3. Computing similarity matrices using:
   - Wu-Palmer similarity (WordNet)
   - Jiang-Conrath similarity (WordNet + IC)
   - Hirst-St-Onge similarity (WordNet)
   - Word2Vec cosine distance
4. Training a neural network (MLP) to learn optimal similarity rankings
5. Saving the trained MLP model as word2vecLLS

The hypothesis is that combining distributional (Word2Vec) and
knowledge-based (WordNet) similarity measures yields better
word similarity predictions.
"""

import os
import argparse

import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import nltk
import joblib

# Ensure required NLTK data is available
nltk.download("wordnet", quiet=True)
nltk.download("wordnet_ic", quiet=True)

# Load information content for Jiang-Conrath similarity
brown_ic = wordnet_ic.ic("ic-brown.dat")


def safe_similarity(func):
    """Decorator to handle exceptions in similarity computations."""
    def wrapper(word1, word2):
        try:
            return func(word1, word2)
        except Exception:
            return 0.0
    return wrapper


@safe_similarity
def wu_palmer_similarity(word1, word2):
    """Compute Wu-Palmer similarity between two words using WordNet.

    Measures similarity based on depth of the two synsets in the
    taxonomy and the depth of their Least Common Subsumer (LCS).
    """
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if synsets1 and synsets2:
        max_sim = max(
            (s1.wup_similarity(s2) or 0) for s1 in synsets1 for s2 in synsets2
        )
        return max_sim if max_sim > 0 else 0.0
    return 0.0


@safe_similarity
def jiang_conrath_similarity(word1, word2):
    """Compute Jiang-Conrath similarity between two words.

    Uses information content from the Brown corpus to measure
    semantic distance.
    """
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if synsets1 and synsets2:
        max_sim = max(
            (s1.jcn_similarity(s2, brown_ic) or 0)
            for s1 in synsets1
            for s2 in synsets2
        )
        return max_sim if max_sim > 0 else 0.0
    return 0.0


@safe_similarity
def hirst_st_onge_similarity(word1, word2):
    """Compute Hirst-St-Onge similarity between two words.

    Based on path length between synsets and direction changes.
    Note: NLTK may not fully support this; falls back to path similarity.
    """
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if synsets1 and synsets2:
        # NLTK doesn't have native HSO; use path_similarity as proxy
        max_sim = max(
            (s1.path_similarity(s2) or 0) for s1 in synsets1 for s2 in synsets2
        )
        return max_sim if max_sim > 0 else 0.0
    return 0.0


def create_similarity_matrix(word, similar_words):
    """Create a 4xN matrix of similarity features.

    Rows:
        0 - Wu-Palmer similarity
        1 - Jiang-Conrath similarity
        2 - Hirst-St-Onge similarity (path similarity proxy)
        3 - Bias term (all 1s)

    Args:
        word: The target word.
        similar_words: List of candidate similar words.

    Returns:
        numpy array of shape (4, len(similar_words))
    """
    M = np.zeros((4, len(similar_words)))
    for i, similar_word in enumerate(similar_words):
        M[0, i] = wu_palmer_similarity(word, similar_word)
        M[1, i] = jiang_conrath_similarity(word, similar_word)
        M[2, i] = hirst_st_onge_similarity(word, similar_word)
        M[3, i] = 1  # bias term
    return M


def normalize_matrix(M):
    """Normalize matrix rows to [0, 1] range using min-max normalization.

    Args:
        M: Input matrix.

    Returns:
        Normalized matrix.
    """
    min_vals = M.min(axis=1, keepdims=True)
    max_vals = M.max(axis=1, keepdims=True)
    return (M - min_vals) / (max_vals - min_vals + 1e-10)


def create_value_matrix(distances, similarity_matrix):
    """Combine Word2Vec distances with normalized similarity features.

    Args:
        distances: Word2Vec cosine distances (1D array).
        similarity_matrix: Normalized similarity matrix (4xN).

    Returns:
        Combined feature matrix of shape (N, 5).
    """
    return np.vstack((distances, similarity_matrix)).T


def load_golden_standard(file_path):
    """Load golden standard concept-word mappings.

    Expected file format (one per line):
        concept: word1, word2, word3, word4, word5

    Args:
        file_path: Path to the golden standard file.

    Returns:
        Dictionary mapping concepts to lists of related words.
    """
    golden_standard = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(":")
            if len(parts) == 2:
                concept = parts[0].strip()
                words = [w.strip() for w in parts[1].split(",")][:5]
                golden_standard[concept] = words
    return golden_standard


def train_word2vecLLS(word2vecLL, golden_standard, max_iter=100, topn=20):
    """Train the Word2Vec-LLS neural network model.

    Uses an MLP regressor to learn the relationship between
    combined similarity features and gold-standard relevance labels.

    Args:
        word2vecLL: Pre-trained Word2Vec model.
        golden_standard: Dictionary of concept -> related words.
        max_iter: Maximum training iterations.
        topn: Number of similar words to consider per concept.

    Returns:
        Trained MLPRegressor model.
    """
    nn_model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        max_iter=1,
        warm_start=True,
        random_state=42,
    )

    # Build training data
    X_train = []
    y_train = []
    skipped = 0

    for concept, related_words in golden_standard.items():
        if concept not in word2vecLL.wv:
            skipped += 1
            continue

        similar_words = [w for w, _ in word2vecLL.wv.most_similar(concept, topn=topn)]
        M = create_similarity_matrix(concept, similar_words)
        SM = normalize_matrix(M)
        D = word2vecLL.wv.distances(concept, similar_words)
        V = create_value_matrix(D, SM)
        X_train.extend(V)
        y_train.extend([1 if word in related_words else 0 for word in similar_words])

    if not X_train:
        print("  ERROR: No training data could be generated.")
        return None

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    print(f"  Training samples: {len(X_train)}")
    print(f"  Concepts skipped (not in vocab): {skipped}")
    print(f"  Training for up to {max_iter} iterations...")
    print()

    for i in range(max_iter):
        nn_model.fit(X_train, y_train)
        y_pred = nn_model.predict(X_train)
        error = mean_squared_error(y_train, y_pred)

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Iteration {i + 1:3d}, MSE: {error:.6f}")

        if error < 0.01:
            print(f"  Converged at iteration {i + 1} (MSE < 0.01)")
            break

    return nn_model


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec-LLS model (enhanced with linguistic similarity)"
    )
    parser.add_argument(
        "--model-path",
        default="models/word2vecLL_quran.model",
        help="Path to pre-trained Word2Vec model (default: models/word2vecLL_quran.model)",
    )
    parser.add_argument(
        "--golden-standard",
        default="data/5w100c.txt",
        help="Path to golden standard file (default: data/5w100c.txt)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save the trained LLS model (default: models)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=100, help="Max training iterations (default: 100)"
    )
    parser.add_argument(
        "--topn", type=int, default=20, help="Top-N similar words per concept (default: 20)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Word2Vec-LLS Training (Lexical Linguistic Similarity)")
    print("=" * 60)

    # Step 1: Load Word2Vec model
    print(f"\n[1/3] Loading Word2Vec model from: {args.model_path}")
    if not os.path.exists(args.model_path):
        print(f"  ERROR: Model not found at {args.model_path}")
        print("  Please run train_word2vec.py first.")
        return
    word2vecLL = Word2Vec.load(args.model_path)
    print(f"  Vocabulary size: {len(word2vecLL.wv.key_to_index)}")

    # Step 2: Load golden standard
    print(f"\n[2/3] Loading golden standard from: {args.golden_standard}")
    if not os.path.exists(args.golden_standard):
        print(f"  ERROR: Golden standard not found at {args.golden_standard}")
        return
    golden_standard = load_golden_standard(args.golden_standard)
    print(f"  Concepts loaded: {len(golden_standard)}")

    # Step 3: Train model
    print(f"\n[3/3] Training Word2Vec-LLS model...")
    word2vecLLS = train_word2vecLLS(word2vecLL, golden_standard, max_iter=args.max_iter, topn=args.topn)

    if word2vecLLS is None:
        print("\n  Training failed.")
        return

    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "word2vecLLS_quran.model")
    joblib.dump(word2vecLLS, output_path)
    print(f"\n  Model saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
