"""
Evaluate and compare Word2Vec-LL vs Word2Vec-LLS models.

This script:
1. Loads both models (base Word2Vec and enhanced LLS)
2. Evaluates them against a golden standard dataset
3. Computes Precision, Recall, and F1-score at various k values
4. Prints a comparison table of results

The goal is to test whether combining distributional similarity
(Word2Vec) with knowledge-based similarity (WordNet) improves
word similarity prediction performance.
"""

import os
import argparse

import numpy as np
from gensim.models import Word2Vec
import joblib

from train_word2vecLLS import (
    create_similarity_matrix,
    normalize_matrix,
    create_value_matrix,
    load_golden_standard,
)


def evaluate_model(model_func, golden_standard, k_values=None):
    """Evaluate a similarity model against a golden standard.

    For each concept in the golden standard, retrieves the top-k
    most similar words and computes precision, recall, and F1.

    Args:
        model_func: Function that takes (word, topn) and returns
                    list of (word, score) tuples.
        golden_standard: Dictionary of concept -> related words.
        k_values: List of k values to evaluate at.

    Returns:
        Tuple of (results_dict, skipped_concepts_list).
    """
    if k_values is None:
        k_values = [20, 50, 100, 200]

    results = {}
    skipped_concepts = []

    for k in k_values:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for concept, related_words in golden_standard.items():
            try:
                predicted_similar = [w for w, _ in model_func(concept, topn=k)]
            except KeyError:
                if concept not in skipped_concepts:
                    skipped_concepts.append(concept)
                continue

            tp = len(set(related_words) & set(predicted_similar))
            true_positives += tp
            false_positives += len(predicted_similar) - tp
            false_negatives += len(set(related_words) - set(predicted_similar))

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results[k] = {"precision": precision, "recall": recall, "f1": f1}

    return results, skipped_concepts


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Word2Vec-LL vs Word2Vec-LLS models"
    )
    parser.add_argument(
        "--w2v-model",
        default="models/word2vecLL_quran.model",
        help="Path to Word2Vec-LL model (default: models/word2vecLL_quran.model)",
    )
    parser.add_argument(
        "--lls-model",
        default="models/word2vecLLS_quran.model",
        help="Path to Word2Vec-LLS model (default: models/word2vecLLS_quran.model)",
    )
    parser.add_argument(
        "--golden-standard",
        default="data/5w100c.txt",
        help="Path to golden standard file (default: data/5w100c.txt)",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[20, 50, 100, 200],
        help="K values for evaluation (default: 20 50 100 200)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Model Evaluation: Word2Vec-LL vs Word2Vec-LLS")
    print("=" * 60)

    # Load models
    print(f"\n[1/3] Loading models...")
    if not os.path.exists(args.w2v_model):
        print(f"  ERROR: Word2Vec model not found at {args.w2v_model}")
        print("  Please run train_word2vec.py first.")
        return
    word2vecLL = Word2Vec.load(args.w2v_model)
    print(f"  Word2Vec-LL loaded (vocab size: {len(word2vecLL.wv.key_to_index)})")

    if not os.path.exists(args.lls_model):
        print(f"  ERROR: LLS model not found at {args.lls_model}")
        print("  Please run train_word2vecLLS.py first.")
        return
    word2vecLLS = joblib.load(args.lls_model)
    print(f"  Word2Vec-LLS loaded")

    # Load golden standard
    print(f"\n[2/3] Loading golden standard from: {args.golden_standard}")
    if not os.path.exists(args.golden_standard):
        print(f"  ERROR: Golden standard not found at {args.golden_standard}")
        return
    golden_standard = load_golden_standard(args.golden_standard)
    print(f"  Concepts loaded: {len(golden_standard)}")

    # Evaluate Word2Vec-LL
    print(f"\n[3/3] Evaluating models...")
    print("\n  Evaluating Word2Vec-LL...")
    results_LL, skipped_LL = evaluate_model(
        word2vecLL.wv.most_similar, golden_standard, args.k_values
    )
    print(f"  Skipped concepts ({len(skipped_LL)}): {skipped_LL}")

    # Evaluate Word2Vec-LLS
    print("\n  Evaluating Word2Vec-LLS...")

    def word2vecLLS_most_similar(word, topn):
        """Get top-N similar words using the LLS re-ranking model."""
        # Get candidates from base Word2Vec (fetch more than needed for re-ranking)
        similar_words = [w for w, _ in word2vecLL.wv.most_similar(word, topn=topn)]
        M = create_similarity_matrix(word, similar_words)
        SM = normalize_matrix(M)
        D = word2vecLL.wv.distances(word, similar_words)
        V = create_value_matrix(D, SM)
        similarities = word2vecLLS.predict(V)
        return sorted(zip(similar_words, similarities), key=lambda x: x[1], reverse=True)[:topn]

    results_LLS, skipped_LLS = evaluate_model(
        word2vecLLS_most_similar, golden_standard, args.k_values
    )
    print(f"  Skipped concepts ({len(skipped_LLS)}): {skipped_LLS}")

    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"\n{'k':<6}{'Model':<16}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 58)

    for k in args.k_values:
        print(
            f"{k:<6}{'Word2Vec-LL':<16}"
            f"{results_LL[k]['precision']:<12.4f}"
            f"{results_LL[k]['recall']:<12.4f}"
            f"{results_LL[k]['f1']:<12.4f}"
        )
        print(
            f"{'':6}{'Word2Vec-LLS':<16}"
            f"{results_LLS[k]['precision']:<12.4f}"
            f"{results_LLS[k]['recall']:<12.4f}"
            f"{results_LLS[k]['f1']:<12.4f}"
        )
        print("-" * 58)

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
