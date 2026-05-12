"""
Train a Word2Vec model (CBOW) on multiple English Quran translations.

This script:
1. Loads 17 English Quran translations from the data/quran/ directory
2. Preprocesses text (lowercasing, removing special characters, stopword removal)
3. Lemmatizes words using WordNet
4. Trains a Word2Vec model using CBOW (Continuous Bag of Words)
5. Saves the trained model to models/word2vecLL_quran.model
"""

import os
import re
import argparse

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from collections import Counter

# Ensure required NLTK data is available
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

# List of available Quran translations
TRANSLATIONS = [
    "en.ahmedali",
    "en.ahmedraza",
    "en.arberry",
    "en.daryabadi",
    "en.hilali",
    "en.itani",
    "en.maududi",
    "en.mubarakpuri",
    "en.pickthall",
    "en.qarai",
    "en.qaribullah",
    "en.sahih",
    "en.sarwar",
    "en.shakir",
    "en.transliteration",
    "en.wahiduddin",
    "en.yusufali",
]


def load_quran_texts(base_path):
    """Load all Quran translation files from the given directory.

    Args:
        base_path: Path to directory containing translation .txt files.

    Returns:
        List of lowercased text strings, one per translation file.
    """
    all_texts = []
    for translation in TRANSLATIONS:
        file_path = os.path.join(base_path, f"{translation}.txt")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                all_texts.append(text.lower())
            print(f"  Loaded: {translation}")
        except FileNotFoundError:
            print(f"  WARNING: File not found - {file_path}")
    return all_texts


def preprocess_text(text):
    """Remove non-alphabetical characters and stopwords.

    Args:
        text: Raw text string.

    Returns:
        List of filtered words.
    """
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    return words


def lemmatize_text(words):
    """Lemmatize a list of words using WordNet lemmatizer.

    Args:
        words: List of word strings.

    Returns:
        List of lemmatized words.
    """
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]


def get_vocabulary(texts):
    """Get word frequency counts across all texts.

    Args:
        texts: List of word lists.

    Returns:
        Counter object with word frequencies.
    """
    all_words = [word for text in texts for word in text]
    return Counter(all_words)


def print_vocabulary_stats(vocab, top_n=20):
    """Print vocabulary statistics.

    Args:
        vocab: Counter object with word frequencies.
        top_n: Number of top words to display.
    """
    print(f"\n  Total unique words: {len(vocab)}")
    print(f"  Top {top_n} most common words:")
    for word, count in vocab.most_common(top_n):
        print(f"    {word}: {count}")


def train_word2vec(texts, vector_size=200, window=10, min_count=5, workers=4):
    """Train a Word2Vec CBOW model.

    Args:
        texts: List of tokenized sentences (list of word lists).
        vector_size: Dimensionality of word vectors.
        window: Context window size.
        min_count: Minimum word frequency threshold.
        workers: Number of training threads.

    Returns:
        Trained Word2Vec model.
    """
    model = Word2Vec(
        sentences=texts,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=0,  # sg=0 for CBOW
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec model on Quran translations"
    )
    parser.add_argument(
        "--data-path",
        default="data/quran",
        help="Path to directory containing Quran translation files (default: data/quran)",
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save the trained model (default: models)",
    )
    parser.add_argument(
        "--vector-size", type=int, default=200, help="Word vector dimensionality (default: 200)"
    )
    parser.add_argument(
        "--window", type=int, default=10, help="Context window size (default: 10)"
    )
    parser.add_argument(
        "--min-count", type=int, default=5, help="Minimum word frequency (default: 5)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Word2Vec (CBOW) Training on Quran Translations")
    print("=" * 60)

    # Step 1: Load texts
    print("\n[1/4] Loading Quran translations...")
    quran_texts = load_quran_texts(args.data_path)
    if not quran_texts:
        print("ERROR: No texts loaded. Check your data path.")
        return

    # Step 2: Preprocess
    print("\n[2/4] Preprocessing and lemmatizing texts...")
    preprocessed_texts = [preprocess_text(text) for text in quran_texts]
    lemmatized_texts = [lemmatize_text(text) for text in preprocessed_texts]

    # Step 3: Vocabulary stats
    vocab = get_vocabulary(lemmatized_texts)
    print_vocabulary_stats(vocab)

    # Step 4: Train model
    print(f"\n[3/4] Training Word2Vec model (vector_size={args.vector_size}, window={args.window})...")
    model = train_word2vec(
        lemmatized_texts,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
    )
    print(f"  Vocabulary size in trained model: {len(model.wv.key_to_index)}")

    # Step 5: Save model
    print(f"\n[4/4] Saving model...")
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "word2vecLL_quran.model")
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
