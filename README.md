# Compare Method: Word2Vec-LL vs Word2Vec-LLS on Quran Translations

A research project comparing **distributional** vs **hybrid (distributional + knowledge-based)** word similarity methods applied to English Quran translations.

## Overview

This project investigates whether combining Word2Vec embeddings with WordNet-based linguistic similarity measures (Wu-Palmer, Jiang-Conrath, Path Similarity) can improve word similarity prediction in the domain of Quranic text.

### Models

| Model | Description |
|-------|-------------|
| **Word2Vec-LL** | Standard Word2Vec (CBOW) trained on 17 English Quran translations with lemmatization |
| **Word2Vec-LLS** | Enhanced model that re-ranks Word2Vec results using a neural network trained on combined distributional + linguistic similarity features |

### Hypothesis

> Combining distributional similarity (Word2Vec) with knowledge-based similarity (WordNet) yields better word similarity predictions than Word2Vec alone.

---

## Project Structure

```
compare-method/
├── README.md
├── requirements.txt
├── data/
│   ├── 5w100c.txt              # Golden standard (51 concepts, 5 related words each)
│   ├── SimLex-999.txt          # SimLex-999 benchmark dataset
│   └── quran/                  # 17 English Quran translations
│       ├── en.ahmedali.txt
│       ├── en.sahih.txt
│       └── ... (15 more)
├── models/                     # Trained models (generated)
│   ├── word2vecLL_quran.model
│   └── word2vecLLS_quran.model
└── src/
    ├── train_word2vec.py       # Step 1: Train base Word2Vec model
    ├── train_word2vecLLS.py    # Step 2: Train enhanced LLS model
    └── evaluate.py             # Step 3: Compare both models
```

---

## Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aalamsy/compare-method.git
   cd compare-method
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   # or
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download required NLTK data (automatic on first run, or manually):**
   ```bash
   python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('wordnet_ic')"
   ```

---

## Usage

Run the scripts **in order** from the project root directory:

### Step 1: Train the base Word2Vec model

```bash
python src/train_word2vec.py
```

This loads all 17 Quran translations, preprocesses them (lowercase, remove punctuation, remove stopwords, lemmatize), and trains a Word2Vec CBOW model.

**Options:**
```bash
python src/train_word2vec.py --help

  --data-path      Path to Quran translations directory (default: data/quran)
  --output-dir     Directory to save model (default: models)
  --vector-size    Word vector dimensions (default: 200)
  --window         Context window size (default: 10)
  --min-count      Minimum word frequency (default: 5)
```

### Step 2: Train the Word2Vec-LLS model

```bash
python src/train_word2vecLLS.py
```

This loads the base Word2Vec model, computes linguistic similarity features (Wu-Palmer, Jiang-Conrath, Path Similarity) for training concepts, and trains a neural network (MLP) to re-rank word similarities.

**Options:**
```bash
python src/train_word2vecLLS.py --help

  --model-path       Path to pre-trained Word2Vec model (default: models/word2vecLL_quran.model)
  --golden-standard  Path to golden standard file (default: data/5w100c.txt)
  --output-dir       Directory to save LLS model (default: models)
  --max-iter         Maximum training iterations (default: 100)
  --topn             Top-N similar words per concept (default: 20)
```

### Step 3: Evaluate and compare models

```bash
python src/evaluate.py
```

This evaluates both models against the golden standard at various k values and prints a comparison table with Precision, Recall, and F1-score.

**Options:**
```bash
python src/evaluate.py --help

  --w2v-model        Path to Word2Vec-LL model (default: models/word2vecLL_quran.model)
  --lls-model        Path to Word2Vec-LLS model (default: models/word2vecLLS_quran.model)
  --golden-standard  Path to golden standard file (default: data/5w100c.txt)
  --k-values         K values for evaluation (default: 20 50 100 200)
```

---

## Methodology

### Pipeline

```
Quran Translations (17 files)
        |
        v
  Preprocessing (lowercase, clean, stopwords removal, lemmatization)
        |
        v
  Word2Vec Training (CBOW, 200 dims, window=10)
        |
        v
  word2vecLL model
        |
        +---> Direct evaluation (cosine similarity)
        |
        v
  Similarity Feature Extraction:
    - Wu-Palmer (WordNet taxonomy depth)
    - Jiang-Conrath (WordNet + information content)
    - Path Similarity (WordNet shortest path)
    - Word2Vec cosine distance
    - Bias term
        |
        v
  MLP Neural Network Training (on golden standard)
        |
        v
  word2vecLLS model (re-ranks using combined features)
        |
        +---> Re-ranked evaluation
```

### Evaluation Metrics

- **Precision@k**: Fraction of top-k predictions that are in the golden standard
- **Recall@k**: Fraction of golden standard words found in top-k predictions
- **F1@k**: Harmonic mean of precision and recall

### Golden Standard

The `5w100c.txt` file contains 51 Islamic/Quranic concepts, each with 5 gold-standard related words. Example:

```
faith: belief, trust, conviction, surrender, submission
god: creator, lord, sustainer, all-knowing, most merciful
```

---

## Data Sources

- **Quran Translations**: 17 English translations from various scholars
- **Golden Standard (5w100c.txt)**: Custom-built dataset of 51 Quranic concepts with 5 related words each
- **SimLex-999**: Standard word similarity benchmark (included for future experiments)

---

## Dependencies

| Package | Purpose |
|---------|---------|
| gensim | Word2Vec model training and similarity computation |
| nltk | WordNet access, stopwords, lemmatization |
| numpy | Numerical computations and matrix operations |
| scikit-learn | MLP neural network (MLPRegressor) |
| joblib | Model serialization |

---

## License

This project is part of academic research (Skripsi/Thesis).
