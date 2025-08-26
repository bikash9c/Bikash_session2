# Word Embedding Explorer

A simple Flask web app to visualize word embeddings using pre-trained GloVe vectors and PCA.

---

## Features

- Enter a list of comma-separated words.
- Visualize their 2D embeddings using PCA.
- See statistics about the input words:
  - Total, valid, and invalid words.
  - Top 3 most similar word pairs based on cosine similarity.
- Interactive and user-friendly UI.

---

## Requirements

- Python 3.7+
- Flask
- matplotlib
- scikit-learn
- gensim

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/word-embedding-explorer.git
cd word-embedding-explorer
