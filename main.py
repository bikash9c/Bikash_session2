from flask import Flask, render_template, request, redirect, url_for
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import numpy as np
from itertools import combinations

app = Flask(__name__)

# Load GloVe model once
model = api.load("glove-wiki-gigaword-50")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        words_input = request.form['words']
        words = [w.strip().lower() for w in words_input.split(',') if w.strip()]

        vectors = []
        valid_words = []
        invalid_words = []

        for word in words:
            if word in model:
                vectors.append(model[word])
                valid_words.append(word)
            else:
                invalid_words.append(word)

        if not valid_words:
            return render_template('index.html', error="No valid words found in GloVe model.")

        # PCA for plotting
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        # Plot
        plt.figure(figsize=(6, 6))
        for i, word in enumerate(valid_words):
            x, y = reduced[i]
            plt.scatter(x, y)
            plt.text(x + 0.01, y + 0.01, word, fontsize=9)
        plt.title('Word Embeddings (PCA)')
        plt.tight_layout()

        os.makedirs("static", exist_ok=True)
        plt.savefig("static/plot.png")
        plt.close()

        # Cosine similarity for valid words
        sim_scores = []
        for (i, j) in combinations(range(len(valid_words)), 2):
            sim = cosine_similarity([vectors[i]], [vectors[j]])[0][0]
            sim_scores.append((valid_words[i], valid_words[j], sim))

        # Sort top 3 most similar pairs
        top_similar = sorted(sim_scores, key=lambda x: -x[2])[:3]

        # Pass stats to result template
        return render_template(
            'result.html',
            total=len(words),
            valid=len(valid_words),
            invalid=invalid_words,
            top_similar=top_similar
        )

    return render_template('index.html')


@app.route('/result')
def result():
    # This route now only works after POST; you can disable direct GET access or keep it empty.
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

