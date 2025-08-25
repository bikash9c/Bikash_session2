from flask import Flask, render_template, request, redirect, url_for
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

app = Flask(__name__)

# Load GloVe model once at startup
model = api.load("glove-wiki-gigaword-50")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        words_input = request.form['words']
        words = [w.strip().lower() for w in words_input.split(',') if w.strip()]

        vectors = []
        valid_words = []

        for word in words:
            if word in model:
                vectors.append(model[word])
                valid_words.append(word)

        if not vectors:
            return render_template('index.html', error="No valid words found in GloVe model.")

        # Reduce dimensions to 2D using PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(vectors)

        # Plot and save to static/plot.png
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

        return redirect(url_for('result'))

    return render_template('index.html')


@app.route('/result')
def result():
    return render_template('result.html')


if __name__ == '__main__':
    app.run(debug=True)
