# NLP_WordSimilarity
A small Python program that builds word vectors from a text sample and computes similarity between words based on their co‑occurrence patterns. It uses a sliding window to count how often words appear near each other, then converts those counts into vectors and compares them using cosine similarity.

# Features
- Tokenizes text (lowercasing, punctuation removal)
- Builds a sorted vocabulary
- Constructs a co‑occurrence matrix
- Converts words into numeric vectors
- Computes cosine similarity between word vectors
- Lists the most similar words for a given target word

# How It Works
- The text is cleaned and split into tokens.
- A vocabulary of unique words is created.
- A co‑occurrence matrix counts how often words appear within a ±2 window.
- Each word becomes a vector based on its co‑occurrence counts.
- Cosine similarity measures how similar two word vectors are.
- The program prints tokens, vocabulary, matrix, vectors, and top similar words.
