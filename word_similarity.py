"""
Word Similarity Engine - Starter Template

Name: Trevor Tourdot
Date: 09/23/2026

"""

import math
import string
from collections import defaultdict


def tokenize(text):
    """
    Convert text to lowercase, remove punctuation, and split into tokens.

    Example:
    "The cat sat." -> ["the", "cat", "sat"]
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Split into words
    tokens = text.split()

    return tokens


def build_vocabulary(tokens):
    """
    Build and return a sorted list of unique words.
    """
    # Return sorted unique words
    vocab = sorted(set(tokens))
    return vocab


def build_cooccurrence_matrix(tokens, vocab, window_size=2):
    """
    Build a word-context co-occurrence matrix.

    Each target word should count the words that appear within
    a window of +/- window_size around it.

    Return:
    A dictionary of dictionaries, like:
    {
        "cat": {"the": 2, "sat": 1},
        "dog": {"the": 1, "barked": 1}
    }
    """
    # Create nested dictionary structure
    matrix = {word: defaultdict(int) for word in vocab}

    # Loop through each token
    for i, word in enumerate(tokens):

        # Look at neighboring words inside the context window
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)

        for j in range(start, end):
            if j == i:
                continue
            context_word = tokens[j]

            # Count co-occurrences
            if context_word in vocab:
                matrix[word][context_word] += 1

    return matrix


def word_to_vector(word, vocab, matrix):
    """
    Convert a word into a vector using the vocabulary order.

    Example:
    If vocab = ["cat", "dog", "sat"]
    and matrix["cat"] = {"dog": 1, "sat": 2}
    then the vector should be:
    [0, 1, 2]
    """
    # Create vector in vocab order
    vector = [matrix[word][v] for v in vocab]
    return vector


def dot_product(vec1, vec2):
    """
    Compute the dot product of two vectors.
    """
    # Multiply matching values and sum them
    return sum(a * b for a, b in zip(vec1, vec2))


def vector_length(vec):
    """
    Compute the Euclidean length of a vector.
    """
    # Use square root of sum of squared values
    return math.sqrt(sum(v * v for v in vec))


def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.

    Formula:
    dot_product(vec1, vec2) / (|vec1| * |vec2|)

    If either vector has length 0, return 0.0
    """
    # Compute vector lengths
    length1 = vector_length(vec1)
    length2 = vector_length(vec2)

    # Handle zero-length vectors
    if length1 == 0 or length2 == 0:
        return 0.0

    # Return cosine similarity
    return dot_product(vec1, vec2) / (length1 * length2)


def most_similar_words(target_word, vocab, matrix, top_n=3):
    """
    Find the top N most similar words to the target word.

    Return:
    A list of tuples like:
    [("dog", 0.82), ("animal", 0.75), ("pet", 0.63)]
    """
    # Check if target word exists
    if target_word not in vocab:
        return []

    # Convert target word to vector
    target_vector = word_to_vector(target_word, vocab, matrix)

    similarities = []

    # Compare target word to every other word
    # Skip the target word itself
    for word in vocab:
        if word == target_word:
            continue

        vec = word_to_vector(word, vocab, matrix)
        score = cosine_similarity(target_vector, vec)
        similarities.append((word, score))

    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return the top N results
    return similarities[:top_n]


def print_matrix(matrix, vocab):
    """
    Print a readable version of the co-occurrence matrix.
    """
    print("\nCo-occurrence Matrix:")
    header = ["word"] + vocab
    print("\t".join(header))

    for word in vocab:
        row = [word]
        for context_word in vocab:
            row.append(str(matrix[word][context_word]))
        print("\t".join(row))


def print_word_vector(word, vocab, matrix):
    """
    Print the vector for a specific word.
    """
    if word not in vocab:
        print(f"'{word}' is not in the vocabulary.")
        return

    vector = word_to_vector(word, vocab, matrix)
    print(f"\nVector for '{word}':")
    print(vector)


def main():
    text = """
    The cat sat on the mat.
    The dog sat on the log.
    Cats and dogs are animals.
    The animal sat on the mat.
    Dogs and cats can be friends.
    The cat and the dog played together.
    The animal and the dog sat together.
    """

    # Step 1: Tokenize text
    tokens = tokenize(text)

    # Step 2: Build vocabulary
    vocab = build_vocabulary(tokens)

    # Step 3: Build co-occurrence matrix
    matrix = build_cooccurrence_matrix(tokens, vocab, window_size=2)

    print("Tokens:")
    print(tokens)

    print("\nVocabulary:")
    print(vocab)

    print_matrix(matrix, vocab)

    # Show vectors for sample words
    print_word_vector("cat", vocab, matrix)
    print_word_vector("dog", vocab, matrix)
    print_word_vector("animal", vocab, matrix)

    # Test similarity
    test_words = ["cat", "dog", "animal", "sat"]

    for word in test_words:
        print(f"\nTarget word: {word}")
        similar = most_similar_words(word, vocab, matrix, top_n=3)

        if not similar:
            print("Word not found in vocabulary.")
        else:
            print("Most similar words:")
            for i, (similar_word, score) in enumerate(similar, start=1):
                print(f"{i}. {similar_word} ({score:.3f})")


if __name__ == "__main__":
    main()