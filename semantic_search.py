"""
Feature-2 Semantic Search
-Load embedding model (all-MiniLM-L6-v2)
-Turn knowledge base and user questions into embeddings
-Find the best matching answer using cosine similarity and list indexing
"""

# Text to embedding
from sentence_transformers import SentenceTransformer

# Compare embedding similarity to find best match
from sklearn.metrics.pairwise import cosine_similarity

# Numpy arrays for embedding comparison and indexes
import numpy as np


# Load model at program start
def load_model():
    print("Loading embedding model: ")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model successful!")
    return model


# Generate embeddings for all knowledge base questions at program start
def generate_embeddings(model, questions):
    print("Generating embeddings for knowledge base questions: ")
    embeddings = model.encode(questions, convert_to_numpy=True)
    print("Embeddings successful!")
    return embeddings


# Find best match for user question and return best answer if found, else default reply
def find_best_answer(user_question, model, kb_questions, kb_answers, kb_embeddings):

    # Embed user question
    # encode() accepts list of strings
    user_embedding = model.encode([user_question], convert_to_numpy=True)

    # Compare user embeddings to stored knowledge base embeddings using cosine similarity
    # cosine_similarity returns 2D array
    similarity_scores = cosine_similarity(user_embedding, kb_embeddings)[0]

    # List index of best match
    best_match_index = int(np.argmax(similarity_scores))
    best_match_score = float(similarity_scores[best_match_index])

    # If the best score is too low, the question is likely unrelated to our knowledge base. But if threshold is set too high, related user questions might get ignored. (Did Trial-and-error, 0 being completely unrelated and 1 being identical match)
    if best_match_score < 0.35:
        return "We don't have a good answer for that, try asking another question!"

    # Return the most related answer to user question
    return kb_answers[best_match_index]