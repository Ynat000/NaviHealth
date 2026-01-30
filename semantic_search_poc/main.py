print("Loading...")

from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

SENTENCE_1_SOURCE = "https://www.healthlinkbc.ca/primary-care/service-type/urgent-and-primary-care-centres - Retrieved Jan 18, 2026"
SENTENCE_1_FILE = "./sample_data/upcc.txt"
SENTENCE_2_SOURCE = "https://www.healthlinkbc.ca/primary-care/service-type/walk-in-clinics - Retrieved Jan 18, 2026"
SENTENCE_2_FILE = "./sample_data/walk_in_clinics.txt"


# Data Preprocessing
def txt_file_to_string(file_path: str) -> str:
    file_path = Path(file_path)
    file_content = file_path.read_text()
    return file_content


def get_sentences():
    sentence_1 = txt_file_to_string(SENTENCE_1_FILE)
    sentence_2 = txt_file_to_string(SENTENCE_2_FILE)
    sentences = [sentence_1, sentence_2]
    return sentences


def data_preprocessing():
    sentences = get_sentences()

    knowledge_embeddings = embedding_model.encode(sentences)

    return knowledge_embeddings


def get_user_query_embedding(user_query):
    user_query_embedding = embedding_model.encode(user_query)
    return user_query_embedding


def rank_semantic_similarity(user_query_embedding, knowledge_embeddings):
    sim1: torch.Tensor = embedding_model.similarity(
        user_query_embedding, knowledge_embeddings[0]
    )
    sim1 = sim1.item()

    sim2: torch.Tensor = embedding_model.similarity(
        user_query_embedding, knowledge_embeddings[1]
    )
    sim2 = sim2.item()

    if sim1 > sim2:
        return 1
    else:
        return 2


def main():
    print("Hello, world!")
    knowledge_embeddings = data_preprocessing()

    user_query = input("Enter User Query: ")
    user_query_embedding = get_user_query_embedding(user_query)

    semantic_rankings = rank_semantic_similarity(
        user_query_embedding, knowledge_embeddings
    )

    sentences = get_sentences()
    similar_sentence = ""
    source = ""
    if semantic_rankings == 1:
        similar_sentence = sentences[0]
        source = SENTENCE_1_SOURCE
    else:
        similar_sentence = sentences[1]
        source = SENTENCE_2_SOURCE

    print("The sentence most similar to: ", user_query)
    print("Is: ", similar_sentence)
    print("Source: ", source)


main()
