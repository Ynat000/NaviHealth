import requests
from bs4 import BeautifulSoup
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
from openai import OpenAI
import json

client = OpenAI()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Get user input
def get_user_input():
    print("Enter your Prompt")
    return input(":")


# scrap HealthBC
# Add headers to avoid being blocked
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

HealthBC_URLS = [
    (
        "Ebola Virus Disease",
        "https://www.healthlinkbc.ca/health-library/health-features/ebola-virus-disease",
    ),
    (
        "Coronavirus Disease (COVID-19)",
        "https://www.healthlinkbc.ca/health-library/health-features/coronavirus-disease-covid-19",
    ),
    (
        "Understanding Measles",
        "https://www.healthlinkbc.ca/health-library/health-features/understanding-measles",
    ),
]


def get_healthbc_text(index):
    filename, url = HealthBC_URLS[index]
    file_content = txt_file_to_string(filename + ".txt")
    return file_content, url


def web_scrap(url):
    # print(url)
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # First find main tag
    main_content = soup.find("main")

    if main_content:
        # Then find article within main
        article = main_content.find("article", class_="node--type-page")

        if article:
            # Get text from this article only
            text = article.get_text()

            # Clean up whitespace/empty lines
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            cleaned_text = "\n".join(lines)

            return cleaned_text
        else:
            return "Article with class 'node--type-page' not found in main"
    else:
        return "Main tag not found"


def scrap_healthbc():
    print("scraping healthbc")

    for page_name, page_url in HealthBC_URLS:
        web_data = web_scrap(page_url)
        filename = page_name + ".txt"
        with open(filename, "w") as f:
            f.write(web_data)


# Semantic Rank Documents
def txt_file_to_string(file_path: str) -> str:
    file_path = Path(file_path)
    file_content = file_path.read_text()
    return file_content


def embed_douments():
    print("Embedding Documents...")

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences: list[str] = []
    for filename, url in HealthBC_URLS:
        filepath = filename + ".txt"
        sentence = txt_file_to_string(filepath)
        sentences.append(sentence)

    knowledge_embeddings = embedding_model.encode(sentences)

    return knowledge_embeddings


# Get Top Semantic Document
def rank_semantic_similarity(user_query_embedding, knowledge_embeddings):
    similarities = []

    for i, knowledge_embedding in enumerate(knowledge_embeddings):
        sim = embedding_model.similarity(user_query_embedding, knowledge_embedding)
        similarities.append((i, sim.item()))

    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Return index of most similar embedding
    return similarities[0][0]


def get_top_semantic_document(user_query, knowledge_embeddings):
    user_query_embedding = embedding_model.encode(user_query)

    top_semantic_document_index = rank_semantic_similarity(
        user_query_embedding, knowledge_embeddings
    )

    top_semantic_document_text, top_semantic_document_source = get_healthbc_text(
        top_semantic_document_index
    )

    return top_semantic_document_text, top_semantic_document_source


# Feed to LLM (OpenAI)
def ask_openai(user_prompt: str):
    response = client.responses.create(
        model="gpt-5-nano",
        instructions="Return a few sentences only.",
        input=user_prompt,
    )

    return response.output_text


def augment_generation(
    user_query, top_semantic_document_text, top_semantic_document_source
):
    # print("Augment LLM Generation")
    # print("The sentence most similar to: ", user_query)
    # print("Is: ", top_semantic_document_text)
    # print("Source: ", top_semantic_document_source)

    response = client.responses.create(
        model="gpt-5.2",  # or "gpt-5" for reasoning models
        instructions=(
            "You are a helpful assistant that answers questions using provided documents. "
            "Always cite the source URL when using information from the document. "
            "If the retrieved document is not relevant to the question, inform the user "
            "and do not use it in your answer."
        ),
        input=f"""Retrieved Document: {top_semantic_document_text}
        Source: {top_semantic_document_source}
        Question: {user_query}
        Please answer the question using the context document above. Cite the source when appropriate.""",
    )

    return response.output_text

def augment_generation1(user_query, top_semantic_document_text, top_semantic_document_source):
    # response = requests.post('http://localhost:11434/api/generate',
    #     json={
    #         'model': 'llama3.2:3b',
    #         'prompt': 'Your user prompt here',
    #         'stream': False
    #     }
    # )
    # print(response.json()['response'])


    response = requests.post(
        'http://localhost:11434/api/chat',
        json={
            'model': 'llama3.2:3b',  # or 'mistral', 'phi3', etc.
            'messages': [
                {
                    'role': 'system',
                    'content': (
                        "You are a helpful assistant that answers questions using provided documents. "
                        "Always cite the source URL when using information from the document. "
                        "If the retrieved document is not relevant to the question, inform the user "
                        "and do not use it in your answer."
                    )
                },
                {
                    'role': 'user',
                    'content': f"""Retrieved Document: {top_semantic_document_text}
                        Source: {top_semantic_document_source}
                        Question: {user_query}
                        Please answer the question using the context document above. Cite the source when appropriate. Return in Markdown Format, with link to source."""
                }
            ],
            'stream': False
        }
    )

    # Extract the response
    result = response.json()
    answer = result['message']['content']
    return answer


def healthbc_rag(user_query):
    print("loading AI...")

    scrap_healthbc()
    knowledge_embeddings = embed_douments()

    # user_query = get_user_input()
    print(user_query)

    # Get Top Semantic Document
    top_semantic_document_text, top_semantic_document_source = (
        get_top_semantic_document(user_query, knowledge_embeddings)
    )

    # OpenAI LLM 
    # llm_output = augment_generation(
    #     user_query, top_semantic_document_text, top_semantic_document_source
    # )
    # print(llm_output)

    # Llama 3.2 on Ollama
    llm_output = augment_generation1(
        user_query, top_semantic_document_text, top_semantic_document_source
    )
    return llm_output
