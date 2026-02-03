import weaviate
import weaviate.classes as wvc
import json
from pathlib import Path
import requests


HEALTHBC_DATA_FOLDERPATH = "/Users/andywang/andy/NaviHealth/healthbc_data/"

# Scrap HealthBC websites into /healthbc_data folder
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


def semantic_search_wv8(query: str):
    with weaviate.connect_to_local() as client:
        collection = client.collections.get("DocumentChunk")

        # Perform near_text search
        response = collection.query.near_text(
            query=query, limit=1, return_metadata=wvc.query.MetadataQuery(distance=True)
        )

        obj = response.objects[0]

        return obj.properties["chunk_file"]


def get_healthbc_text(filename):
    file_path = HEALTHBC_DATA_FOLDERPATH + filename + ".txt"
    file_path = Path(file_path)
    file_content = file_path.read_text()
    return file_content


def augment_generation(
    user_query, top_semantic_document_text, top_semantic_document_source
):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",  # or 'mistral', 'phi3', etc.
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers questions using provided documents. "
                        "Always cite the source URL when using information from the document. "
                        "If the retrieved document is not relevant to the question, inform the user "
                        "and do not use it in your answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""Retrieved Document: {top_semantic_document_text}
                        Source: {top_semantic_document_source}
                        Question: {user_query}
                        Please answer the question using the context document above. Cite the source when appropriate. Return in Markdown Format, with link to source.""",
                },
            ],
            "stream": False,
        },
    )

    # Extract the response
    result = response.json()
    answer = result["message"]["content"]
    return answer


def healthbc_rag(user_query):
    print("HealthBC RAG - Semantic Search of WV8")
    top_semantic_search_filename = semantic_search_wv8(user_query)
    top_semantic_document_text = get_healthbc_text(top_semantic_search_filename)
    top_semantic_document_source = ""
    for topic, url in HealthBC_URLS:
        if topic == top_semantic_document_text:
            top_semantic_document_source = "url"

    # augment llm with document
    print("HealthBC RAG - Augment Generation with Ollama Llama3.2")
    llm_output = augment_generation(
        user_query, top_semantic_document_text, top_semantic_document_source
    )
    return llm_output
