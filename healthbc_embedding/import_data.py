import weaviate
import weaviate.classes as wvc
from pathlib import Path

import requests
from bs4 import BeautifulSoup

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


def web_scrap(url):
    # Add headers to avoid being blocked
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

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
        filepath = HEALTHBC_DATA_FOLDERPATH + filename
        with open(filepath, "w") as f:
            f.write(web_data)


# Chunk HealthBC Files into WV8 DocumentChunk Data Objects
def txt_file_to_string(file_path: str) -> str:
    file_path = Path(file_path)
    file_content = file_path.read_text()
    return file_content


def get_healthbc_text(filename):
    filepath = HEALTHBC_DATA_FOLDERPATH + filename + ".txt"
    file_content = txt_file_to_string(filepath)
    return file_content


def chunk_file(filename: str) -> list[dict]:
    chunk_size = 150
    overlap = 50

    # Read the file
    text = get_healthbc_text(filename)

    # Split into words
    words = text.split()

    # Calculate step size (how many words to advance each time)
    step = chunk_size - overlap

    # Generate chunks
    chunks = []
    for i in range(0, len(words), step):
        # Get chunk_size words starting from position i
        chunk_words = words[i : i + chunk_size]

        # Stop if we don't have enough words left
        if len(chunk_words) == 0:
            break

        # Join words back into text
        chunk_text = " ".join(chunk_words)

        # Create object
        chunks.append({"chunk": chunk_text, "chunk_file": filename})

        # Break if this was the last chunk
        if i + chunk_size >= len(words):
            break

    return chunks


def get_data_objects():
    data_objects = []
    for filename, url in HealthBC_URLS:
        data_objects.extend(chunk_file(filename))

    return data_objects


def import_data():
    # batch size refers to objects per import request. It has nothing to do with how the data is embedded.

    data_objects = get_data_objects()

    with weaviate.connect_to_local() as client:
        document_chunks = client.collections.use("DocumentChunk")
        with document_chunks.batch.fixed_size(batch_size=200) as batch:
            for obj in data_objects:
                batch.add_object(properties=obj)

        print(
            f"Imported & vectorized {len(document_chunks)} objects into the DocumentChunk collection"
        )


# scrap_healthbc() # 1. Do Once
# import_data() # 2. Do Once
