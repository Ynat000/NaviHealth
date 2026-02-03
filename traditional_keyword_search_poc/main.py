import requests
from rank_bm25 import BM25Okapi
from bs4 import BeautifulSoup


# Add headers to avoid being blocked
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

HealthBC_URLS = [
    ("Ebola Virus Disease", "https://www.healthlinkbc.ca/health-library/health-features/ebola-virus-disease"),
    ("Coronavirus Disease (COVID-19)", "https://www.healthlinkbc.ca/health-library/health-features/coronavirus-disease-covid-19"),
    ("Understanding Measles", "https://www.healthlinkbc.ca/health-library/health-features/understanding-measles")
]

def web_scrap(url):
    # print(url)
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')

    # First find main tag
    main_content = soup.find('main')

    if main_content:
        # Then find article within main
        article = main_content.find('article', class_='node--type-page')
        
        if article:
            # Get text from this article only
            text = article.get_text()
            
            # Clean up whitespace/empty lines
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)
            
            return(cleaned_text)
        else:
            return("Article with class 'node--type-page' not found in main")
    else:
        return("Main tag not found")

def search_corpus(bm25, corpus, top_n):
    print("Enter Search Query")
    user_query = input(":")
    tokenized_query = user_query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)
    # print(doc_scores)

    top_n_queries = bm25.get_top_n(tokenized_query, corpus, n=top_n)
    # print(top_n_queries)

    return top_n_queries

def clean_txt_file(filepath):
    # Method 3: Remove all extra whitespace (multiple blank lines → single line)

    cleaned = ""

    with open(filepath, 'r') as f:
        text = f.read()
        # Remove multiple newlines
        import re
        cleaned = re.sub(r'\n\s*\n', '\n\n', text)  # Max 1 blank line between paragraphs

    with open(filepath, "w") as f:
        f.write(cleaned)


def scrap_healthbc_data():
    for page_name, page_url in HealthBC_URLS:
        web_data = web_scrap(page_url)
        filename = page_name + ".txt"
        with open(filename, "w") as f:
            f.write(web_data)

def main(): 

    scrap_healthbc_data()

    # Read each file into the corpus
    corpus = []
    for filepathname, url in HealthBC_URLS:
        filepath = filepathname + ".txt"
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()  # Read entire file and remove leading/trailing whitespace
            corpus.append(content)

    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    top_n = 1 
    top_n_documents = search_corpus(bm25, corpus, top_n)
    print(top_n_documents)

main() 
