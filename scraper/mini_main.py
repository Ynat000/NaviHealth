import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.healthlinkbc.ca/health-library/immunizations"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

response = requests.get(BASE_URL, headers=headers, timeout=30)
print(response)
soup = BeautifulSoup(response.content, 'html.parser')
print(soup.title.string)

# Save prettified HTML
def save_to_file(soup, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(soup.prettify())