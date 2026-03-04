import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import json
import time

MAX_PAGES = 10
BASE_URL = "https://www.healthlinkbc.ca/"
REQUESTS_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

class WebsiteCrawler:
    def __init__(self, base_url):
        self.base_url = base_url
        self.visited = set()
        self.history = []
        self.page_ID = 0 
        self.delay = 2.0
        
    def crawl(self, start_url, max_pages=MAX_PAGES):
        queue = deque([start_url]) 
        
        while queue and len(self.visited) < max_pages:
            time.sleep(self.delay)
            url = queue.popleft()
            
            if url in self.visited:
                continue
                
            # Fetch and parse
            try:
                response = requests.get(url, headers=REQUESTS_HEADERS, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')

                self.visited.add(url)

                filepath = "./pages/" + str(self.page_ID) + ".html" 
                self.save_to_file(soup, filepath)

                self.history.append((url, filepath))
                
                # Find all links on this page
                for link in soup.find_all('a', href=True):
                    # urljoin converts relative URLs to absolute URLs by combining them with the current page's URL.
                    full_url = urljoin(url, link['href'])
                    
                    # Only follow links within same domain
                    # Note: print(parsed.netloc)   # "www.example.gov"
                    if urlparse(full_url).netloc == urlparse(self.base_url).netloc:
                        if full_url not in self.visited:
                            queue.append(full_url)

                self.page_ID += 1
                            
            except Exception as e:
                print(f"Error crawling {url}: {e}")
    
    def save_visited_sites(self, filename='history.json'):
        """Save hierarchy to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"History saved to {filename}")


    # Save prettified HTML
    def save_to_file(self, soup, filepath):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(soup.prettify())

crawler = WebsiteCrawler(
    base_url=BASE_URL,
)

crawler.crawl(BASE_URL)

# Save to file
crawler.save_visited_sites()

