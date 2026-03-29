"""
HealthLinkBC Files Scraper
Fetches all files and their last-updated dates from:
https://www.healthlinkbc.ca/health-library/healthlinkbc-files
"""

import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime
from src.globals import ARTIFACTS_DIR, URL_CRAWLS_DIR, HEADERS

URL = "https://www.healthlinkbc.ca/health-library/healthlinkbc-files"


def scrape_hlbc_files() -> list[dict]:
    """Scrape all HealthLinkBC files and return as a list of dicts."""
    print(f"Fetching {URL} ...")
    response = requests.get(URL, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # The table has a thead with columns: #, File name, Last updated
    # Find all <tr> rows inside the table body
    table = soup.find("table")
    if not table:
        raise ValueError("Could not find the files table in the page HTML.")

    rows = table.find("tbody").find_all("tr") if table.find("tbody") else table.find_all("tr")[1:]

    files = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) < 3:
            continue

        file_number = cols[0].get_text(strip=True)

        # File name cell contains an <a> tag with the link
        link_tag = cols[1].find("a")
        file_name = link_tag.get_text(strip=True) if link_tag else cols[1].get_text(strip=True)
        file_url = link_tag["href"] if link_tag else None

        # Make relative URLs absolute
        if file_url and file_url.startswith("/"):
            file_url = "https://www.healthlinkbc.ca" + file_url

        last_updated = cols[2].get_text(strip=True)

        files.append({
            "file_number": file_number,
            "file_name": file_name,
            "last_updated": last_updated,
            "url": file_url,
        })

    return files


def save_csv(files: list[dict], path: str | None = None) -> None:
    """Save the file list to a CSV."""
    output_path: Path
    if path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        URL_CRAWLS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = URL_CRAWLS_DIR / f"hlbc_files_{timestamp}.csv"
    else:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file_number", "file_name", "last_updated", "url"])
        writer.writeheader()
        writer.writerows(files)
    print(f"Saved {len(files)} rows to {output_path}")


if __name__ == "__main__":
    files = scrape_hlbc_files()

    print(f"\nFound {len(files)} files.\n")
    print(f"{'#':<8} {'Last Updated':<14} File Name")
    print("-" * 80)
    for f in files:
        print(f"{f['file_number']:<8} {f['last_updated']:<14} {f['file_name']}")

    # Save output as timestamped CSV:
    save_csv(files)