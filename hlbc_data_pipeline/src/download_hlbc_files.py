"""Download HealthLink BC file pages from the latest crawl CSV."""

from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from src.globals import ARTIFACTS_DIR, URL_CRAWLS_DIR, DOWNLOAD_DIR, HEADERS

CSV_PATTERN = "hlbc_files_*.csv"


@dataclass(frozen=True)
class CrawlRow:
    """Single CSV row required for downloading a file page."""

    file_number: str
    file_name: str
    last_updated: str
    url: str


def get_latest_crawl_csv(crawls_dir: Path = URL_CRAWLS_DIR) -> Path:
    """Return the latest timestamped HLBC crawl CSV in artifacts/url_crawls."""
    csv_paths = sorted(crawls_dir.glob(CSV_PATTERN))
    if not csv_paths:
        raise FileNotFoundError("No HLBC crawl CSV files found in artifacts/url_crawls.")
    return csv_paths[-1]


def read_crawl_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read crawl rows from CSV."""
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def fetch_html(url: str, timeout_seconds: int = 30) -> str:
    """Fetch URL and return parsed HTML as string using BeautifulSoup."""
    response = requests.get(url, headers=HEADERS, timeout=timeout_seconds)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return str(soup)


def build_output_filename(file_number: str, file_name: str, last_updated: str) -> str:
    """Build safe output filename as file number + file name + last updated."""
    normalized_number = re.sub(r"\s+", "", file_number.strip())
    normalized_name = re.sub(r"\s+", "_", file_name.strip())
    normalized_name = re.sub(r"[^A-Za-z0-9_.-]", "", normalized_name)
    normalized_last_updated = re.sub(r"\s+", "_", last_updated.strip())
    normalized_last_updated = re.sub(r"[^A-Za-z0-9_.-]", "", normalized_last_updated)
    if not normalized_last_updated:
        normalized_last_updated = "unknown"
    return f"{normalized_number}_{normalized_name}_{normalized_last_updated}.html"


def save_html_file(html: str, output_path: Path) -> None:
    """Write HTML content to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def parse_crawl_row(row: dict[str, str]) -> CrawlRow | None:
    """Parse and validate one crawl CSV row."""
    crawl_row = CrawlRow(
        file_number=(row.get("file_number") or "").strip(),
        file_name=(row.get("file_name") or "").strip(),
        last_updated=(row.get("last_updated") or "").strip(),
        url=(row.get("url") or "").strip(),
    )
    if not crawl_row.file_number or not crawl_row.file_name or not crawl_row.url:
        return None
    return crawl_row


def download_single_row(crawl_row: CrawlRow) -> str:
    """Download one HLBC file page.

    Returns one of: downloaded, skipped_existing, skipped_error.
    """
    filename = build_output_filename(
        crawl_row.file_number,
        crawl_row.file_name,
        crawl_row.last_updated,
    )
    output_path = DOWNLOAD_DIR / filename
    if output_path.exists():
        return "skipped_existing"

    try:
        print(f"Downloading {crawl_row.file_number} - {crawl_row.file_name}")
        html = fetch_html(crawl_row.url)
        save_html_file(html, output_path)
        return "downloaded"
    except Exception as exc:
        print(f"Skipping {crawl_row.file_number} ({crawl_row.file_name}) due to error: {exc}")
        return "skipped_error"


def download_hlbc_files(delay_seconds: float = 5.0) -> int:
    """Download all HLBC URLs from latest crawl CSV with a fixed delay."""
    latest_csv = get_latest_crawl_csv()
    rows = read_crawl_rows(latest_csv)
    counts = {
        "downloaded": 0,
        "skipped_existing": 0,
        "skipped_error": 0,
    }

    print(f"Using crawl list: {latest_csv}")
    print(f"Rows to process: {len(rows)}")

    for row in rows:
        time.sleep(delay_seconds)

        crawl_row = parse_crawl_row(row)
        if crawl_row is None:
            continue

        status = download_single_row(crawl_row)
        counts[status] += 1

    print(f"Downloaded {counts['downloaded']} files to {DOWNLOAD_DIR}")
    print(f"Skipped {counts['skipped_existing']} files already up to date")
    if counts["skipped_error"]:
        print(f"Skipped {counts['skipped_error']} files due to errors")
    return counts["downloaded"]


if __name__ == "__main__":
    download_hlbc_files()
