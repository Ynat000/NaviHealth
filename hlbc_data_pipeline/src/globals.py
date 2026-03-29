"""Global constants for artifact/output directories and HTTP headers."""

from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
URL_CRAWLS_DIR = ARTIFACTS_DIR / "url_crawls"
DOWNLOAD_DIR = ARTIFACTS_DIR / "hlbc_files_html"
RAW_HTML_DIR = ARTIFACTS_DIR / "hlbc_files_html"
CLEAN_HTML_DIR = ARTIFACTS_DIR / "hlbc_files_clean"
MD_OUTPUT_DIR = ARTIFACTS_DIR / "hlbc_files_md"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
