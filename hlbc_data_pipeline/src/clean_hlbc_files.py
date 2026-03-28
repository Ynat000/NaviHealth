"""Clean downloaded HealthLink BC HTML files."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Comment
from src.globals import RAW_HTML_DIR, CLEAN_HTML_DIR

TAGS_TO_REMOVE = {
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "iframe",
    "header",
    "footer",
    "nav",
}


def get_raw_html_files(raw_dir: Path = RAW_HTML_DIR) -> list[Path]:
    """Return raw downloaded HLBC HTML files."""
    return sorted(raw_dir.glob("*.html"))


def clean_html_content(html: str) -> str:
    """Remove noisy markup and return prettified HTML."""
    soup = BeautifulSoup(html, "html.parser")

    for tag_name in TAGS_TO_REMOVE:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    main_content = soup.find("main")
    if main_content is not None:
        wrapper = BeautifulSoup("<html><body></body></html>", "html.parser")
        wrapper.body.append(main_content)
        return wrapper.prettify()

    return soup.prettify()


def clean_single_file(input_path: Path, output_path: Path) -> None:
    """Clean one HTML file and save to output path."""
    html = input_path.read_text(encoding="utf-8", errors="ignore")
    cleaned_html = clean_html_content(html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned_html, encoding="utf-8")


def clean_hlbc_files() -> int:
    """Clean all downloaded HLBC HTML files and save to artifacts/hlbc_files_clean."""
    raw_files = get_raw_html_files()
    if not raw_files:
        print(f"No raw HTML files found in {RAW_HTML_DIR}")
        return 0

    cleaned_count = 0
    for input_path in raw_files:
        output_path = CLEAN_HTML_DIR / input_path.name
        clean_single_file(input_path, output_path)
        cleaned_count += 1

    print(f"Cleaned {cleaned_count} files into {CLEAN_HTML_DIR}")
    return cleaned_count


if __name__ == "__main__":
    clean_hlbc_files()

