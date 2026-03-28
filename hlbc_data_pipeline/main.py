
"""Main entry point for the NaviHealthPipeline workflow."""

from src.claude_crawler import scrape_hlbc_files, save_csv
from src.download_hlbc_files import download_hlbc_files
from src.clean_hlbc_files import clean_hlbc_files
from src.html_to_md import convert_folder

def main() -> None:
    """Run the end-to-end pipeline steps in order."""
    print("Step 1: Crawling HLBC file metadata...")
    files = scrape_hlbc_files()
    save_csv(files)

    print("\nStep 2: Downloading HLBC HTML files...")
    download_hlbc_files()

    print("\nStep 3: Cleaning downloaded HTML files...")
    clean_hlbc_files()

    print("\nStep 4: Converting cleaned HTML to Markdown...")
    convert_folder()

    print("\nPipeline complete.")

if __name__ == "__main__":
    main()
