
"""Convert cleaned HLBC HTML files to Markdown."""

from pathlib import Path
import html2text
from src.globals import CLEAN_HTML_DIR, MD_OUTPUT_DIR

def html_to_markdown_file(input_path: Path, output_path: Path) -> None:
    """Convert a single HTML file to Markdown and save to output path."""
    html_content = input_path.read_text(encoding="utf-8", errors="ignore")
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = False
    h.ignore_emphasis = False
    h.body_width = 0
    markdown_content = h.handle(html_content)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_content, encoding="utf-8")
    print(f"Converted: {input_path} -> {output_path}")

def convert_folder(input_dir: Path = CLEAN_HTML_DIR, output_dir: Path = MD_OUTPUT_DIR) -> int:
    """Convert all HTML files in input_dir to Markdown files in output_dir.
    Returns the number of files converted.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_files = sorted(list(input_dir.glob("*.html")) + list(input_dir.glob("*.htm")))
    if not html_files:
        print(f"No HTML files found in {input_dir}")
        return 0
    print(f"Found {len(html_files)} HTML file(s) to convert.")
    count = 0
    for html_file in html_files:
        output_file = output_dir / html_file.with_suffix(".md").name
        try:
            html_to_markdown_file(html_file, output_file)
            count += 1
        except Exception as e:
            print(f"Error converting {html_file}: {e}")
    print(f"\nConversion complete! {count} file(s) saved to {output_dir}")
    return count



if __name__ == "__main__":
    convert_folder()