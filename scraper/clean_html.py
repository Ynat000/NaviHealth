from pathlib import Path
import html2text

def html_to_markdown_file(input_filepath, output_filepath):
    """
    Convert an HTML file to Markdown format.
    
    Args:
        input_filepath: Path to the input HTML file
        output_filepath: Path to save the output Markdown file
    """
    # Read the HTML file
    with open(input_filepath, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Configure html2text
    h = html2text.HTML2Text()
    h.ignore_links = False  # Keep links
    h.ignore_images = False  # Keep images
    h.ignore_emphasis = False  # Keep bold/italic
    h.body_width = 0  # Don't wrap lines
    
    # Convert to Markdown
    markdown_content = h.handle(html_content)
    
    # Ensure output directory exists
    Path(output_filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Write the Markdown file
    with open(output_filepath, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    print(f"Converted: {input_filepath} -> {output_filepath}")


def convert_folder(input_folder, output_folder):
    """
    Convert all HTML files in input_folder to Markdown files in output_folder.
    
    Args:
        input_folder: Path to folder containing HTML files
        output_folder: Path to folder where Markdown files will be saved
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all HTML files
    html_files = list(input_path.glob("*.html")) + list(input_path.glob("*.htm"))
    
    if not html_files:
        print(f"No HTML files found in {input_folder}")
        return
    
    print(f"Found {len(html_files)} HTML file(s) to convert")
    
    # Convert each HTML file
    for html_file in html_files:
        # Create output filepath with .md extension
        output_file = output_path / html_file.with_suffix(".md").name
        
        try:
            html_to_markdown_file(html_file, output_file)
        except Exception as e:
            print(f"Error converting {html_file}: {e}")
    
    print(f"\nConversion complete! Files saved to {output_folder}")


# Example usage:
if __name__ == "__main__":
    convert_folder("pages", "clean_pages")