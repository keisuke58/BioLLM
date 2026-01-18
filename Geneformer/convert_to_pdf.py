#!/usr/bin/env python3
"""
Convert Markdown report to PDF.
This script converts the final report from Markdown to PDF format.
"""
import os
import sys
from pathlib import Path

def convert_markdown_to_pdf_markdown2pdf(md_file, pdf_file):
    """Convert using markdown2pdf library."""
    try:
        from markdown2pdf import convert
        convert(md_file, pdf_file)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"Error with markdown2pdf: {e}")
        return False

def convert_markdown_to_pdf_weasyprint(md_file, pdf_file):
    """Convert using markdown + weasyprint."""
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
        
        # Read markdown
        with open(md_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML
        html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])
        
        # Add basic styling
        html_doc = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                @page {{
                    size: A4;
                    margin: 2cm;
                }}
                body {{
                    font-family: 'DejaVu Sans', Arial, sans-serif;
                    line-height: 1.6;
                }}
                h1 {{ font-size: 24pt; margin-top: 20pt; }}
                h2 {{ font-size: 20pt; margin-top: 16pt; }}
                h3 {{ font-size: 16pt; margin-top: 12pt; }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 10pt 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8pt;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2pt 4pt;
                }}
            </style>
        </head>
        <body>
        {html_content}
        </body>
        </html>
        """
        
        # Convert to PDF
        HTML(string=html_doc).write_pdf(pdf_file)
        return True
    except ImportError as e:
        print(f"Required library not installed: {e}")
        return False
    except Exception as e:
        print(f"Error with weasyprint: {e}")
        return False

def convert_markdown_to_pdf_pypandoc(md_file, pdf_file):
    """Convert using pypandoc (requires pandoc and LaTeX)."""
    try:
        import pypandoc
        pypandoc.convert_file(md_file, 'pdf', outputfile=pdf_file, extra_args=['--pdf-engine=pdflatex'])
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"Error with pypandoc: {e}")
        return False

def main():
    """Convert final report to PDF."""
    workdir = Path(__file__).parent
    
    # Input and output files
    md_file = workdir / "submission_package" / "FINAL_REPORT.md"
    pdf_file = workdir / "submission_package" / "FINAL_REPORT.pdf"
    
    if not md_file.exists():
        print(f"Error: Markdown file not found: {md_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("Converting Markdown to PDF")
    print("=" * 60)
    print(f"Input: {md_file}")
    print(f"Output: {pdf_file}")
    print()
    
    # Try different conversion methods
    methods = [
        ("weasyprint", convert_markdown_to_pdf_weasyprint),
        ("pypandoc", convert_markdown_to_pdf_pypandoc),
        ("markdown2pdf", convert_markdown_to_pdf_markdown2pdf),
    ]
    
    success = False
    for method_name, method_func in methods:
        print(f"Trying {method_name}...")
        try:
            if method_func(md_file, pdf_file):
                print(f"✓ Successfully converted using {method_name}")
                print(f"PDF saved to: {pdf_file}")
                success = True
                break
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    if not success:
        print()
        print("=" * 60)
        print("PDF Conversion Failed")
        print("=" * 60)
        print("Could not convert Markdown to PDF automatically.")
        print()
        print("Alternative options:")
        print("1. Install conversion tools:")
        print("   pip install markdown weasyprint")
        print("   # or")
        print("   pip install pypandoc")
        print("   # (requires pandoc and LaTeX)")
        print()
        print("2. Use online converter:")
        print("   - https://www.markdowntopdf.com/")
        print("   - https://dillinger.io/ (export as PDF)")
        print()
        print("3. Use word processor:")
        print("   - Open FINAL_REPORT.md in Word/Google Docs")
        print("   - Export as PDF")
        print()
        print("4. Submit Markdown file (if allowed):")
        print("   - Check with instructor if .md format is acceptable")
        sys.exit(1)

if __name__ == "__main__":
    main()
