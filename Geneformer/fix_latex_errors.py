#!/usr/bin/env python3
"""
Fix LaTeX file errors: escape issues and section numbering.
"""
import re
from pathlib import Path

def fix_latex_file(tex_file):
    """Fix LaTeX file errors."""
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix escaped textbackslash issues
    # Replace \textbackslash\{\}textbf{ with \textbf{
    content = re.sub(r'\\textbackslash\{\}textbf\{', r'\\textbf{', content)
    content = re.sub(r'\\\{\}textbf\{', r'\\textbf{', content)
    content = re.sub(r'\\textbackslash\{\}', r'\\', content)
    
    # Fix duplicate section numbers (e.g., "1.2.2 1.2 Problem Statement")
    # Remove duplicate numbering patterns
    content = re.sub(r'(\d+\.\d+\.\d+)\s+(\d+\.\d+)\s+', r'\2 ', content)
    content = re.sub(r'(\d+\.\d+\.\d+)\s+(\d+\.\d+\.\d+)\s+', r'\2 ', content)
    
    # Fix section numbering issues in subsection headers
    # Remove numbers from subsection titles (LaTeX handles numbering automatically)
    content = re.sub(r'\\subsubsection\{(\d+\.\d+\.\d+)\s+([^}]+)\}', r'\\subsubsection{\2}', content)
    content = re.sub(r'\\subsubsection\{(\d+\.\d+)\s+([^}]+)\}', r'\\subsubsection{\2}', content)
    
    # Fix subsection headers that have numbers in them
    lines = content.split('\n')
    fixed_lines = []
    for line in lines:
        # Fix lines like "\subsubsection{1.2.2 1.2 Problem Statement}"
        if '\\subsubsection{' in line:
            # Remove duplicate numbering
            line = re.sub(r'\\subsubsection\{(\d+\.\d+\.\d+)\s+(\d+\.\d+)\s+([^}]+)\}', 
                         r'\\subsubsection{\3}', line)
            line = re.sub(r'\\subsubsection\{(\d+\.\d+)\s+([^}]+)\}', 
                         r'\\subsubsection{\2}', line)
        fixed_lines.append(line)
    
    content = '\n'.join(fixed_lines)
    
    # Fix any remaining escape issues
    content = content.replace('\\textbackslash{}', '\\')
    content = content.replace('\\{\\}', '')
    
    # Write fixed content
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Fixed LaTeX file: {tex_file}")

if __name__ == "__main__":
    workdir = Path(__file__).parent
    tex_file = workdir / "submission_package" / "FINAL_REPORT.tex"
    fix_latex_file(tex_file)
