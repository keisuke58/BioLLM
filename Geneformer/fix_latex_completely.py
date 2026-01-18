#!/usr/bin/env python3
"""
Completely fix LaTeX file: all escape issues.
"""
import re
from pathlib import Path

def fix_latex_completely(tex_file):
    """Fix all LaTeX escape issues."""
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix all variations of textbackslash issues
    # Pattern 1: \textbackslash\{\}textbf{ -> \textbf{
    content = re.sub(r'\\textbackslash\{\}textbf\{', r'\\textbf{', content)
    # Pattern 2: \textbackslashtextbf{ -> \textbf{
    content = re.sub(r'\\textbackslashtextbf\{', r'\\textbf{', content)
    # Pattern 3: \{\}textbf{ -> \textbf{
    content = re.sub(r'\\\{\}textbf\{', r'\\textbf{', content)
    # Pattern 4: \textbackslash\{\} -> \
    content = re.sub(r'\\textbackslash\{\}', r'\\', content)
    # Pattern 5: \textbackslash{}{} -> \
    content = re.sub(r'\\textbackslash\{\}\{\}', r'\\', content)
    
    # Fix any remaining textbackslash issues
    content = content.replace('\\textbackslash{}', '\\')
    content = content.replace('\\textbackslash', '\\')
    
    # Remove duplicate section numbers in subsection titles
    # Fix patterns like "1.2.2 1.2 Problem Statement" -> "Problem Statement"
    content = re.sub(r'\\subsubsection\{(\d+\.\d+\.\d+)\s+(\d+\.\d+)\s+([^}]+)\}', 
                     r'\\subsubsection{\3}', content)
    content = re.sub(r'\\subsubsection\{(\d+\.\d+)\s+([^}]+)\}', 
                     r'\\subsubsection{\2}', content)
    
    # Write fixed content
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Completely fixed LaTeX file: {tex_file}")

if __name__ == "__main__":
    workdir = Path(__file__).parent
    tex_file = workdir / "submission_package" / "FINAL_REPORT.tex"
    fix_latex_completely(tex_file)
