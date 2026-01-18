#!/usr/bin/env python3
"""
Convert Markdown report to LaTeX format with embedded figures and equations.
This creates a publication-quality PDF.
"""
import re
import os
from pathlib import Path

def escape_latex(text):
    """Escape special LaTeX characters."""
    if not text:
        return ""
    replacements = {
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '^': r'\textasciicircum{}',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '\\': r'\textbackslash{}',
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

def process_table(lines, start_idx):
    """Process a markdown table and convert to LaTeX."""
    table_lines = []
    i = start_idx
    while i < len(lines) and '|' in lines[i]:
        if '---' not in lines[i]:  # Skip separator line
            table_lines.append(lines[i])
        i += 1
    
    if not table_lines:
        return "", start_idx
    
    # Parse header
    header = [c.strip() for c in table_lines[0].split('|')[1:-1]]
    num_cols = len(header)
    
    latex = "\\begin{table}[H]\n\\centering\n"
    latex += f"\\begin{{tabular}}{{{'l' * num_cols}}}\n"
    latex += "\\toprule\n"
    latex += ' & '.join([escape_latex(h) for h in header]) + ' \\\\\n'
    latex += "\\midrule\n"
    
    # Parse data rows
    for row_line in table_lines[1:]:
        cells = [c.strip() for c in row_line.split('|')[1:-1]]
        if len(cells) == num_cols:
            # Process bold text in cells
            processed_cells = []
            for cell in cells:
                cell = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', cell)
                processed_cells.append(escape_latex(cell))
            latex += ' & '.join(processed_cells) + ' \\\\\n'
    
    latex += "\\bottomrule\n\\end{tabular}\n"
    latex += "\\caption{Performance comparison}\n"
    latex += "\\label{tab:comparison}\n"
    latex += "\\end{table}\n\n"
    
    return latex, i

def markdown_to_latex(md_file, tex_file, figures_dir):
    """Convert Markdown to LaTeX format."""
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    # Figure mapping
    figure_map = {
        'umap_labels_pbmc3k.png': ('UMAP visualization of PBMC3k cell types', 'umap_labels'),
        'umap_geneformer_emb_pbmc3k.png': ('UMAP visualization of Geneformer embeddings', 'umap_geneformer'),
        'confusion_geneformer_pbmc3k.png': ('Confusion matrix for Geneformer (frozen)', 'confusion_geneformer'),
        'confusion_scgpt.png': ('Confusion matrix for scGPT (frozen)', 'confusion_scgpt'),
        'umap_scgpt.png': ('UMAP visualization of scGPT embeddings', 'umap_scgpt'),
    }
    
    # LaTeX document header
    latex = r"""\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{float}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{natbib}
\usepackage{xcolor}
\usepackage{enumitem}

% Page margins
\geometry{margin=2.5cm}

% Title information
\title{Understanding the Limits of Single-Cell Foundation Models on Downstream Tasks}
\author{Keisuke Nishioka (Student ID: 10081049) \\ 
        AI Foundation Models in Biomedicine, WiSe 2025/26 \\
        Leibniz University of Hannover}
\date{January 2026}

\begin{document}

\maketitle

\begin{abstract}
Single-cell foundation models have emerged as powerful paradigms for analyzing transcriptomic data. However, the conditions under which these models excel or fail remain poorly understood. This project evaluates the performance of single-cell foundation models (Geneformer and scGPT) on downstream cell type classification tasks, comparing frozen representations with fine-tuned models. We demonstrate that fine-tuning significantly improves performance, with Geneformer achieving 97.8\% accuracy after fine-tuning compared to 61.3\% with frozen representations. Our findings highlight the importance of task-specific fine-tuning for optimal performance on downstream tasks.

\textbf{Keywords:} Single-cell RNA-seq, Foundation Models, Geneformer, scGPT, Fine-tuning, Cell Type Classification
\end{abstract}

"""
    
    i = 0
    in_list = False
    list_type = None
    
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Skip horizontal rules
        if line.strip() == '---':
            i += 1
            continue
        
        # Headers
        if line.startswith('# '):
            if in_list:
                latex += "\\end{itemize}\n\n" if list_type == 'itemize' else "\\end{enumerate}\n\n"
                in_list = False
            latex += f"\n\\section{{{escape_latex(line[2:].strip())}}}\n"
        elif line.startswith('## '):
            if in_list:
                latex += "\\end{itemize}\n\n" if list_type == 'itemize' else "\\end{enumerate}\n\n"
                in_list = False
            latex += f"\n\\subsection{{{escape_latex(line[3:].strip())}}}\n"
        elif line.startswith('### '):
            if in_list:
                latex += "\\end{itemize}\n\n" if list_type == 'itemize' else "\\end{enumerate}\n\n"
                in_list = False
            latex += f"\n\\subsubsection{{{escape_latex(line[4:].strip())}}}\n"
        
        # Tables
        elif '|' in line and line.strip().startswith('|'):
            table_latex, new_i = process_table(lines, i)
            latex += table_latex
            i = new_i
            continue
        
        # Numbered lists
        elif re.match(r'^\d+\.\s+', line):
            if not in_list or list_type != 'enumerate':
                if in_list:
                    latex += "\\end{itemize}\n\n"
                latex += "\\begin{enumerate}\n"
                in_list = True
                list_type = 'enumerate'
            content = re.sub(r'^\d+\.\s+', '', line)
            # Process bold
            content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
            latex += f"\\item {escape_latex(content)}\n"
        
        # Bullet lists
        elif line.strip().startswith('- '):
            if not in_list or list_type != 'itemize':
                if in_list:
                    latex += "\\end{enumerate}\n\n"
                latex += "\\begin{itemize}\n"
                in_list = True
                list_type = 'itemize'
            content = line.strip()[2:]
            # Process bold
            content = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', content)
            latex += f"\\item {escape_latex(content)}\n"
        
        # Regular paragraphs
        elif line.strip() and not line.startswith('#'):
            if in_list:
                latex += "\\end{itemize}\n\n" if list_type == 'itemize' else "\\end{enumerate}\n\n"
                in_list = False
            
            # Process bold text
            processed = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', line)
            
            # Check for figure references
            fig_added = False
            for fig_file, (caption, label) in figure_map.items():
                if fig_file.replace('.png', '').replace('_', ' ') in processed.lower():
                    fig_path = figures_dir / fig_file
                    if fig_path.exists():
                        # Use relative path from LaTeX file location
                        rel_path = os.path.relpath(fig_path, tex_file.parent)
                        latex += f"\\begin{{figure}}[H]\n\\centering\n"
                        latex += f"\\includegraphics[width=0.8\\textwidth]{{{rel_path}}}\n"
                        latex += f"\\caption{{{caption}}}\n"
                        latex += f"\\label{{fig:{label}}}\n"
                        latex += f"\\end{{figure}}\n\n"
                        fig_added = True
                        break
            
            if not fig_added:
                latex += escape_latex(processed) + '\n\n'
        
        i += 1
    
    # Close any open list
    if in_list:
        latex += "\\end{itemize}\n\n" if list_type == 'itemize' else "\\end{enumerate}\n\n"
    
    # Add figures section if figures exist
    figures_added = False
    for fig_file, (caption, label) in figure_map.items():
        fig_path = figures_dir / fig_file
        if fig_path.exists():
            if not figures_added:
                latex += "\n\\section{Figures}\n\n"
                figures_added = True
            rel_path = os.path.relpath(fig_path, tex_file.parent)
            latex += f"\\begin{{figure}}[H]\n\\centering\n"
            latex += f"\\includegraphics[width=0.8\\textwidth]{{{rel_path}}}\n"
            latex += f"\\caption{{{caption}}}\n"
            latex += f"\\label{{fig:{label}}}\n"
            latex += f"\\end{{figure}}\n\n"
    
    # References section
    latex += r"""
\begin{thebibliography}{9}

\bibitem{theodoris2023}
Theodoris, C. V., et al. (2023). Transfer learning enables predictions in network biology. \textit{Nature}, 618(7965), 616-624.

\bibitem{cui2023}
Cui, H., et al. (2023). scGPT: Towards building a foundation model for single-cell multi-omics using generative AI. \textit{bioRxiv}.

\bibitem{kedzierska}
Kedzierska, K. Z., et al. (bioRxiv). Evaluation of single-cell foundation models. \textit{bioRxiv}.

\bibitem{boiarsky}
Boiarsky, R., et al. (bioRxiv). Systematic evaluation of single-cell foundation models. \textit{bioRxiv}.

\bibitem{10xgenomics}
10x Genomics. (2023). PBMC 68k dataset. https://www.10xgenomics.com/

\bibitem{tabula2022}
Tabula Sapiens Consortium. (2022). The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans. \textit{Science}, 376(6594), eabl4896.

\end{thebibliography}

\end{document}
"""
    
    # Write LaTeX file
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(latex)
    
    print(f"✓ LaTeX file created: {tex_file}")

if __name__ == "__main__":
    workdir = Path(__file__).parent
    md_file = workdir / "submission_package" / "FINAL_REPORT.md"
    tex_file = workdir / "submission_package" / "FINAL_REPORT.tex"
    figures_dir = workdir / "submission_package" / "results" / "figures"
    
    markdown_to_latex(md_file, tex_file, figures_dir)
