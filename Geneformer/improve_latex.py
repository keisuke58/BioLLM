#!/usr/bin/env python3
"""
Improve LaTeX file: fix escaping issues, add equations, and improve formatting.
"""
import re
from pathlib import Path

def improve_latex(tex_file):
    """Improve LaTeX file with proper formatting and equations."""
    
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Fix double escaping issues
    content = content.replace('\\textbackslash{}', '\\')
    content = content.replace('\\textbackslash{}{', '{')
    
    # Fix percentage signs
    content = content.replace('\\textbackslash{}%', '\\%')
    
    # Add equations section after Experimental Setup
    equations_section = r"""
\subsection{Mathematical Formulation}

The evaluation metrics used in this study are defined as follows:

\subsubsection{Accuracy}
The overall classification accuracy is defined as:
\begin{equation}
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} = \frac{TP + TN}{TP + TN + FP + FN}
\label{eq:accuracy}
\end{equation}
where $TP$, $TN$, $FP$, and $FN$ denote true positives, true negatives, false positives, and false negatives, respectively.

\subsubsection{Macro F1 Score}
The Macro F1 score is the unweighted mean of F1 scores across all classes:
\begin{equation}
\text{Macro F1} = \frac{1}{C} \sum_{i=1}^{C} F1_i
\label{eq:macro_f1}
\end{equation}
where $C$ is the number of classes and $F1_i$ is the F1 score for class $i$:
\begin{equation}
F1_i = \frac{2 \cdot \text{Precision}_i \cdot \text{Recall}_i}{\text{Precision}_i + \text{Recall}_i}
\label{eq:f1}
\end{equation}
with
\begin{align}
\text{Precision}_i &= \frac{TP_i}{TP_i + FP_i} \label{eq:precision} \\
\text{Recall}_i &= \frac{TP_i}{TP_i + FN_i} \label{eq:recall}
\end{align}

\subsubsection{Performance Improvement}
The absolute improvement from frozen to fine-tuned models is calculated as:
\begin{equation}
\Delta_{\text{abs}} = \text{Accuracy}_{\text{fine-tuned}} - \text{Accuracy}_{\text{frozen}}
\label{eq:abs_improvement}
\end{equation}
The relative improvement is:
\begin{equation}
\Delta_{\text{rel}} = \frac{\Delta_{\text{abs}}}{\text{Accuracy}_{\text{frozen}}} \times 100\%
\label{eq:rel_improvement}
\end{equation}

For Geneformer, we observe $\Delta_{\text{abs}} = 0.978 - 0.613 = 0.365$ (36.5 percentage points) and $\Delta_{\text{rel}} = \frac{0.365}{0.613} \times 100\% \approx 59.6\%$ as shown in Equation~\eqref{eq:abs_improvement} and Equation~\eqref{eq:rel_improvement}.

"""
    
    # Insert equations after "Implementation Details" subsection
    if '\\subsubsection{3.4 Implementation Details}' in content:
        idx = content.find('\\subsubsection{3.4 Implementation Details}')
        end_idx = content.find('\\section', idx + 1)
        if end_idx == -1:
            end_idx = content.find('\\subsection', idx + 1)
        if end_idx > 0:
            # Find the end of this subsection
            next_section = content.find('\\section', end_idx)
            if next_section == -1:
                next_section = content.find('\\subsection{4.', end_idx)
            if next_section > 0:
                content = content[:next_section] + equations_section + content[next_section:]
    
    # Fix table formatting - ensure proper alignment
    content = re.sub(r'\\begin\{tabular\}\{l+\}', 
                     lambda m: f"\\begin{{tabular}}{{{'l' * len(m.group(0).split('l'))}}}", 
                     content)
    
    # Improve figure paths - use absolute or relative paths correctly
    # Replace backslash paths with forward slashes for LaTeX
    content = re.sub(r'results\\figures\\([^}]+)', r'results/figures/\1', content)
    
    # Add proper figure references in text
    content = content.replace('UMAP visualization', 'Figure~\\ref{fig:umap_labels} shows the UMAP visualization')
    
    # Fix references to use proper citation format
    content = content.replace('Theodoris et al.', 'Theodoris et al.~\\cite{theodoris2023}')
    content = content.replace('Cui et al.', 'Cui et al.~\\cite{cui2023}')
    
    # Write improved LaTeX
    with open(tex_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Improved LaTeX file: {tex_file}")

if __name__ == "__main__":
    workdir = Path(__file__).parent
    tex_file = workdir / "submission_package" / "FINAL_REPORT.tex"
    improve_latex(tex_file)
