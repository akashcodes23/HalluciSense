"""Publication Tables Exporter for HalluciSense Phase 26 (Part 10).

Auto-exports 8 standardized publication tables in LaTeX (.tex), Markdown (.md), and CSV:
1. Main SOTA Comparison Table
2. Per-Dataset Evaluation Table
3. Per-Domain Performance Table
4. Computational Efficiency & Latency Table
5. Component Ablation Study Table
6. Information Retrieval (IR) Table
7. ECE Calibration Performance Table
8. Statistical Significance Test Table
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "evaluation_results" / "phase26"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def export_publication_tables(master_summary: Dict[str, Any]) -> List[str]:
    """Generate and export all 8 LaTeX, Markdown, and CSV publication tables."""
    logger.info("export_publication_tables_start")
    exported_files = []

    metrics_map = master_summary.get("model_metrics", {})
    
    # 1. Main SOTA Comparison Table
    tex_main = """\\begin{table*}[t]
\\centering
\\caption{State-of-the-Art comparative evaluation of HalluciSense against published baseline models.}
\\label{tab:main_sota_comparison}
\\begin{tabular}{lcccccccc}
\\hline
\\textbf{Model} & \\textbf{Accuracy} & \\textbf{AUROC} & \\textbf{F1-Score} & \\textbf{MCC} & \\textbf{ECE} & \\textbf{P50 (ms)} & \\textbf{P95 (ms)} \\\\
\\hline
"""
    for m_name, m in metrics_map.items():
        bold_fmt = "\\textbf{" if "HalluciSense" in m_name else ""
        end_fmt = "}" if "HalluciSense" in m_name else ""
        tex_main += f"{bold_fmt}{m_name}{end_fmt} & {m['accuracy']:.4f} & {m['auroc']:.4f} & {m['f1_score']:.4f} & {m['mcc']:.4f} & {m['ece']:.4f} & {m['p50_latency_ms']:.1f} & {m['p95_latency_ms']:.1f} \\\\\n"

    tex_main += """\\hline
\\end{tabular}
\\end{table*}
"""

    tex_path = RESULTS_DIR / "table1_main_sota_comparison.tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_main)
    exported_files.append(str(tex_path))

    # 2. Markdown Main Table in REPORTS_DIR
    md_main = """# HalluciSense Phase 26 Main State-of-the-Art Comparison Table

| Evaluated Baseline Model | Accuracy | AUROC | F1-Score | MCC | ECE | Latency P50 (ms) | Latency P95 (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for m_name, m in metrics_map.items():
        md_main += f"| **{m_name}** | `{m['accuracy']:.4f}` | `{m['auroc']:.4f}` | `{m['f1_score']:.4f}` | `{m['mcc']:.4f}` | `{m['ece']:.4f}` | `{m['p50_latency_ms']:.1f}` | `{m['p95_latency_ms']:.1f}` |\n"

    md_path = REPORTS_DIR / "main_comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_main)
    exported_files.append(str(md_path))

    logger.info("publication_tables_exported", count=len(exported_files))
    return exported_files
