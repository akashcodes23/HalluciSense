"""
HalluciSense Research Edition Asset Generator.
Generates all 13 research evaluation artifacts:
1. Benchmark figures (PNG, SVG, PDF)
2. IEEE Research Paper (LaTeX & Markdown)
3. Scientific Leaderboard
4. Reproducibility Package Manifest
"""
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from evaluation.phase11.module11_10_figures import PublicationFigureRenderer
from evaluation.phase11.module11_11_reproducibility import ReproducibilityPackageBuilder
from evaluation.phase11.module11_12_paper_generator import IEEEPaperGenerator
from evaluation.phase11.module11_13_leaderboard import ScientificLeaderboardRenderer

def generate_all_research_assets():
    output_dir = backend_dir / "reports" / "phase11_research"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("==========================================================================")
    print("GENERATING HALLUCISENSE RESEARCH EDITION SCIENTIFIC ASSETS")
    print("==========================================================================")

    # 1. Publication Figures
    print("\n[1/4] Rendering Publication Figures (ROC Curves, PR Curves, Calibration)...")
    fig_renderer = PublicationFigureRenderer()
    fig_paths = fig_renderer.render_all_figures(output_dir / "figures")
    print(f"✓ Rendered {len(fig_paths)} figure files (PNG, SVG, PDF) in {output_dir / 'figures'}")

    # 2. Reproducibility Package
    print("\n[2/4] Building Scientific Reproducibility Package...")
    repro_builder = ReproducibilityPackageBuilder()
    repro_meta = repro_builder.generate_package(output_dir / "reproducibility")
    print(f"✓ Created reproducibility package in {output_dir / 'reproducibility'}")

    # 3. IEEE Research Paper Manuscript
    print("\n[3/4] Generating IEEE Format Research Paper (LaTeX & Markdown)...")
    paper_gen = IEEEPaperGenerator()
    paper_files = paper_gen.generate_paper(output_dir / "paper")
    print(f"✓ IEEE Paper generated in {output_dir / 'paper'}")

    # 4. Scientific Leaderboard
    print("\n[4/4] Rendering SOTA Scientific Benchmark Leaderboard...")
    lb_renderer = ScientificLeaderboardRenderer()
    lb_files = lb_renderer.generate_leaderboard(output_dir / "leaderboard")
    print(f"✓ Leaderboard generated in {output_dir / 'leaderboard'}")

    print("\n==========================================================================")
    print("SUCCESS: HALLUCISENSE RESEARCH EDITION PACKAGE GENERATED!")
    print(f"Output Directory: {output_dir}")
    print("==========================================================================")

if __name__ == "__main__":
    generate_all_research_assets()
