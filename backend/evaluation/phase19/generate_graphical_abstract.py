"""Generate clean scientific graphical abstract for Elsevier submission."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "submission" / "graphical_abstract"
OUT_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Main Flow Elements
ax.text(0.5, 0.92, "User Query / LLM Response Generation", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", fc="#EFF6FF", ec="#1E40AF", lw=1.8), fontsize=11, fontweight="bold")

ax.annotate("", xy=(0.5, 0.82), xytext=(0.5, 0.87), arrowprops=dict(arrowstyle="->", lw=2, color="#1E3A8A"))

ax.text(0.5, 0.77, "Atomic Claim Decomposition ($\\{c_1, c_2, \\dots, c_K\\}$)", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FEF3C7", ec="#92400E", lw=1.5), fontsize=10)

ax.annotate("", xy=(0.20, 0.63), xytext=(0.45, 0.72), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4B5563"))
ax.annotate("", xy=(0.50, 0.63), xytext=(0.50, 0.72), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4B5563"))
ax.annotate("", xy=(0.80, 0.63), xytext=(0.55, 0.72), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4B5563"))

# Tri-Pillar Signals
ax.text(0.20, 0.58, "Pillar 1: Evidence Grounding\n(BM25 + FAISS + DeBERTa NLI)", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#D1FAE5", ec="#065F46", lw=1.5), fontsize=9)
ax.text(0.50, 0.58, "Pillar 2: Predictive Confidence\n(Token Entropy \& Gap)", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#E0E7FF", ec="#3730A3", lw=1.5), fontsize=9)
ax.text(0.80, 0.58, "Pillar 3: Semantic Consistency\n(Multi-Sample Embeddings)", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FCE7F3", ec="#831843", lw=1.5), fontsize=9)

ax.annotate("", xy=(0.5, 0.44), xytext=(0.20, 0.51), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4B5563"))
ax.annotate("", xy=(0.5, 0.44), xytext=(0.50, 0.51), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4B5563"))
ax.annotate("", xy=(0.5, 0.44), xytext=(0.80, 0.51), arrowprops=dict(arrowstyle="->", lw=1.5, color="#4B5563"))

# Adaptive Fusion & Calibration
ax.text(0.5, 0.38, "Availability-Aware Adaptive Fusion ($H_{\\text{adaptive}} = \\frac{\\sum m_i r_i w_i S_i}{\\sum m_i r_i w_i}$)\nDynamic Indicator Mask $\\mathbf{m} \\in \\{0, 1\\}^3$ + Platt Logistic Scaling", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.6", fc="#2563EB", ec="#1E3A8A", lw=2.0), fontsize=10, color="white", fontweight="bold")

ax.annotate("", xy=(0.30, 0.25), xytext=(0.45, 0.30), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E3A8A"))
ax.annotate("", xy=(0.70, 0.25), xytext=(0.55, 0.30), arrowprops=dict(arrowstyle="->", lw=1.5, color="#1E3A8A"))

# Decision & Safety Gates
ax.text(0.30, 0.20, "Selective Verification Gate\n(Risk = 0.00\% @ 80\% Cov)", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#DCFCE7", ec="#15803D", lw=1.5), fontsize=9, fontweight="bold")
ax.text(0.70, 0.20, "Selective Abstention Gate\n(Insufficient Evidence)", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#FEE2E2", ec="#B91C1C", lw=1.5), fontsize=9, fontweight="bold")

ax.annotate("", xy=(0.30, 0.08), xytext=(0.30, 0.14), arrowprops=dict(arrowstyle="->", lw=1.5, color="#15803D"))

# Closed-Loop Repair
ax.text(0.30, 0.04, "Deterministic Symbolic Repair $\\to$ Independent Reverification Gate ($H_{\\text{post}} < 0.20$)\n[CSR = 88.4\%, CIHR = 2.1\%]", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", fc="#F3F4F6", ec="#374151", lw=1.2), fontsize=9)

ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "graphical_abstract.png", dpi=300)
fig.savefig(OUT_DIR / "graphical_abstract.svg")
plt.close(fig)
print("Graphical abstract generated successfully.")
