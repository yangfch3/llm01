"""Generate autoregressive inference diagram for non-technical readers.

Output: fig_autoregressive.png in the same directory.

Shows:
1. Token sequence with curved arrows from each previous token to the target [?]
2. Bar chart of next-token candidate probabilities
3. Highlighted selected token
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch

# ─── Config ───────────────────────────────────────────────────────────────
TOKENS = ["<s>", "a", "robot", "must", "obey", "the", "orders", "given"]
# Fake attention weights (illustrative, sum to ~1)
# "a" and "robot" dominate; others are negligible
ATTN_WEIGHTS = [0.005, 0.30, 0.50, 0.008, 0.007, 0.006, 0.005, 0.005]
CANDIDATES = {"it": 0.32, "the": 0.18, "all": 0.12, "orders": 0.08, "...": 0.30}
CHOSEN = "it"

FIG_W, FIG_H = 14, 5.5
OUTPUT_PATH = Path(__file__).parent.parent.parent / "Doc" / "Courseware" / "ch07-generation" / "fig_autoregressive.png"

# ─── Draw ─────────────────────────────────────────────────────────────────
fig, (ax_seq, ax_bar) = plt.subplots(
    1, 2, figsize=(FIG_W, FIG_H), gridspec_kw={"width_ratios": [3, 1.2]}
)
fig.patch.set_facecolor("white")

# === Left panel: sequence + arrows ===
ax_seq.set_xlim(-0.5, len(TOKENS) + 1.5)
ax_seq.set_ylim(-1.8, 3.5)
ax_seq.axis("off")
ax_seq.set_aspect("equal")

BOX_W, BOX_H = 0.85, 0.6
Y_TOK = 0.0

# Draw existing token boxes
for i, tok in enumerate(TOKENS):
    x = i
    rect = mpatches.FancyBboxPatch(
        (x - BOX_W / 2, Y_TOK - BOX_H / 2),
        BOX_W,
        BOX_H,
        boxstyle="round,pad=0.05",
        facecolor="#e3f2fd",
        edgecolor="#1976d2",
        linewidth=1.5,
    )
    ax_seq.add_patch(rect)
    ax_seq.text(x, Y_TOK, tok, ha="center", va="center", fontsize=10, fontweight="bold")

# Draw target position [?]
target_x = len(TOKENS)
rect_target = mpatches.FancyBboxPatch(
    (target_x - BOX_W / 2, Y_TOK - BOX_H / 2),
    BOX_W,
    BOX_H,
    boxstyle="round,pad=0.05",
    facecolor="#fff9c4",
    edgecolor="#f57f17",
    linewidth=2.5,
)
ax_seq.add_patch(rect_target)
ax_seq.text(
    target_x, Y_TOK, "?", ha="center", va="center", fontsize=14, fontweight="bold", color="#e65100"
)

# Draw curved arrows: each token top → target top (arching upward)
# Arrow thickness & opacity proportional to attention weight
max_w = max(ATTN_WEIGHTS)
for i in range(len(TOKENS)):
    w = ATTN_WEIGHTS[i]
    distance = len(TOKENS) - i
    alpha = 0.3 + 0.7 * (w / max_w)
    lw = 0.8 + 2.2 * (w / max_w)
    # Arch radius: farther tokens need bigger arc to avoid overlap
    rad = 0.15 + 0.08 * distance
    arrow = FancyArrowPatch(
        (i, Y_TOK + BOX_H / 2 + 0.03),
        (target_x, Y_TOK + BOX_H / 2 + 0.03),
        connectionstyle=f"arc3,rad=-{rad}",
        arrowstyle="->,head_width=4,head_length=3",
        color="#1976d2",
        alpha=alpha,
        linewidth=lw,
    )
    ax_seq.add_patch(arrow)

    # Attention weight label above each token box (orange)
    label_text = f".{int(w * 100):02d}" if w >= 0.01 else "<.01"
    ax_seq.text(
        i, Y_TOK + BOX_H / 2 + 0.15,
        label_text,
        ha="center", va="bottom",
        fontsize=8,
        fontweight="bold" if w >= 0.1 else "normal",
        color="#e65100",
    )

# Title annotation
ax_seq.text(
    (len(TOKENS) - 1) / 2,
    3.2,
    "Every previous token influences the prediction of the next one",
    ha="center",
    va="center",
    fontsize=12,
    fontstyle="italic",
    color="#333333",
)

# Selected token annotation
ax_seq.text(
    target_x,
    -1.2,
    f"Selected: {CHOSEN}",
    ha="center",
    va="center",
    fontsize=12,
    fontweight="bold",
    color="#2e7d32",
    bbox={"facecolor": "#e8f5e9", "edgecolor": "#2e7d32", "boxstyle": "round,pad=0.3"},
)

# Small arrow from "Selected" label up to the ? box
ax_seq.annotate(
    "",
    xy=(target_x, Y_TOK - BOX_H / 2 - 0.05),
    xytext=(target_x, -0.85),
    arrowprops={"arrowstyle": "->,head_width=0.2", "color": "#2e7d32", "lw": 1.5},
)

# === Right panel: candidate probability bar chart ===
words = list(CANDIDATES.keys())
probs = list(CANDIDATES.values())
# Chosen = green, "..." = light gray (not a real candidate), others = blue-gray
colors = []
for w in words:
    if w == CHOSEN:
        colors.append("#2e7d32")
    elif w == "...":
        colors.append("#cfd8dc")
    else:
        colors.append("#78909c")

bars = ax_bar.barh(words, probs, color=colors, edgecolor="white", height=0.55)
ax_bar.set_xlim(0, 0.45)
ax_bar.set_title("Next-token candidates", fontsize=12, fontweight="bold", pad=10)
ax_bar.invert_yaxis()

# X-axis as percentage
ax_bar.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
ax_bar.set_xlabel("")

# Probability labels on bars
for bar, prob, word in zip(bars, probs, words):
    ax_bar.text(
        bar.get_width() + 0.008,
        bar.get_y() + bar.get_height() / 2,
        f"{prob:.0%}",
        va="center",
        fontsize=10,
        fontweight="bold" if word == CHOSEN else "normal",
        color="#2e7d32" if word == CHOSEN else "#555555",
    )

ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)

# Make y-axis labels bold for chosen
for label in ax_bar.get_yticklabels():
    if label.get_text() == CHOSEN:
        label.set_fontweight("bold")
        label.set_color("#2e7d32")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTPUT_PATH}")
