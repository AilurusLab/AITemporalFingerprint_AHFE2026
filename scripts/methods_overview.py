"""Methods overview (Fig. 1), v10: three-zone pipeline (datasets, analyses,
outputs) modeled on the CoAuthor and RealHumanEval Figure 1s. All text black,
domain icons (pencil / code) instead of text tags, dataset attributions,
within-author comparison drawn from the stream itself. Times New Roman, 600 dpi."""

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import os

import fontawesomefree
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Wedge

FA = font_manager.FontProperties(fname=os.path.join(
    os.path.dirname(fontawesomefree.__file__),
    "static", "fontawesomefree", "webfonts", "fa-solid-900.ttf"))
ICON_PENCIL = "\uf303"  # pencil
ICON_CODE = "\uf121"  # code

INK = "#1a1a1a"
GRAY = "#8a8a8a"
FAINT = "#c4c4c4"
CARD = "#f5f5f4"
RED = "#c8442c"
RED_FADE = "#e8b3a6"

fig, ax = plt.subplots(figsize=(11.3, 3.6))
ax.set_xlim(0, 11.3)
ax.set_ylim(0, 3.6)
ax.axis("off")


def text(x, y, s, size=8.5, color=INK, ha="left", weight="normal", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, fontweight=weight)


def zone(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.14",
                                linewidth=0, facecolor=CARD))
    text(x + w / 2, y + h - 0.18, title, 14.5, ha="center", weight="bold")


def card(x, y, w, h, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.0 if dashed else 0,
                                edgecolor=GRAY if dashed else "none",
                                facecolor="white",
                                linestyle=(0, (4, 2)) if dashed else "-"))


def ticks(x0, y, groups, color=INK, h=0.68, lw=1.1):
    x = x0
    for gap, n, sp in groups:
        x += gap
        for _ in range(n):
            ax.plot([x, x], [y, y + h], color=color, lw=lw, solid_capstyle="butt")
            x += sp
    return x


def ai_block(x, y, w, h=0.68, label_size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01",
                                linewidth=0, facecolor=RED))
    text(x + w / 2, y + h / 2, "AI", label_size, color="white", ha="center",
         weight="bold")


def doc(x, y, w=0.34, h=0.60):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                linewidth=0.9, edgecolor=INK, facecolor="white"))
    for i in range(5):
        ax.plot([x + 0.05, x + w - 0.05], [y + h - 0.09 - i * 0.10] * 2,
                color=FAINT, lw=0.9)


def arrow(x0, y0, x1, y1, rad=0.0, color=GRAY, lw=1.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                connectionstyle=f"arc3,rad={rad}"))


def fa_icon(x, y, glyph, size=11.5, ha="center", color=INK):
    ax.text(x, y, glyph, fontsize=size, color=color, ha=ha, va="center",
            fontproperties=FA)


# ================= zone 1: datasets =================
Z1X, Z1Y, Z1W, Z1H = 0.08, 0.08, 6.34, 3.44
zone(Z1X, Z1Y, Z1W, Z1H, "Datasets")

CW, CH = 3.02, 1.46
AX, AY = 0.18, 1.74
BX, BY = 3.30, 1.74
CX, CY = 0.18, 0.18
DX, DY = 3.30, 0.18


def log_header(x, y, name, attrib, attrib_dx):
    text(x + 0.16, y + CH - 0.19, name, 13, weight="bold")
    if attrib:
        text(x + attrib_dx, y + CH - 0.19, attrib, 9.5)


def log_footer(x, y, stat, kind):
    text(x + 0.16, y + 0.20, stat, 10)
    text(x + CW - 0.16, y + 0.20, kind, 10, ha="right")


# ---- CoAuthor (writing keystrokes) ----
card(AX, AY, CW, CH)
log_header(AX, AY, "CoAuthor", "(Lee et al.)", 1.08)
fa_icon(AX + CW - 0.26, AY + CH - 0.19, ICON_PENCIL)
yv = AY + 0.42
xe = ticks(AX + 0.16, yv, [(0.0, 7, 0.034), (0.12, 5, 0.034), (0.12, 7, 0.034)])
ai_block(xe + 0.10, yv, 0.50, label_size=11)
ticks(xe + 0.72, yv, [(0.0, 6, 0.034), (0.12, 7, 0.034), (0.12, 5, 0.034)])
log_footer(AX, AY, "1,447 sessions", "human + AI suggestions")

# ---- RealHumanEval (code snapshots) ----
card(BX, BY, CW, CH)
log_header(BX, BY, "RealHumanEval", "(Mozannar et al.)", 1.52)
fa_icon(BX + CW - 0.20, BY + CH - 0.19, ICON_CODE)
yd = BY + 0.50
for i, xd in enumerate([BX + 0.18, BX + 0.90, BX + 1.62, BX + 2.34]):
    doc(xd, yd)
    if i < 3:
        ax.annotate("", xy=(xd + 0.68, yd + 0.30), xytext=(xd + 0.40, yd + 0.30),
                    arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.0))
ax.plot([BX + 0.95, BX + 1.91], [yd - 0.11, yd - 0.11], color=RED, lw=2.6,
        solid_capstyle="round")
text(BX + 1.99, yd - 0.11, "AI", 10, color=RED, weight="bold")
log_footer(BX, BY, "243 records", "human + AI suggestions")

# ---- CS1 (pre-LLM keyboard baseline) ----
card(CX, CY, CW, CH)
log_header(CX, CY, "CS1 2019", "(Edwards & Hart)", 0.98)
fa_icon(CX + CW - 0.28, CY + CH - 0.19, ICON_CODE)
yv = CY + 0.42
ticks(CX + 0.16, yv, [(0.0, 9, 0.034), (0.11, 7, 0.034), (0.11, 9, 0.034),
                      (0.11, 6, 0.034), (0.11, 8, 0.034), (0.11, 7, 0.034),
                      (0.11, 8, 0.034)])
log_footer(CX, CY, "5.1M keystrokes", "human only")

# ---- simulated delegation ----
card(DX, DY, CW, CH, dashed=True)
log_header(DX, DY, "Delegation", None, 0)
fa_icon(DX + CW - 0.64, DY + CH - 0.19, ICON_PENCIL)
fa_icon(DX + CW - 0.28, DY + CH - 0.19, ICON_CODE)
yv = DY + 0.42
ai_block(DX + 0.16, yv, 1.55, label_size=14.5)
ticks(DX + 1.86, yv, [(0.0, 2, 0.034), (0.16, 3, 0.034), (0.20, 2, 0.034),
                      (0.22, 2, 0.034)])
log_footer(DX, DY, "400 writing, 150 code", "AI delegation")

# ================= zone 2: analyses =================
Z2X, Z2Y, Z2W, Z2H = 6.86, 0.08, 2.10, 3.44
zone(Z2X, Z2Y, Z2W, Z2H, "Analyses")

ACX, ACW, ACH = 6.96, 1.90, 0.94
AYS = [2.26, 1.22, 0.18]

# ---- within-author tests: own burst and AI insertion from the same stream ----
y = AYS[0]
card(ACX, y, ACW, ACH)
ys, th, sp = y + 0.74, 0.11, 0.024
xe = ticks(7.11, ys, [(0.0, 5, sp), (0.07, 4, sp)], h=th, lw=0.9)
bx0 = xe + 0.05
ax.add_patch(FancyBboxPatch((bx0, ys), 0.22, th, boxstyle="round,pad=0.008",
                            linewidth=0, facecolor=RED))
ticks(bx0 + 0.30, ys, [(0.0, 4, sp), (0.06, 5, sp), (0.06, 4, sp)], h=th, lw=0.9)
yb = y + 0.38
ax.plot([7.20, 7.40], [ys - 0.02, yb + 0.14], color=FAINT, lw=0.9)
ax.plot([bx0 + 0.11, 8.27], [ys - 0.02, yb + 0.14], color=FAINT, lw=0.9)
ax.add_patch(FancyBboxPatch((7.14, yb), 0.50, 0.12, boxstyle="round,pad=0.01",
                            linewidth=0, facecolor=INK))
text(7.39, yb + 0.06, "human", 8.5, color="white", ha="center", weight="bold")
text(7.74, yb + 0.06, "vs", 10)
ax.add_patch(FancyBboxPatch((7.92, yb), 0.65, 0.12, boxstyle="round,pad=0.01",
                            linewidth=0, facecolor=RED))
text(8.245, yb + 0.06, "AI", 9, color="white", ha="center", weight="bold")
text(ACX + ACW / 2, y + 0.16, "Within-author tests", 11, ha="center", weight="bold")


def mini_doc(x, y, w, h, line_colors):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                linewidth=0.9, edgecolor=INK, facecolor="white"))
    n = len(line_colors)
    for i, lc in enumerate(line_colors):
        if lc is None:
            continue
        yy = y + h - 0.09 - i * (h - 0.16) / (n - 1)
        ax.plot([x + 0.06, x + w - 0.06], [yy, yy], color=lc, lw=1.5,
                solid_capstyle="round")


# ---- content survival: inserted content into the final document ----
y = AYS[1]
card(ACX, y, ACW, ACH)
mini_doc(7.11, y + 0.30, 0.48, 0.52, [INK, RED, INK, RED, INK])
arrow(7.70, y + 0.56, 8.12, y + 0.56, color=GRAY, lw=1.1)
mini_doc(8.23, y + 0.30, 0.48, 0.52, [INK, RED, INK, None, INK])
text(ACX + ACW / 2, y + 0.16, "Content survival", 11, ha="center", weight="bold")

# ---- 3-class classification: each stream is assigned a class ----
y = AYS[2]
card(ACX, y, ACW, ACH)
rows = [y + 0.72, y + 0.53, y + 0.34]
th, sp = 0.11, 0.024
ticks(7.11, rows[0], [(0.0, 6, sp), (0.07, 5, sp), (0.07, 6, sp)], h=th, lw=0.9)
xe = ticks(7.11, rows[1], [(0.0, 5, sp), (0.07, 4, sp)], h=th, lw=0.9)
ax.add_patch(FancyBboxPatch((xe + 0.05, rows[1]), 0.16, th,
                            boxstyle="round,pad=0.008", linewidth=0, facecolor=RED))
ticks(xe + 0.26, rows[1], [(0.0, 5, sp)], h=th, lw=0.9)
ax.add_patch(FancyBboxPatch((7.11, rows[2]), 0.42, th,
                            boxstyle="round,pad=0.008", linewidth=0, facecolor=RED))
ticks(7.60, rows[2], [(0.0, 2, sp), (0.06, 2, sp)], h=th, lw=0.9)
for ry, badge in zip(rows, ["open", "half", "red"]):
    cy = ry + th / 2
    ax.annotate("", xy=(8.30, cy), xytext=(8.10, cy),
                arrowprops=dict(arrowstyle="-|>", color=FAINT, lw=1.0))
    if badge == "open":
        ax.add_patch(Circle((8.45, cy), 0.045, facecolor="white", edgecolor=INK,
                            linewidth=1.0))
    elif badge == "half":
        ax.add_patch(Wedge((8.45, cy), 0.045, 90, 270, facecolor=INK, linewidth=0))
        ax.add_patch(Circle((8.45, cy), 0.045, facecolor="none", edgecolor=INK,
                            linewidth=1.0))
    else:
        ax.add_patch(Circle((8.45, cy), 0.045, facecolor=RED, linewidth=0))
text(ACX + ACW / 2, y + 0.16, "3-class classification", 9.3, ha="center",
     weight="bold")

# ================= zone 3: outputs =================
Z3X, Z3Y, Z3W, Z3H = 9.42, 0.08, 1.80, 3.44
zone(Z3X, Z3Y, Z3W, Z3H, "Outputs")

OCX, OCW = 9.52, 1.60
ox = OCX + OCW / 2

# ---- effect sizes: shifted distributions with mean gap ----
y = AYS[0]
card(OCX, y, OCW, ACH)
xb = np.linspace(0, 1, 160)
base = y + 0.34
for mu, col in [(0.30, INK), (0.68, RED)]:
    yy = base + 0.30 * np.exp(-((xb - mu) ** 2) / (2 * 0.13 ** 2))
    ax.plot(9.70 + 1.24 * xb, yy, color=col, lw=1.4)
ax.plot([9.70, 10.94], [base, base], color=FAINT, lw=0.9)
ax.annotate("", xy=(9.70 + 1.24 * 0.68, y + 0.70), xytext=(9.70 + 1.24 * 0.30, y + 0.70),
            arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.0))
text(ox, y + 0.16, "Effect sizes", 11, ha="center", weight="bold")

# ---- retention rates: how much AI content persists, by domain ----
y = AYS[1]
card(OCX, y, OCW, ACH)
fa_icon(9.78, y + 0.735, ICON_PENCIL, 10)
ax.add_patch(FancyBboxPatch((9.90, y + 0.68), 0.44, 0.11,
                            boxstyle="round,pad=0.008", linewidth=0, facecolor=RED))
arrow(10.42, y + 0.735, 10.56, y + 0.735, color=GRAY, lw=1.0)
ax.add_patch(FancyBboxPatch((10.62, y + 0.68), 0.36, 0.11,
                            boxstyle="round,pad=0.008", linewidth=0, facecolor=RED))
fa_icon(9.78, y + 0.475, ICON_CODE, 10)
ax.add_patch(FancyBboxPatch((9.90, y + 0.42), 0.44, 0.11,
                            boxstyle="round,pad=0.008", linewidth=0, facecolor=RED))
arrow(10.42, y + 0.475, 10.56, y + 0.475, color=GRAY, lw=1.0)
ax.add_patch(FancyBboxPatch((10.62, y + 0.42), 0.09, 0.11,
                            boxstyle="round,pad=0.008", linewidth=0, facecolor=RED))
text(ox, y + 0.16, "Retention rates", 11, ha="center", weight="bold")

# ---- delegation separability: gap between delegation and the rest ----
y = AYS[2]
card(OCX, y, OCW, ACH)
yl = y + 0.60
ax.plot([9.66, 10.98], [yl, yl], color=FAINT, lw=0.9)
for xx, kind in [(9.76, "dot"), (9.87, "open"), (9.97, "dot"), (10.08, "open"),
                 (10.17, "dot")]:
    if kind == "dot":
        ax.add_patch(Circle((xx, yl), 0.030, facecolor=INK, linewidth=0))
    else:
        ax.add_patch(Circle((xx, yl), 0.030, facecolor="white", edgecolor=INK,
                            linewidth=0.9))
ax.plot([10.48, 10.48], [y + 0.44, y + 0.76], color=GRAY, lw=1.0,
        linestyle=(0, (3, 2)))
for xx in [10.70, 10.81, 10.92]:
    ax.add_patch(Rectangle((xx - 0.028, yl - 0.028), 0.056, 0.056, facecolor=RED,
                           linewidth=0))
text(9.97, y + 0.38, "authentic", 9.5, ha="center")
text(10.81, y + 0.38, "delegation", 9.5, ha="center")
text(ox, y + 0.15, "Delegation separability", 11, ha="center", weight="bold")

# ================= arrows =================
arrow(6.46, 2.50, 6.82, 2.10, rad=-0.25)
arrow(6.46, 0.90, 6.82, 1.30, rad=0.25)
for yy in AYS:
    arrow(9.00, yy + ACH / 2, 9.38, yy + ACH / 2)

fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig("results/fig0_methods_overview.png", dpi=600, facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
print("wrote results/fig0_methods_overview.png")
