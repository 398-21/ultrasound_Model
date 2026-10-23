from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "pdf.fonttype": 42, "font.size": 9})
fig, ax = plt.subplots(figsize=(5.5, 3.1))
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set(xlim=(0, 5.5), ylim=(0, 3.1)); ax.axis("off")
def box(x,y,w,h,title,body,edge="#2a78d6",fill="#edf4fc"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.05",facecolor=fill,edgecolor=edge,linewidth=1.1))
    ax.text(x+w/2,y+h-0.13,title,ha="center",va="center",fontsize=9.3,fontweight="bold")
    ax.text(x+w/2,y+(h-0.22)/2,body,ha="center",va="center",fontsize=8.5,linespacing=1.15)
def arrow(start,end):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=8,linewidth=1,color="#444444",shrinkA=1,shrinkB=1))
ax.text(1.61,3.01,"Parallel backbones",ha="center",fontsize=9,fontstyle="italic")
ax.text(4.04,3.01,"Fusion and refinement",ha="center",fontsize=9,fontstyle="italic")
box(.03,1.55,.56,.52,"RGB", "224 x 224",edge="#888888",fill="#f6f6f6")
box(.86,2.16,1.51,.72,"VGG16 + CBAM","ImageNet\nGlobal pooling, Dense 256")
box(.86,1.12,1.51,.72,"ViT-B/16","ImageNet-21k\nPooler output: 768")
box(2.73,1.63,1.03,.80,"Capsule fusion","256 dimensions\nRouting: 2 iterations")
box(4.10,1.63,1.34,.80,"Refinement","4 tokens x 64\n2 layers, 4 heads\nMean pool: 64")
arrow((.59,1.85),(.84,2.52));arrow((.59,1.75),(.84,1.48))
arrow((2.39,2.51),(2.71,2.13));arrow((2.39,1.48),(2.71,1.91));arrow((3.78,2.03),(4.08,2.03))
for x,y,t in [(2.55,2.58,"256"),(2.55,1.43,"768"),(3.94,2.17,"256")]:ax.text(x,y,t,ha="center",fontsize=8.5)
box(.20,.14,1.73,.66,"Embedding","256 dimensions, L2 norm\nArcFace: s = 32, m = 0.3",edge="#eb6834",fill="#fff2eb")
box(2.24,.14,1.33,.66,"Shape","Softmax\nCross-entropy x 0.2",edge="#1baf7a",fill="#eaf7f2")
box(3.87,.14,1.36,.66,"Colour","Sigmoid\nBinary cross-entropy x 0.2",edge="#1baf7a",fill="#eaf7f2")
ax.plot([4.77,4.77,1.065],[1.61,1.00,1.00],color="#444444",linewidth=1)
ax.text(4.88,1.30,"64",fontsize=8.5)
for x in [1.065,2.905,4.55]:arrow((x,1.00),(x,.82))
fig.savefig(ROOT/"fig02_architecture_readable.pdf")
fig.savefig(ROOT/"review/revisions/architecture_readable.png",dpi=200)
plt.close(fig)
