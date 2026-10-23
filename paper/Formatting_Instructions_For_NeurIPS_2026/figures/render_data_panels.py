from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","DejaVu Serif"],"pdf.fonttype":42,"font.size":8.5})
fig,ax=plt.subplots(figsize=(2.65,2.2))
fig.subplots_adjust(left=.30,right=.94,bottom=.37,top=.97)
vals=np.array([[145,22,25],[630,66,49]])
colors=["#2a78d6","#eda100","#eb6834"]
left=np.zeros(2)
for col,color in enumerate(colors):
 widths=100*vals[:,col]/vals.sum(axis=1)
 ax.barh([1,0],widths,left=left,color=color,edgecolor="white",height=.49)
 for i in range(2):ax.text(left[i]+widths[i]/2,[1,0][i],str(vals[i,col]),ha="center",va="center",fontsize=8,color="white" if col==0 else "black")
 left+=widths
ax.set_yticks([1,0],["Types\n(192)","Photos\n(745)"],fontsize=8)
ax.set_xticks([0,50,100],labels=["0","50","100"],fontsize=8)
ax.set_xlim(0,100);ax.set_ylim(-.55,1.55)
ax.set_xlabel("Share of fold 4 (%)",fontsize=8)
for k in ["top","right","left"]:ax.spines[k].set_visible(False)
ax.spines["bottom"].set_color("#888888");ax.tick_params(axis="y",length=0)
handles=[plt.Rectangle((0,0),1,1,color=c) for c in colors]
fig.legend(handles,["Both sides","Front only","Back only"],loc="lower center",bbox_to_anchor=(.5,.015),ncol=3,fontsize=7.6,frameon=False,columnspacing=.6,handlelength=.8,handletextpad=.3)
fig.savefig(ROOT/"figD4_epillid_sides_compact.pdf")
plt.close(fig)
fig=plt.figure(figsize=(2.65,2.05))
photos=WORKSPACE/"pharmascan-lectrace/images"
for row,(name,ids) in enumerate([("Metformin ER 750 mg",range(0,3)),("Lisinopril 5 mg",range(3,6))]):
 y=.53 if row==0 else .04
 fig.text(.02,.96 if row==0 else .47,name,fontsize=8.5,va="top")
 for col,i in enumerate(ids):
  path=next(photos.glob(f"{i:02d}_enrol_*_crop.png"))
  a=fig.add_axes([.015+col*.33,y,.31,.34]);a.imshow(plt.imread(path));a.axis("off")
fig.savefig(ROOT/"fig04_qualitative_compact.pdf")
plt.close(fig)
