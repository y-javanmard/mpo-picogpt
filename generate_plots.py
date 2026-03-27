"""
generate_plots.py  —  MPO-PicoGPT publication figures
Run: python generate_plots.py
Put all fig*.pdf into the figs/ folder on Overleaf.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path

OUT = Path(".")

# mathtext + STIX: renders \chi, \mathbf, subscripts correctly WITHOUT needing LaTeX
plt.rcParams.update({
    "text.usetex":       False,
    "mathtext.fontset":  "stix",
    "font.family":       "STIXGeneral",
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "legend.fontsize":   12,        # large enough for chi to be readable
    "legend.framealpha": 0.93,
    "legend.edgecolor":  "#cccccc",
    "legend.borderpad":  0.6,
    "figure.dpi":        150,
})

# ── Data ──────────────────────────────────────────────────────────────────────
STEPS = np.array([0,100,200,300,400,500,600,700,800,900,
                  1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
DATA = {
    "Dense": {
        "train":[4.17,3.21,2.72,2.41,2.21,2.06,1.95,1.87,1.81,1.77,1.74,1.72,1.70,1.69,1.68,1.67,1.67,1.66,1.66,1.65,1.65],
        "val":  [4.19,3.29,2.83,2.55,2.37,2.24,2.14,2.07,2.02,1.98,1.95,1.93,1.91,1.90,1.89,1.88,1.87,1.87,1.86,1.86,1.86],
        "acc":  [0.014,0.072,0.152,0.224,0.289,0.341,0.383,0.414,0.438,0.456,0.471,0.482,0.491,0.499,0.505,0.510,0.515,0.519,0.522,0.526,0.528],
    },
    r"MPO $\chi=4$": {
        "train":[4.17,3.51,3.12,2.87,2.68,2.54,2.44,2.36,2.30,2.26,2.23,2.20,2.18,2.16,2.14,2.13,2.12,2.11,2.11,2.10,2.10],
        "val":  [4.19,3.57,3.20,2.97,2.79,2.66,2.56,2.49,2.43,2.39,2.36,2.33,2.31,2.29,2.27,2.26,2.25,2.24,2.23,2.23,2.22],
        "acc":  [0.014,0.053,0.091,0.136,0.175,0.209,0.238,0.262,0.283,0.299,0.313,0.323,0.332,0.339,0.346,0.351,0.356,0.360,0.363,0.366,0.368],
    },
    r"MPO $\chi=8$": {
        "train":[4.17,3.34,2.89,2.59,2.38,2.22,2.11,2.03,1.97,1.93,1.90,1.87,1.85,1.84,1.83,1.82,1.81,1.80,1.80,1.79,1.79],
        "val":  [4.19,3.41,2.98,2.71,2.52,2.37,2.26,2.18,2.13,2.09,2.06,2.03,2.01,2.00,1.99,1.98,1.97,1.96,1.96,1.95,1.95],
        "acc":  [0.014,0.062,0.118,0.182,0.236,0.282,0.322,0.352,0.375,0.393,0.406,0.418,0.428,0.436,0.443,0.449,0.454,0.458,0.462,0.465,0.468],
    },
    r"MPO $\chi=16$": {
        "train":[4.17,3.26,2.77,2.47,2.27,2.11,2.00,1.92,1.86,1.82,1.78,1.76,1.74,1.73,1.72,1.71,1.71,1.70,1.70,1.70,1.69],
        "val":  [4.19,3.33,2.87,2.59,2.42,2.27,2.17,2.09,2.03,1.99,1.96,1.93,1.91,1.90,1.89,1.88,1.87,1.87,1.86,1.86,1.85],
        "acc":  [0.014,0.068,0.140,0.209,0.268,0.317,0.359,0.394,0.421,0.441,0.455,0.466,0.475,0.483,0.490,0.496,0.501,0.506,0.510,0.513,0.516],
    },
    r"MPO $\chi=32$": {
        "train":[4.17,3.22,2.73,2.42,2.22,2.07,1.96,1.88,1.82,1.78,1.75,1.73,1.71,1.69,1.68,1.67,1.67,1.66,1.66,1.65,1.65],
        "val":  [4.19,3.30,2.84,2.56,2.38,2.25,2.15,2.07,2.02,1.98,1.95,1.93,1.91,1.90,1.89,1.88,1.87,1.87,1.86,1.86,1.85],
        "acc":  [0.014,0.071,0.150,0.221,0.286,0.338,0.380,0.411,0.435,0.453,0.468,0.479,0.488,0.496,0.502,0.507,0.512,0.516,0.520,0.523,0.524],
    },
}
CHI_VALS = [4,8,16,32]
LAYER_ERRORS = {
    r"$\mathbf{W}_{Q/K/V/O}$ (128$\times$128, $L$=2)": [0.395,0.261,0.135,0.065],
    r"$\mathbf{W}_1$ up-proj  (512$\times$128, $L$=3)": [0.318,0.196,0.098,0.047],
    r"$\mathbf{W}_2$ down-proj (128$\times$512, $L$=3)":[0.336,0.209,0.104,0.051],
    r"$\mathbf{W}_\mathrm{LM}$ (65$\times$128, $L$=2)": [0.421,0.283,0.148,0.072],
}
PARAMS = {"Dense":1020224, r"MPO $\chi=4$":78336, r"MPO $\chi=8$":110592,
          r"MPO $\chi=16$":191872, r"MPO $\chi=32$":408832}
LABELS_PARAMS = {
    "Dense":          r"Dense  (1.02 M)",
    r"MPO $\chi=4$":  r"MPO $\chi$=4  (78 K)",
    r"MPO $\chi=8$":  r"MPO $\chi$=8  (111 K)",
    r"MPO $\chi=16$": r"MPO $\chi$=16  (192 K)",
    r"MPO $\chi=32$": r"MPO $\chi$=32  (409 K)",
}
STYLE = {
    "Dense":          {"color":"#1a3a5c","ls":"-",           "lw":2.2,"marker":"o"},
    r"MPO $\chi=4$":  {"color":"#d4541a","ls":"--",          "lw":1.8,"marker":"s"},
    r"MPO $\chi=8$":  {"color":"#b03060","ls":"-.",          "lw":1.8,"marker":"^"},
    r"MPO $\chi=16$": {"color":"#6c3483","ls":":",           "lw":1.8,"marker":"D"},
    r"MPO $\chi=32$": {"color":"#1a7a4a","ls":(0,(5,2,1,2)),"lw":1.8,"marker":"v"},
}
LSTYLE = {
    r"$\mathbf{W}_{Q/K/V/O}$ (128$\times$128, $L$=2)": {"color":"#1a3a5c","marker":"o","ls":"-"},
    r"$\mathbf{W}_1$ up-proj  (512$\times$128, $L$=3)": {"color":"#1a7a4a","marker":"s","ls":"--"},
    r"$\mathbf{W}_2$ down-proj (128$\times$512, $L$=3)":{"color":"#6c3483","marker":"^","ls":"-."},
    r"$\mathbf{W}_\mathrm{LM}$ (65$\times$128, $L$=2)": {"color":"#d4541a","marker":"D","ls":":"},
}

def pub_style(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9); ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(labelsize=11,length=4,width=0.9)
    ax.grid(True,linestyle="--",linewidth=0.5,alpha=0.45,color="#aaaaaa")
    ax.set_axisbelow(True)

def pct_fmt(y,_): return f"{y:.0f}%"

def save(fig,stem,dpi=300):
    for ext in ["pdf","png"]:
        p=OUT/f"{stem}.{ext}"; fig.savefig(p,dpi=dpi,bbox_inches="tight")
        print(f"  {p}")

LEG = dict(frameon=True, handlelength=2.4, labelspacing=0.45, borderpad=0.6)

# ── Fig 1 ─────────────────────────────────────────────────────────────────────
def fig_reconstruction():
    fig,ax=plt.subplots(figsize=(7,4.8)); pub_style(ax)
    for lbl,errs in LAYER_ERRORS.items():
        s=LSTYLE[lbl]
        ax.plot(CHI_VALS,errs,color=s["color"],ls=s["ls"],
                marker=s["marker"],markersize=7,lw=2.0,label=lbl)
    ax.set_xscale("log",base=2); ax.set_xticks(CHI_VALS)
    ax.set_xticklabels([str(c) for c in CHI_VALS])
    ax.set_xlabel(r"Bond dimension $\chi$")
    ax.set_ylabel(r"Relative error $\|W - \hat{W}\|_F / \|W\|_F$")
    ax.set_title("MPO Reconstruction Error per Layer"); ax.set_ylim(0,0.48)
    ax.legend(loc="upper right",fontsize=10,**{k:v for k,v in LEG.items() if k!="handlelength"},handlelength=2.2)
    fig.tight_layout(); save(fig,"fig1_reconstruction_error"); plt.close(fig)

# ── Fig 2 ─────────────────────────────────────────────────────────────────────
def fig_train_loss():
    fig,ax=plt.subplots(figsize=(6.5,4.5)); pub_style(ax)
    for lbl,d in DATA.items():
        s=STYLE[lbl]; ax.plot(STEPS,d["train"],color=s["color"],ls=s["ls"],lw=s["lw"],label=lbl)
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Training Loss"); ax.set_xlim(0,2000); ax.set_ylim(1.55,4.4)
    ax.legend(loc="upper right",fontsize=12,**LEG)
    fig.tight_layout(); save(fig,"fig2_train_loss"); plt.close(fig)

# ── Fig 3 ─────────────────────────────────────────────────────────────────────
def fig_val_loss():
    fig,ax=plt.subplots(figsize=(6.5,4.5)); pub_style(ax)
    for lbl,d in DATA.items():
        s=STYLE[lbl]; ax.plot(STEPS,d["val"],color=s["color"],ls=s["ls"],lw=s["lw"],label=lbl)
    ax.set_xlabel("Training step"); ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Validation Loss"); ax.set_xlim(0,2000); ax.set_ylim(1.78,4.4)
    ax.legend(loc="upper right",fontsize=12,**LEG)
    fig.tight_layout(); save(fig,"fig3_val_loss"); plt.close(fig)

# ── Fig 4 ─────────────────────────────────────────────────────────────────────
def fig_accuracy():
    fig,ax=plt.subplots(figsize=(6.5,4.5)); pub_style(ax)
    for lbl,d in DATA.items():
        s=STYLE[lbl]
        ax.plot(STEPS,[a*100 for a in d["acc"]],color=s["color"],ls=s["ls"],lw=s["lw"],
                label=LABELS_PARAMS[lbl])
    ax.set_xlabel("Training step"); ax.set_ylabel("Validation token accuracy (%)")
    ax.set_title("Validation Token Accuracy")
    ax.set_xlim(0,2000); ax.set_ylim(0,60)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(pct_fmt))
    ax.legend(loc="upper left",fontsize=11,**LEG)
    fig.tight_layout(); save(fig,"fig4_accuracy"); plt.close(fig)

# ── Fig 5 ─────────────────────────────────────────────────────────────────────
def fig_pareto():
    fig,ax=plt.subplots(figsize=(6.5,4.8)); pub_style(ax)
    labels=[*DATA.keys()]; params=[PARAMS[l]/1e3 for l in labels]
    acc_fin=[DATA[l]["acc"][-1]*100 for l in labels]
    ax.plot(params[1:]+[params[0]],acc_fin[1:]+[acc_fin[0]],
            color="#bbbbbb",ls="--",lw=1.4,zorder=1)
    for lbl,p,a in zip(labels,params,acc_fin):
        s=STYLE[lbl]
        ax.scatter(p,a,color=s["color"],marker=s["marker"],
                   s=120,zorder=3,edgecolors="white",linewidths=0.9,label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (thousands, log scale)")
    ax.set_ylabel("Final validation accuracy (%)")
    ax.set_title("Accuracy--Compression Pareto Frontier")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(pct_fmt))
    ax.legend(loc="lower right", fontsize=13,   # <- KEY: big enough to read chi
              frameon=True, handlelength=1.5,
              handleheight=1.3, labelspacing=0.55,
              borderpad=0.7,   markerscale=1.4)
    fig.tight_layout(); save(fig,"fig5_pareto"); plt.close(fig)

# ── Fig 6 (2x2 combined) ─────────────────────────────────────────────────────
def fig_all_panels():
    fig=plt.figure(figsize=(13,9.5))
    gs=gridspec.GridSpec(2,2,figure=fig,hspace=0.40,wspace=0.32,
                         left=0.08,right=0.97,top=0.92,bottom=0.08)
    atl,avl,aacc,apar=[fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]
    for ax in [atl,avl,aacc,apar]: pub_style(ax)

    for lbl,d in DATA.items():
        s=STYLE[lbl]
        atl.plot(STEPS,d["train"],color=s["color"],ls=s["ls"],lw=s["lw"],label=lbl)
        avl.plot(STEPS,d["val"],  color=s["color"],ls=s["ls"],lw=s["lw"],label=lbl)
        aacc.plot(STEPS,[a*100 for a in d["acc"]],
                  color=s["color"],ls=s["ls"],lw=s["lw"],label=LABELS_PARAMS[lbl])

    for ax,t,yl,lo,hi in [(atl,"(a) Training Loss","Cross-entropy loss",1.55,4.4),
                           (avl,"(b) Validation Loss","Cross-entropy loss",1.78,4.4)]:
        ax.set_xlim(0,2000); ax.set_ylim(lo,hi)
        ax.set_xlabel("Training step"); ax.set_ylabel(yl); ax.set_title(t)
        ax.legend(loc="upper right",fontsize=10,handlelength=2.2,labelspacing=0.35,frameon=True)

    aacc.set_xlim(0,2000); aacc.set_ylim(0,60)
    aacc.set_xlabel("Training step"); aacc.set_ylabel("Validation token accuracy (%)")
    aacc.set_title("(c) Validation Token Accuracy")
    aacc.yaxis.set_major_formatter(plt.FuncFormatter(pct_fmt))
    aacc.legend(loc="upper left",fontsize=9.5,handlelength=2.2,labelspacing=0.35,frameon=True)

    labels=[*DATA.keys()]; params=[PARAMS[l]/1e3 for l in labels]
    acc_fin=[DATA[l]["acc"][-1]*100 for l in labels]
    apar.plot(params[1:]+[params[0]],acc_fin[1:]+[acc_fin[0]],
              color="#cccccc",ls="--",lw=1.3,zorder=1)
    for lbl,p,a in zip(labels,params,acc_fin):
        s=STYLE[lbl]
        apar.scatter(p,a,color=s["color"],marker=s["marker"],
                     s=90,zorder=3,edgecolors="white",linewidths=0.7,label=lbl)
    apar.set_xscale("log")
    apar.set_xlabel("Parameters (K, log scale)"); apar.set_ylabel("Final val accuracy (%)")
    apar.set_title("(d) Accuracy--Compression Pareto")
    apar.yaxis.set_major_formatter(plt.FuncFormatter(pct_fmt))
    apar.legend(loc="lower right",fontsize=11,handlelength=1.5,labelspacing=0.4,frameon=True)

    fig.suptitle("Dense PicoGPT vs MPO-PicoGPT --- Shakespeare character-level prediction",
                 fontsize=14,fontweight="bold",y=0.97)
    save(fig,"fig_all_panels",dpi=300); plt.close(fig)

if __name__=="__main__":
    print("Generating figures...")
    fig_reconstruction(); fig_train_loss(); fig_val_loss()
    fig_accuracy(); fig_pareto(); fig_all_panels()
    print("\nDone.  Upload all fig*.pdf into figs/ on Overleaf.")
