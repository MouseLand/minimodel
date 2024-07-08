"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""
import string
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

cmap_emb = ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0.05, 0.95), 100))


kp_colors = np.array([[0.55,0.55,0.55],
                      [0.,0.,1],
                      [0.8,0,0],
                      [1.,0.4,0.2],
                      [0,0.6,0.4],
                      [0.2,1,0.5],
                      ])

default_font = 12
rcParams["font.family"] = "Arial"
rcParams["savefig.dpi"] = 300
rcParams["axes.spines.top"] = False
rcParams["axes.spines.right"] = False
rcParams["axes.titlelocation"] = "left"
rcParams["axes.titleweight"] = "normal"
rcParams["font.size"] = default_font

ltr = string.ascii_lowercase
fs_title = 16
weight_title = "normal"


def add_apml(ax, xpos, ypos, dx=300, dy=300, tp=30):
    x0, x1, y0, y1 = (
        xpos.min() - dx / 2,
        xpos.min() + dx / 2,
        ypos.max(),
        ypos.max() + dy,
    )
    ax.plot(np.ones(2) * (y0 + dy / 2), [x0, x1], color="k")
    ax.plot([y0, y1], np.ones(2) * (x0 + dx / 2), color="k")
    ax.text(y0 + dy / 2, x0 - tp, "P", ha="center", va="top", fontsize="small")
    ax.text(y0 + dy / 2, x0 + dx + tp, "A", ha="center", va="bottom", fontsize="small")
    ax.text(y0 - tp, x0 + dx / 2, "M", ha="right", va="center", fontsize="small")
    ax.text(y0 + dy + tp, x0 + dx / 2, "L", ha="left", va="center", fontsize="small")
    print(x0, y0)

def plot_label(ltr, il, ax, trans, fs_title=20):
    ax.text(
        0.0,
        1.0,
        ltr[il],
        transform=ax.transAxes + trans,
        va="bottom",
        fontsize=fs_title,
        fontweight="bold",
    )
    il += 1
    return il