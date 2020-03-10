import matplotlib.pyplot as plt

import utils
import matplotlib.patches as mpatches
from utils import DETECTION_THRESH

PLOT_DIR = "plots"


def plot_risk(risk, threshold, real_outbreaks, admin2_pcode):
    # TODO:add date and not only the #month
    detections = utils.get_detections(risk, threshold)
    fig, ax = plt.subplots()
    ax.plot(risk,lw=3)
    ax.axhline(threshold, c='gray',ls='--')
    for i, d in enumerate(detections):
        ax.axvline(x=d, lw=2, alpha=0.5, c='y', label="detections" if i == 0 else None)
        ax.fill_between(range(d,d+DETECTION_THRESH),risk[d:d+DETECTION_THRESH],alpha=0.5, color='y')
    for i, r in enumerate(real_outbreaks):
        ax.axvline(x=r, c='r', label="outbreaks" if i == 0 else None)
    ax.set_ylabel('risk')
    ax.set_xlabel('month')
    ax.set_ylim(0, 1)
    handles, _ = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='y', alpha=0.5, label='Detection window')
    handles.append(patch) 
    plt.legend(handles=handles, loc='upper center')
    # ax.legend()
    fig.savefig(f'{PLOT_DIR}/risk_{admin2_pcode}.png')
    plt.close(fig)


def plot_f1(df, admin2_pcode):
    fig, ax = plt.subplots()
    df[['thresh', 'precision', 'recall', 'f1']].plot(x='thresh', ax=ax)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(f'{PLOT_DIR}/f1_{admin2_pcode}.png')
    plt.close(fig)
