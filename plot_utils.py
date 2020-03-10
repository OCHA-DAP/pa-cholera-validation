import matplotlib.pyplot as plt

import utils

PLOT_DIR = "plots"


def plot_risk(risk, threshold, real_outbreaks, admin2_pcode):
    # TODO:add date and not only the #month
    detections = utils.get_detections(risk, threshold)
    fig, ax = plt.subplots()
    ax.plot(risk)
    ax.axhline(threshold, c='k')
    for i, d in enumerate(detections):
        ax.axvline(x=d, lw=2, alpha=0.5, c='y', label="detections" if i == 0 else None)
    for i, r in enumerate(real_outbreaks):
        ax.axvline(x=r, c='r', label="outbreaks" if i == 0 else None)
    ax.set_ylabel('risk')
    ax.set_xlabel('month')
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(f'{PLOT_DIR}/risk_{admin2_pcode}.png')
    plt.close(fig)


def plot_f1(df, admin2_pcode):
    fig, ax = plt.subplots()
    df[['thresh', 'precision', 'recall', 'f1']].plot(x='thresh', ax=ax)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(f'{PLOT_DIR}/f1_{admin2_pcode}.png')
    plt.close(fig)
