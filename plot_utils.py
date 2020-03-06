import matplotlib.pyplot as plt

import utils

PLOT_DIR = "plots"


def plot_risk(risk, threshold, real_outbreaks, admin2_pcode):
    detections = utils.get_detections(risk, threshold)
    fig, ax = plt.subplots()
    ax.plot(risk)
    ax.axhline(threshold, c='k')
    for d in detections:
        ax.axvline(x=d, c='y')
    for r in real_outbreaks:
        ax.axvline(x=r, c='r')
    ax.set_ylabel('risk')
    ax.set_xlabel('month')
    ax.set_ylim(0, 1)
    fig.savefig(f'{PLOT_DIR}/risk_{admin2_pcode}.png')
    plt.close(fig)


def plot_f1(df, admin2_pcode):
    fig, ax = plt.subplots()
    df[['thresh', 'precision', 'recall', 'f1']].plot(x='thresh', ax=ax)
    ax.set_ylim(-0.05, 1.05)
    fig.savefig(f'{PLOT_DIR}/f1_{admin2_pcode}.png')
    plt.close(fig)
