import matplotlib.pyplot as plt

import utils


def plot_risk(risk, threshold, real_outbreaks):
    detections = utils.get_detections(risk, threshold)
    fig, ax = plt.subplots()
    ax.plot(risk)
    ax.axhline(threshold, c='k')
    for d in detections:
        ax.axvline(x=d, c='r')
    for r in real_outbreaks:
        ax.axvline(x=r, c='y')
    ax.set_ylabel('risk')
    ax.set_xlabel('month')
    fig.show()


def plot_f1(df):
    df[['thresh', 'precision', 'recall', 'f1']].plot(x='thresh')
    plt.show()
