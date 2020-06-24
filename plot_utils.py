import matplotlib.pyplot as plt
import geopandas as gpd

import utils
import matplotlib.patches as mpatches
from utils import DETECTION_THRESH, RISK_THRESH

PLOT_DIR = "plots"


def plot_adm2(df_risk, df_performance, real_outbreaks, shocks, admin2_pcode, admin2_name):
    fig, axs = plt.subplots(2, 2,figsize=(25, 15))
    # top left: map
    plot_map(admin2_pcode, fig, ax=axs[0, 0])
    # top right: risk
    plot_risk(df_risk, real_outbreaks, shocks, axs[0, 1])
    # bottom right: precision/recall
    plot_confusion_matrix(df_performance, ax=axs[1, 0], scale_table_for_mosaic=True)
    # bottom right: precision/recall
    plot_performance(df_performance, ax=axs[1, 1])
    fig.savefig(f'{PLOT_DIR}/{admin2_pcode}_{admin2_name}.png')
    plt.close(fig)


def plot_risk(df_risk, real_outbreaks, shocks, ax):
    # TODO:add date and not only the #month
    risk = df_risk['risk']
    detections = utils.get_detections(risk, RISK_THRESH)
    ax.plot(risk, lw=3)
    ax.axhline(RISK_THRESH, c='gray', ls='--')
    for i, d in enumerate(detections):
        ax.axvline(x=d, lw=2, alpha=0.5, c='y', label="detections" if i == 0 else None)
        try:
            ax.fill_between(range(d, d+DETECTION_THRESH), risk[d:d+DETECTION_THRESH], alpha=0.5, color='y')
        except:
            ax.fill_between(range(d, len(risk)), risk[d:len(risk)], alpha=0.5, color='y')
    for i, r in enumerate(real_outbreaks):
        ax.axvline(x=r, c='r', label="outbreaks" if i == 0 else None)
    for i, s in enumerate(shocks):
        ax.axvline(x=s[0], c='g', label="shocks" if i == 0 else None)
    ax.set_ylabel('risk')
    ax.set_xlabel('month')
    ax.set_ylim(0, 1)
    handles, _ = ax.get_legend_handles_labels()
    patch = mpatches.Patch(color='y', alpha=0.5, label='Detection window')
    handles.append(patch) 
    ax.legend(handles=handles, loc='upper center')


def plot_shocks_and_outbreaks(ax, real_outbreaks, shocks, admin2_name, df_risk,
                              show_x_axis=False):
    for i, r in enumerate(real_outbreaks):
        ax.axvline(x=r, c='r', lw=2)
    for i, s in enumerate(shocks):
        ax.axvline(x=s[0], c='g', lw=2)
        ax.fill_between(x=(s[0], s[1]), y1=0, y2=1, facecolor='g', alpha=0.2)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, len(df_risk['risk']))
    ax.minorticks_on()
    ax.set_yticks([])
    if show_x_axis:
        ax.set_xlabel('Month number')
    else:
        ax.set_xticklabels([])
    ax.text(1, 0.5, admin2_name)


def plot_performance(df, ax):
    df[['thresh', 'precision', 'recall', 'f1']].plot(x='thresh', ax=ax)
    ax.set_ylim(0.0, 1.05)


def plot_map(admin2_pcode, fig, ax):
    zwe_boundaries = utils.get_boundaries_data()
    zwe_boundaries.boundary.plot(ax=ax)
    adm2_boundary=zwe_boundaries[zwe_boundaries['ADM2_PCODE']==admin2_pcode]
    adm2_boundary.plot(ax=ax,color='red')
    fig.suptitle('{} - {}'.format(adm2_boundary.iloc[0]['ADM1_EN'],adm2_boundary.iloc[0]['ADM2_EN']),fontsize=30)


def plot_confusion_matrix(df, ax, scale_table_for_mosaic=False):
    ax.axis('off')
    df = df.loc[:, ['thresh', 'TP', 'FP', 'FN']]
    df['thresh'] = df['thresh'].apply(lambda x: f'{x : 0.2f}')
    df = df.astype({'TP': int, 'FP': int, 'FN': int})
    the_table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', rowLoc='center', colLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    if scale_table_for_mosaic:
        the_table.scale(1.2, 1.7)
