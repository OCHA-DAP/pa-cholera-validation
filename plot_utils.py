import matplotlib.pyplot as plt
import geopandas as gpd
import os

import utils
import matplotlib.patches as mpatches
from utils import DETECTION_THRESH, RISK_THRESH

PLOT_DIR = "plots"


def plot_risk(df_risk, real_outbreaks, admin2_pcode,ax):
    # TODO:add date and not only the #month
    risk=df_risk['risk']
    detections = utils.get_detections(risk, RISK_THRESH)
    # fig, ax = plt.subplots()
    ax.plot(risk,lw=3)
    ax.axhline(RISK_THRESH, c='gray',ls='--')
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
    ax.legend(handles=handles, loc='upper center')
    # ax.legend()
    # fig.savefig(f'{PLOT_DIR}/risk_{admin2_pcode}.png')
    # plt.close(fig)

# def plot_f1(df, admin2_pcode,ax):
def plot_performance(df, ax):
    # fig, ax = plt.subplots()
    df[['thresh', 'precision', 'recall']].plot(x='thresh', ax=ax)
    ax.set_ylim(-0.05, 1.05)
    # fig.savefig(f'{PLOT_DIR}/f1_{admin2_pcode}.png')
    # plt.close(fig)

def plot_map(admin2_pcode,fig,ax):
    zwe_boundaries=gpd.read_file('{}/zwe_admbnda_adm2_zimstat_ocha_20180911/zwe_admbnda_adm2_zimstat_ocha_20180911.shp'.format(os.getcwd()))
    zwe_boundaries.boundary.plot(ax=ax)
    adm2_boundary=zwe_boundaries[zwe_boundaries['ADM2_PCODE']==admin2_pcode]
    adm2_boundary.plot(ax=ax,color='red')
    fig.suptitle('{} - {}'.format(adm2_boundary.iloc[0]['ADM1_EN'],adm2_boundary.iloc[0]['ADM2_EN']))

def plot_confusion_matrix(df, ax):
    ax.axis('off')
    df=df.loc[:,['thresh','TP','FP','FN']]
    df['thresh']=df['thresh'].round(decimals=1)
    the_table=ax.table(cellText=df.values, colLabels=df.columns, loc='center',rowLoc='center',colLoc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1.2,1.7)


def plot_adm2(df_risk,df_performance, real_outbreaks, admin2_pcode,admin2_name):
    fig, axs = plt.subplots(2,2,figsize=(15,15))
    # top left: map
    plot_map(admin2_pcode,fig,ax=axs[0,0])
    # top right: risk
    plot_risk(df_risk, real_outbreaks, admin2_pcode,ax=axs[0,1])
    # bottom right: precision/recall
    plot_confusion_matrix(df_performance, ax=axs[1,0])
    # bottom right: precision/recall
    plot_performance(df_performance, ax=axs[1,1])
    fig.savefig(f'{PLOT_DIR}/{admin2_pcode}_{admin2_name}.png')
    plt.close(fig)
    # plt.show()
