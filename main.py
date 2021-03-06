import pandas as pd
import matplotlib.pyplot as plt

import utils, plot_utils

SEED = 12345

START_DATE = '2008-01'
END_DATE = '2020-01'

FILENAME_ZIMBABWE = '01_zim_original-1.xlsx'


def main():
    # rs = np.random.RandomState(SEED)
    # Get the outbreaks, and loop through the
    df_outbreaks = utils.get_outbreaks()
    df_shocks = utils.get_shocks_data()
    df_risk_all = pd.read_excel(f'input/risk/{FILENAME_ZIMBABWE}')
    df_performance_all = utils.get_df_performance_all()
    # Get adm2 present in risks file
    adm2_shortlist = utils.get_adm2_shortlist(df_risk_all)
    # Plot for outbreaks and shocks
    fig, axs = plt.subplots(len(adm2_shortlist), 1, figsize=(10, 10))
    for iadm2, (admin2_pcode, admin2_name) in enumerate(adm2_shortlist):
        print(f'Analyzing admin region {admin2_name}')
        df_outbreak = df_outbreaks[df_outbreaks['admin2Pcode'] == admin2_pcode]
        df_shock = df_shocks[df_shocks['pcode'] == admin2_pcode]
        # Make the fake data
        # df_risk = utils.generate_fake_risk(rs, START_DATE, END_DATE)
        # Get risk from Zimbabwe data
        df_risk = utils.get_risk_df(df_risk_all, admin2_name)
        # Get outbreak date indices
        df_risk['outbreak'] = df_risk['date'].isin(df_outbreak['Outbreak month'])
        real_outbreaks = df_risk[df_risk['outbreak']].index.values
        # Get shocks
        shocks, df_risk = utils.get_shocks(df_shock, df_risk)
        # Get detections per threshold
        df_performance = utils.loop_over_thresholds(df_risk['risk'], real_outbreaks)
        df_performance = utils.calculate_f1(df_performance)
        # Add it to the full frame
        df_performance_all = (pd.concat([df_performance[['thresh', 'TP', 'FP', 'FN']], df_performance_all])
                              .groupby(['thresh'])
                              .sum()
                              .reset_index())
        # Make plots
        plot_utils.plot_adm2(df_risk, df_performance, real_outbreaks, shocks, admin2_pcode, admin2_name)
        # Plot shocks / outbreaks
        plot_utils.plot_shocks_and_outbreaks(axs[iadm2], real_outbreaks, shocks, admin2_name, df_risk,
                                             show_x_axis=(iadm2 == len(adm2_shortlist) - 1))
        # TODO: evaluate the best threshold value and calculate the overall value of precision and recall
    # Save the shocks / outbreaks figure
    fig.savefig('plots/outbreaks_shocks.png')
    plt.close(fig)
    # Caclulate overall performance
    df_performance_all = utils.calculate_f1(df_performance_all)
    # Confusion matrix
    fig, ax = plt.subplots()
    plot_utils.plot_confusion_matrix(df_performance_all, ax)
    fig.savefig('plots/full_confusion_matrix.png')
    plt.close()
    # Performance
    fig, ax = plt.subplots()
    plot_utils.plot_performance(df_performance_all, ax)
    fig.savefig('plots/full_performance.png')
    plt.close()


if __name__ == '__main__':
    main()
