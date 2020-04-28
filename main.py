import pandas as pd
import matplotlib.pyplot as plt

import utils, plot_utils

SEED = 12345

START_DATE = '2008-01'
END_DATE = '2020-01'

FILENAME_ZIMBABWE = 'zimbabwe_final.xlsx'


def main():
    # rs = np.random.RandomState(SEED)
    # Get the outbreaks, and loop through the
    df_outbreaks = utils.get_outbreaks()
    df_shocks = utils.get_shocks()
    df_risk_all = pd.read_excel(f'input/{FILENAME_ZIMBABWE}')
    df_performance_all = utils.get_df_performance_all()
    for admin2_pcode, admin2_name in utils.get_adm2_shortlist():
        df_outbreak = df_outbreaks[df_outbreaks['admin2Pcode'] == admin2_pcode]
        df_shocks = df_shocks[df_shocks['district'] == admin2_name]
        # Go to next pcode if not in Zimbabwe
        if admin2_name not in list(df_risk_all['adm2']):
            continue
        print(f'Analyzing admin region {admin2_name}')
        # Make the fake data
        # df_risk = utils.generate_fake_risk(rs, START_DATE, END_DATE)
        # Get risk from Zimbabwe data
        df_risk = utils.get_risk_df(df_risk_all, admin2_name)
        # Get outbreak date indices
        df_risk['outbreak'] = df_risk['date'].isin(df_outbreak['Outbreak month'])
        real_outbreaks = df_risk[df_risk['outbreak']].index.values
        # Get detections per threshold
        df_performance = utils.loop_over_thresholds(df_risk['risk'], real_outbreaks)
        df_performance = utils.calculate_f1(df_performance)
        # Add it to the full frame
        df_performance_all = (pd.concat([df_performance[['thresh', 'TP', 'FP', 'FN']], df_performance_all])
                              .groupby(['thresh'])
                              .sum()
                              .reset_index())
        # Make plots
        plot_utils.plot_adm2(df_risk, df_performance, real_outbreaks, admin2_pcode, admin2_name)
        # TODO: evaluate the best threshold value and calculate the overall value of precision and recall
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
