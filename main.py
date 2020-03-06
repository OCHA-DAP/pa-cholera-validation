import numpy as np

import utils, plot_utils

SEED = 12345

START_DATE = '2008-01'
END_DATE = '2020-01'


def main():
    rs = np.random.RandomState(SEED)
    # Get the outbreaks, and loop through the
    df_outbreaks = utils.get_outbreaks()
    for admin2_pcode, df_outbreak in df_outbreaks.groupby('admin2Pcode'):
        print(f'Analyzing admin region {admin2_pcode}')
        # Make the fake data
        df_risk = utils.generate_fake_risk(rs, START_DATE, END_DATE)
        # Get outbreak date indices
        # TODO: just pass the risk dataframe
        df_risk['outbreak'] = df_risk['date'].isin(df_outbreak['Outbreak month'])
        real_outbreaks = df_risk[df_risk['outbreak']].index.values
        # Make plot for threshold 0.5
        plot_utils.plot_risk(df_risk['risk'], 0.5, real_outbreaks, admin2_pcode)
        # Get detections per threshold
        df = utils.loop_over_thresholds(df_risk['risk'], real_outbreaks)
        df = utils.calculate_f1(df)
        plot_utils.plot_f1(df, admin2_pcode)


if __name__ == '__main__':
    main()
