import numpy as np
import pandas as pd

import utils, plot_utils

SEED = 12345

START_DATE = '2008-01'
END_DATE = '2020-01'

FILENAME_ZIMBABWE = 'zimbabwe_final.xlsx'


def main():
    # rs = np.random.RandomState(SEED)
    # Get the outbreaks, and loop through the
    df_outbreaks = utils.get_outbreaks()
    df_risk_all = pd.read_excel(FILENAME_ZIMBABWE)
    for admin2_pcode,admin2_name in utils.get_adm2shortlist():
        df_outbreak=df_outbreaks[df_outbreaks['admin2Pcode']==admin2_pcode]
        # admin2_name = df_outbreak['admin2Name_en'].iloc[0]
        # Go to next pcode if not in Zimbabwe
        if admin2_name not in list(df_risk_all['adm2']):
            continue
        print(f'Analyzing admin region {admin2_pcode}')
        # Make the fake data
        # df_risk = utils.generate_fake_risk(rs, START_DATE, END_DATE)
  
        # Get risk from Zimbabwe data
        df_risk = df_risk_all[df_risk_all['adm2'] == admin2_name].drop(columns=['adm2']).T.reset_index()
        df_risk.columns = ['date', 'risk']
        # TODO: NOT URGENT keep the date to look for the 4 months window. The risk date is the first day of the month
        df_risk['date'] = df_risk['date'].dt.to_period('M')
        # Get outbreak date indices
        # TODO: just pass the risk dataframe
        df_risk['outbreak'] = df_risk['date'].isin(df_outbreak['Outbreak month'])
        real_outbreaks = df_risk[df_risk['outbreak']].index.values
        # Get detections per threshold
        df_performance = utils.loop_over_thresholds(df_risk['risk'], real_outbreaks)
        df_performance = utils.calculate_f1(df_performance)
        plot_utils.plot_adm2(df_risk,df_performance, real_outbreaks, admin2_pcode,admin2_name)

        # TODO: evaluate the best threshold value and calculate the overall value of precision and recall


if __name__ == '__main__':
    main()
