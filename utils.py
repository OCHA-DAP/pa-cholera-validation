import array
import datetime
import collections

import numpy as np
import pandas as pd
import geopandas as gpd

# How much earlier a detection can be with respect to a real outbreak
DETECTION_THRESH = 4  # months
RISK_THRESH = 0.5     # value for the example and the overall evaluation
THRESHOLD_STEP = 0.05
# Window for non-zero risk after shock
SHOCK_WINDOW = 4

FILENAME_OUTBREAKS = 'List of Admin Units.xlsx'
SHEET_NAME_SHORTLIST = 'Proposed Shortlist'

FILENAME_SHOCKS = 'cholera_shocks.xlsx'
SHEET_NAME_EMDAT = 'zimbabwe_emdat'
SHEET_NAME_GDACS = 'zimbabwe_gdacs'
SHEET_NAME_ADM2 = 'ADM2_Zimbabwe'

FILENAME_BOUNDARIES = 'zwe_admbnda_adm2_zimstat_ocha_20180911'


def get_df_performance_all(threshold_step: float = THRESHOLD_STEP) -> pd.DataFrame:
    df_risk_all = pd.DataFrame({
        'thresh': np.arange(0, 1 + threshold_step, threshold_step),
        'TP': 0, 'FP': 0, 'FN': 0
    })
    return df_risk_all


def generate_fake_risk(rs: np.random.RandomState, start_date: str, end_date: str, p0: float = 0.5) -> pd.DataFrame:
    """
    Generate fake risk data that goes between 0 and 1 with a random walk
    :param rs: a Numpy random state
    :param start_date: start date string formatted 'YY-mm'
    :param end_date: end date string formatted 'YY-mm'
    :param p0: the probability of step size 0. The probability of step size -1 / 1 will then be (1-p0)/2
    :return: dataframe with month and risk columns
    """
    # Make dataframe with dates
    df_risk = pd.DataFrame({'date': pd.date_range(start=start_date, end=end_date, freq='M').to_period('M')})
    # Create the steps
    steps = rs.choice(a=[-1, 0, 1], size=len(df_risk)-1, p=[(1 - p0) / 2, p0, (1 - p0) / 2])
    # Compute the path values based on the steps
    path = np.concatenate([[0], steps]).cumsum(0)
    # Normalize between 0 and 1
    df_risk['risk'] = (path - min(path)) / (max(path) - min(path))
    return df_risk


def get_outbreaks(sheet_name: str = 'Outbreaks_Zimbabwe') -> pd.DataFrame:
    """
    Get a dataframe with the outbreaks for a country. Create a column containing the outbreak month.
    :param sheet_name: The name of the sheet in the excel file
    :return: dataframe with outbreaks
    """
    df_outbreaks = pd.read_excel(f'input/{FILENAME_OUTBREAKS}', sheet_name=sheet_name)
    df_outbreaks['Outbreak month'] = df_outbreaks['Outbreak date'].dt.to_period('M')
    return df_outbreaks


def get_shocks_data() -> pd.DataFrame:
    df_shocks = pd.concat([get_shocks_emdat(), get_shocks_gdacs()])
    df_shocks['month_start'] = df_shocks['date_start'].dt.to_period('M')
    df_shocks['month_end'] = df_shocks['date_end'].dt.to_period('M')
    return df_shocks


def get_shocks_emdat() -> pd.DataFrame:
    # Read in the shocks and admin 2 to admin 1 region mapping
    df_shocks_input = pd.read_excel(f'input/{FILENAME_SHOCKS}', sheet_name=SHEET_NAME_EMDAT)
    df_adm2 = pd.read_excel(f'input/{FILENAME_OUTBREAKS}', sheet_name=SHEET_NAME_ADM2)
    # Clean shocks input
    clist = ['Start Year', 'Start Month', 'Start Day', 'End Year', 'End Month', 'End Day']
    df_shocks_input[clist] = df_shocks_input[clist].fillna(-1).astype(int).replace(-1, None)
    df_shocks_input[['admin1', 'admin2']] = (df_shocks_input[['admin1', 'admin2']]
                                             .fillna('')
                                             .applymap(lambda x: x.split(',')))
    # Turn df_adm2 into dictionary
    adm2_dict = df_adm2.groupby('admin1Name_en')['admin2Name_en'].apply(list).to_dict()
    adm2_pcode_dict = df_adm2.set_index('admin2Name_en')['admin2Pcode'].to_dict()
    # Make an empty shocks table
    df_shocks = pd.DataFrame(columns=['date_start', 'date_end', 'district', 'pcode', 'event', 'details' ])
    for _, row in df_shocks_input.iterrows():
        new_row = {
            'event': row['Disaster Type'],
            'details': row['Disaster Subtype'],
            'date_start': datetime.datetime(
                row['Start Year'],
                row['Start Month'],
                row['Start Day'] if row['Start Day'] != -1 else 1),
            'date_end': (datetime.datetime(
                row['End Year'],
                row['End Month'],
                row['End Day'] if row['End Day'] != -1 else 1)
                if row['End Month'] != -1 else None)
        }
        # Get the districts
        districts = [admin2.strip() for admin2 in row['admin2'] if admin2 != '']
        districts += [adm2_dict[admin1.strip()] for admin1 in row['admin1'] if admin1 != '']
        districts = flatten(districts)
        # Loop through each district and add to shock
        for district in districts:
            df_shocks = df_shocks.append(
                dict(new_row, **{'district': district, 'pcode': adm2_pcode_dict[district]}), ignore_index=True)
    # Create end date if doesn't exists
    df_shocks['date_end'] = df_shocks.apply(lambda x: x['date_start'] if x['date_end'] is None else x['date_end'],
                                            axis=1)
    return df_shocks


def get_shocks_gdacs() -> pd.DataFrame:
    df_shocks = pd.read_excel(f'input/{FILENAME_SHOCKS}', sheet_name=SHEET_NAME_GDACS)
    df_shocks = gpd.GeoDataFrame(df_shocks, geometry=gpd.points_from_xy(df_shocks._x, df_shocks._y))
    df_boundaries = get_boundaries_data()
    df_shocks['pcode'] = df_shocks['geometry'].apply(lambda x:
        [y['ADM2_PCODE'] for _, y in df_boundaries.iterrows() if y['geometry'].contains(x)])
    df_shocks['pcode'] = df_shocks['pcode'].apply(lambda x: x[0] if len(x) else None)
    df_shocks = df_shocks.rename({'gdacs_fromdate': 'date_start',
                                  'gdacs_todate': 'date_end',
                                  'gdacs_eventtype': 'event',
                                  'Title': 'details'})
    # For shocks with no Pcode (i.e. main location was outside of country),
    # add to full country
    df_to_add = pd.DataFrame()
    df_adm2 = pd.read_excel(f'input/{FILENAME_OUTBREAKS}', sheet_name=SHEET_NAME_ADM2)
    print('Adding GDACS shocks to all regions...')
    for _, row in df_shocks.iterrows():
        if row['pcode'] is None:
            for pcode in df_adm2['admin2Pcode']:
                row['pcode'] = pcode
                df_to_add = df_to_add.append(row)
    print('...done')
    return df_shocks


def get_boundaries_data() -> pd.DataFrame:
    return gpd.read_file(f'input/{FILENAME_BOUNDARIES}/{FILENAME_BOUNDARIES}.shp')


def flatten(x: list) -> list:
    result = []
    for el in x:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def get_adm2_shortlist() -> array:
    """
    Get shortlist of admin 2 regions for cholera outbreaks
    :param sheet_name: The name of the sheet in the excel file
    :return: array with [[admin2 pcode, admin2 english name]]
    """
    df_adm2 = pd.read_excel(f'input/{FILENAME_OUTBREAKS}', sheet_name=SHEET_NAME_SHORTLIST)
    adm2_shortlist = df_adm2[['admin2Pcode', 'admin2Name_en']].values
    return adm2_shortlist


def get_risk_df(df_risk_all: pd.DataFrame, admin2_name: str) -> pd.DataFrame:
    df_risk = df_risk_all[df_risk_all['adm2'] == admin2_name].drop(columns=['adm2']).T.reset_index()
    df_risk.columns = ['date', 'risk']
    # TODO: NOT URGENT keep the date to look for the 4 months window. The risk date is the first day of the month
    df_risk['date'] = df_risk['date'].dt.to_period('M')
    return df_risk


def get_shocks(df_shock: pd.DataFrame, df_risk: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df_risk['shocks'] = df_risk['date'].isin(df_shock['month_start'])
    shocks = df_risk[df_risk['shocks']].index.values
    # Use shocks to define risk window
    shocks_with_window = flatten([list(np.arange(SHOCK_WINDOW + 1) + shock) for shock in shocks])
    df_risk['shocks'] = df_risk.index.isin(shocks_with_window)
    df_risk['risk'] = np.where(df_risk['shocks'], df_risk['risk'], 0)
    return shocks, df_risk


def get_detections(risk: array, threshold: float) -> array:
    """
    Given an array of risk vales as a function of time, mark detections as the first time
    a sequence of values goes above the risk threshold
    :param risk: array of floats between 0 and 1 denoting risk
    :param threshold:
    :return: array of indices where the risk first goes above the threshold
    """
    # Groups also contains the last index of each sequence, in case we ever need it
    groups = np.where(np.diff(np.hstack(([False], risk > threshold, [False]))))[0].reshape(-1, 2)
    return groups[:, 0]


def validate_detections(detections: array, real_outbreaks: array) -> pd.DataFrame:
    """
    Given a list of detections, compare them against real outbreaks to see if they are TP or FP.
    Also check missed real outbreaks for FN.
    :param detections: array of detection indices
    :param real_outbreaks: array of real outbreak indices
    :return: DataFrame with number of TP, FP and FN
    """
    # Make a DF of detections
    df = pd.concat([
        pd.DataFrame({"event_date": detections, "true_event": False}),
        pd.DataFrame({"event_date": real_outbreaks, "true_event": True})
    ]).sort_values(by="event_date").assign(TP=False, FP=False, FN=False)

    # For all non-true events, if the next event is a true event
    # and within DETECTION_THRESH, this is a TP
    df.loc[~df['true_event'] & df['true_event'].shift(-1) &
           (df['event_date'].shift(-1)-df['event_date'] <= DETECTION_THRESH), "TP"] = True

    # All remaining non-true events are FP
    df.loc[~df['true_event'] & ~df['TP'], 'FP'] = True

    # For true events, if the previous event is not a TP,
    # then that true event is an FN
    df.loc[df['true_event'] & ~df['TP'].shift(1, fill_value=False), 'FN'] = True

    return df[['TP', 'FP', 'FN']].apply(sum)


def loop_over_thresholds(risk: array, real_outbreaks: array, threshold_step: float = THRESHOLD_STEP) -> pd.DataFrame:
    """
    For a given threshold step, loop over thresholds from 0 to 1 and calculate TP, FP and FN
    :param risk: array of risk over time
    :param real_outbreaks: array of real outbreak indices
    :param threshold_step: the threshold step size
    :return: DataFrame with TP, FP, and FN columns as a function of threshold
    """
    df = pd.DataFrame({'thresh': np.arange(0, 1 + threshold_step, threshold_step)})
    df[['TP', 'FP', 'FN']] = df.apply(
        lambda x: validate_detections(get_detections(risk, x['thresh']), real_outbreaks),
        axis=1,
    )
    # df.plot(kind='bar', x='thresh', stacked=True)
    return df


def calculate_f1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the precision, recall and f1
    :param df: DataFrame with columns TP, FP and FN
    :return: input dataframe with precision, recall and f1 columns
    """
    # Avoid 0s in the denominator or else you get division by 0 error
    df[['FP', 'FN']] = df[['FP', 'FN']].replace(0, np.nan)
    # Calc precision and recall
    df['precision'] = df.apply(lambda x: x['TP'] / (x['TP'] + x['FP']), axis=1)
    df['recall'] = df.apply(lambda x: x['TP'] / (x['TP'] + x['FN']), axis=1)

    # Only calc F1 if precision and recall are > 0 to avoid division by 0 error
    idx = (df['precision'] > 0) & (df['recall'] > 0)
    df.loc[idx, 'f1'] = df[idx].apply(lambda x: 2 / (1 / x['precision'] + 1 / x['recall']), axis=1)

    # Put nans back to 0
    return df.fillna(0)
