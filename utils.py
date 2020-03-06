import array

import numpy as np
import pandas as pd

# How much earlier a detection can be with respect to a real outbreak
DETECTION_THRESH = 4  # months


def generate_fake_risk(n_steps: int, rs: np.random.RandomState, p0: float = 0.5) -> array:
    """
    Generate fake risk data that goes between 0 and 1 with a random walk
    :param n_steps: the number of steps to take
    :param rs: a Numpy random state
    :param p0: the probability of step size 0. The probability of step size -1 / 1 will then be (1-p0)/2
    :return: an array
    """
    # Create the steps
    steps = rs.choice(a=[-1, 0, 1], size=n_steps, p=[(1 - p0) / 2, p0, (1 - p0) / 2])
    # Compute the path values based on the steps
    path = np.concatenate([[0], steps]).cumsum(0)
    # Normalize between 0 and 1
    path = (path - min(path)) / (max(path) - min(path))
    return path


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


def loop_over_thresholds(risk: array, real_outbreaks: array, threshold_step: float = 0.05) -> pd.DataFrame:
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
    return df
