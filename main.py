import numpy as np

import utils, plot_utils

SEED = 1234
N_STEPS = 15 * 12
REAL_OUTBREAKS = [35, 100,  127]


def main():
    rs = np.random.RandomState(SEED)
    risk = utils.generate_fake_risk(N_STEPS, rs)
    #plot_utils.plot_risk(risk, 0.5, REAL_OUTBREAKS)
    df = utils.loop_over_thresholds(risk, REAL_OUTBREAKS)
    df = utils.calculate_f1(df)
    plot_utils.plot_f1(df)


if __name__ == '__main__':
    main()
