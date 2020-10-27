"""Python code implementing statistical methods used in analysis of
relative incidence of COVID in healthcare workers versus
non-healthcare workers.
"""
import numpy as np, matplotlib.pyplot as plt, pandas as pd


date_fmt = "%B %-d, %Y"


def load_data():
    df = pd.read_csv('data/fb_data_usa.csv', index_col=0, low_memory=False)
    df['date'] = df.date.map(pd.Timestamp)

    min_date = df.date.min()
    most_recent_data_date = df.date.max()
    data_start_date = max(min_date, most_recent_data_date - pd.Timedelta(days=6 * 7))

    return locals()


def subset_data(data, rows, exposure, outcome):
    df = pd.DataFrame(index=data['df'][rows].index)
    df['exposure'] = data['df'].loc[rows, exposure].astype(float)
    df['outcome'] = data['df'].loc[rows, outcome].astype(float)
    df['weight'] = data['df'].loc[rows, 'weight'].astype(float)

    df['weighted_outcome'] = df.weight * df.outcome

    return df


def calc_rr(df):
    g =  df.groupby('exposure')
    p = g.weighted_outcome.sum() / g.weight.sum()
    if len(p) != 2:
        return np.nan
    else:
        p_nonhcw, p_hcw = p
    return p_hcw / p_nonhcw


def sample_rr_draws(df):
    """Use bootstrap resampling to obtain uncertainty in relative incidence ratio
    that includes survey weights
    """
    rr = {}
    
    rr['point_est'] = calc_rr(df)
    
    for i in range(1_000):
        resampled_rows = np.random.choice(df.index, size=len(df.index))

        rr[f'draw_{i}'] = calc_rr(df.loc[resampled_rows])

    rr = pd.Series(rr)
    
    
    return rr


def my_summarize(s):
    rr_mean = s['point_est']
    result = s.filter(like='draw_').describe(percentiles=[.025, .975]).filter(['2.5%', '97.5%'])
    return rr_mean, result['2.5%'], result['97.5%']


def my_calc_and_summarize(df):
    results = [sum(df.exposure == 0), sum(df.exposure == 1)]
    results += list(my_summarize(sample_rr_draws(df)))
    return results
