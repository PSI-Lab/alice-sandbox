import pandas as pd
from scipy.stats import spearmanr


def compute_corr(df1, df2, tx_ids, min_coverages):
    data = []
    for tx_id in tx_ids:
        row = {'tx_id': tx_id}

        for c in min_coverages:
            _df = pd.merge(
                df1[(df1.tx_id == tx_id) & (df1.coverage >= c)][
                    ['transcript_position', 'reactivity']].rename(columns={'reactivity': 'reactivity_1'}),
                df2[(df2.tx_id == tx_id) & (df2.coverage >= c)][
                    ['transcript_position', 'reactivity']].rename(columns={'reactivity': 'reactivity_2'}),
                on='transcript_position', how='outer')

            df = _df.dropna()

            corr, pval = spearmanr(df['reactivity_1'], df['reactivity_2'])

            _row = {
                'n1_{}'.format(c): len(_df['reactivity_1'].dropna()),
                'n2_{}'.format(c): len(_df['reactivity_2'].dropna()),
                'n_{}'.format(c): len(df),
                'corr_{}'.format(c): corr,
                'pval_{}'.format(c): pval,
            }

            row.update(_row)
        data.append(row)

    # histogram of correlation at different cutoff
    df = pd.DataFrame(data)

    return df


def get_max_median_min(df, df1, df2, c, n):
    _corr = df[(df['pval_{}'.format(c)] < 0.01) & (df['n_{}'.format(c)] >= n)]['corr_{}'.format(c)]
    tx_id_max = df[df['corr_{}'.format(c)] == _corr.max()].iloc[0]['tx_id']
    # to avoid the problem where median is not an actual data point, make sure the total number is odd
    if len(_corr) % 2 != 0:
        _corr_median = _corr.median()
        tx_id_med = df[df['corr_{}'.format(c)] == _corr_median].iloc[0]['tx_id']
    else:
        _corr_median = _corr[:-1].median()  # quite arbitrary
        tx_id_med = df[df['corr_{}'.format(c)] == _corr_median].iloc[0]['tx_id']
    tx_id_min = df[df['corr_{}'.format(c)] == _corr.min()].iloc[0]['tx_id']
    print("Max correlation: {} from {}".format(_corr.max(), tx_id_max))
    print("Median correlation: {} from {}".format(_corr_median, tx_id_med))
    print("Min correlation: {} from {}".format(_corr.min(), tx_id_min))

    result = {}
    for name, tx_id in zip(['max', 'median', 'min'], [tx_id_max, tx_id_med, tx_id_min]):
        df = pd.merge(df1[(df1.tx_id == tx_id) & (df1.coverage >= c)][['transcript_position', 'reactivity']].rename(
            columns={'reactivity': 'reactivity_1'}),
            df2[(df2.tx_id == tx_id) & (df2.coverage >= c)][['transcript_position', 'reactivity']].rename(
                columns={'reactivity': 'reactivity_2'}),
            on='transcript_position', how='outer')

        df = df.dropna()
        result[name] = (tx_id, df)
    return result
