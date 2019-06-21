import matplotlib
matplotlib.use('Agg')   # do not remove, this is to turn off X server so plot works on Linux
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(color_codes=True)
import cufflinks as cf
cf.go_offline()
cf.set_config_file(theme='ggplot')
from plotly.offline import plot, iplot
from scipy.stats import spearmanr


# load N rows for analysis
N = 2000000
df_father = pd.read_csv('data/GM12891_renatured.tab.gz', nrows=N, sep='\t')
df_mother = pd.read_csv('data/GM12892_renatured.tab.gz', nrows=N, sep='\t')

# discard the last transcript (since it's most probably incomplete)
df_father = df_father[df_father['tx_id'] != df_father.iloc[-1]['tx_id']]
df_mother = df_mother[df_mother['tx_id'] != df_mother.iloc[-1]['tx_id']]

# use transcript in the intersection of both dataset
s1 = df_father.tx_id.unique()
s2 = df_mother.tx_id.unique()
tx_common = set(s1).intersection(set(s2))
print("Number of transcripts to be processed: {}".format(len(tx_common)))

# for each transcript, find the correlation
# apply different min coverage threshold
min_coverages = [10, 20,
                 50, 100]  # no need to apply threshold below 10 since that's the threshold used in data processing
data = []
for tx_id in set(s1).intersection(set(s2)):
    row = {'tx_id': tx_id}

    for c in min_coverages:
        _df = pd.merge(
            df_father[(df_father.tx_id == tx_id) & (df_father.coverage_s1 >= c) & (df_father.coverage_v1 >= c)][
                ['transcript_position', 'reactivity']].rename(columns={'reactivity': 'reactivity_father'}),
            df_mother[(df_mother.tx_id == tx_id) & (df_mother.coverage_s1 >= c) & (df_mother.coverage_v1 >= c)][
                ['transcript_position', 'reactivity']].rename(columns={'reactivity': 'reactivity_mother'}),
            on='transcript_position', how='outer')

        df = _df.dropna()

        corr, pval = spearmanr(df['reactivity_father'], df['reactivity_mother'])

        _row = {
            'nf_{}'.format(c): len(_df['reactivity_father'].dropna()),
            'nm_{}'.format(c): len(_df['reactivity_mother'].dropna()),
            'n_{}'.format(c): len(df),
            'corr_{}'.format(c): corr,
            'pval_{}'.format(c): pval,
        }

        row.update(_row)
    data.append(row)

# histogram of correlation at different cutoff
df = pd.DataFrame(data)
fig = df[['corr_{}'.format(c) for c in min_coverages]].iplot(kind='histogram', bins=100, histnorm='probability',
                                                             asFigure=True)
plot(fig, filename="plot/family_pair_correlation_histogram.html", auto_open=False)


# Select transcript with high, med, low correlation (and low p value) and make scatter plot
# at cutoff 20 (arbitrary for now)
# with minimum number of data points 100
c_plot = 20
n_plot = 100
_corr = df[(df['pval_{}'.format(c_plot)] < 0.01) & (df['n_{}'.format(c_plot)] >= n_plot)]['corr_{}'.format(c_plot)]
tx_id_max = df[df['corr_{}'.format(c_plot)] == _corr.max()].iloc[0]['tx_id']
# to avoid the problem where median is not an actual data point, make sure the total number is odd
if len(_corr) % 2 != 0:
    _corr_median = _corr.median()
    tx_id_med = df[df['corr_{}'.format(c_plot)] == _corr_median].iloc[0]['tx_id']
else:
    _corr_median = _corr[:-1].median()  # quite arbitrary
    tx_id_med = df[df['corr_{}'.format(c_plot)] == _corr_median].iloc[0]['tx_id']
tx_id_min = df[df['corr_{}'.format(c_plot)] == _corr.min()].iloc[0]['tx_id']
print("Max correlation: {} from {}".format(_corr.max(), tx_id_max))
print("Median correlation: {} from {}".format(_corr_median, tx_id_med))
print("Min correlation: {} from {}".format(_corr.min(), tx_id_min))

for name, tx_id in zip(['max', 'median', 'min'], [tx_id_max, tx_id_med, tx_id_min]):
    df = pd.merge(df_father[(df_father.tx_id == tx_id) & (df_father.coverage_s1 >= c_plot) & (
            df_father.coverage_v1 > c_plot)][['transcript_position', 'reactivity']].rename(
        columns={'reactivity': 'reactivity_father'}),
        df_mother[(df_mother.tx_id == tx_id) & (df_mother.coverage_s1 >= c_plot) & (
                df_mother.coverage_v1 >= c_plot)][['transcript_position', 'reactivity']].rename(
            columns={'reactivity': 'reactivity_mother'}),
        on='transcript_position', how='outer')

    df = df.dropna()

    corr, pval = spearmanr(df['reactivity_father'], df['reactivity_mother'])

    fig = df.iplot(kind='scatter', x='reactivity_father', y='reactivity_mother', mode='markers', size=3,
                   xTitle='reactivity_father', yTitle='reactivity_mother',
                   title="{} corr ({}): {} ({})".format(tx_id, name, corr, pval), asFigure=True)
    plot(fig, filename="plot/family_pair_transcript_scatter_{}.html".format(name), auto_open=False)
