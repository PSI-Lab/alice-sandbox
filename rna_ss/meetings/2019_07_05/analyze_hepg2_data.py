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
import datacorral as dc
from utils import compute_corr, get_max_median_min, parse_expression_data


# load N rows for analysis
N = 2000000
# N = 100000  # debug
df_1 = pd.read_csv('data/HepG2_rep1.tab.gz', nrows=N, sep='\t')
df_2 = pd.read_csv('data/HepG2_rep2.tab.gz', nrows=N, sep='\t')
df_exp = parse_expression_data(dc.Client().get_path('Pnb4DN'), 'Hep G2')

# discard the last transcript (since it's most probably incomplete)
df_1 = df_1[df_1['tx_id'] != df_1.iloc[-1]['tx_id']]
df_2 = df_2[df_2['tx_id'] != df_2.iloc[-1]['tx_id']]

# use transcript in the intersection of both dataset
s1 = df_1.tx_id.unique()
s2 = df_2.tx_id.unique()
tx_common = set(s1).intersection(set(s2))
print("Number of transcripts to be processed: {}".format(len(tx_common)))

# compare two reps
min_coverages = [10, 20, 50, 100]
# df = compute_corr(df_1, df_2, tx_common, min_coverages, df_exp).rename(
#     columns={'reactivity_1': 'reactivity_rep1', 'reactivity_2': 'reactivity_rep2'})
df = compute_corr(df_1, df_2, tx_common, min_coverages, df_exp)
fig = df[['corr_{}'.format(c) for c in min_coverages]].iplot(kind='histogram', bins=100, histnorm='probability',
                                                             asFigure=True)
plot(fig, filename="plot/hepg2_rep1_rep2_correlation_histogram.html", auto_open=False)

# gene expression v.s. reactivity quality
# use n >= 20
_df_plot = df[['corr_20', 'log_tpm']].dropna()
_corr, _pval = spearmanr(_df_plot['corr_20'], _df_plot['log_tpm'])
fig = _df_plot.iplot(kind='scatter', x='corr_20', y='log_tpm', mode='markers', size=3,
                     xTitle='reactivity_correlation_between_reps', yTitle='gene_log_tpm',
                     title="Reactivity quality v.s. gene expression {} ({:.2e})".format(_corr, _pval), asFigure=True)
plot(fig, filename="plot/hepg2_reactivity_quality_gene_expression.html", auto_open=False)
# selected for significant ones
_df_plot = df[df['pval_20'] < 0.05][['corr_20', 'log_tpm']].dropna()
_corr, _pval = spearmanr(_df_plot['corr_20'], _df_plot['log_tpm'])
fig = _df_plot.iplot(kind='scatter', x='corr_20', y='log_tpm', mode='markers',
                     size=3, xTitle='reactivity_correlation_between_reps',
                     yTitle='gene_log_tpm',
                     title="Reactivity quality v.s. gene expression {} ({:.2e})".format(_corr, _pval),
                     asFigure=True)
plot(fig, filename="plot/hepg2_reactivity_quality_gene_expression_significant.html", auto_open=False)

result = get_max_median_min(df, df_1, df_2, 20, 100)
for name, _d in result.iteritems():
    tx_id, _df = _d
    _df = _df.rename(columns={'reactivity_1': 'reactivity_rep1', 'reactivity_2': 'reactivity_rep2'})
    corr, pval = spearmanr(_df['reactivity_rep1'], _df['reactivity_rep2'])
    fig = _df.iplot(kind='scatter', x='reactivity_rep1', y='reactivity_rep2', mode='markers', size=3,
                    xTitle='reactivity_rep1', yTitle='reactivity_rep2',
                    title="{} corr ({}): {} ({})".format(tx_id, name, corr, pval), asFigure=True)
    plot(fig, filename="plot/hepg2_rep1_rep2_transcript_scatter_{}.html".format(name), auto_open=False)



