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
from utils import compute_corr, get_max_median_min


# load N rows for analysis
N = 2000000
df_1 = pd.read_csv('data/K562_vivo1_DMS_300.tab.gz', nrows=N, sep='\t')
df_2 = pd.read_csv('data/K562_vivo2_DMS_400.tab.gz', nrows=N, sep='\t')
df_3 = pd.read_csv('data/K562_vitro.tab.gz', nrows=N, sep='\t')

# discard the last transcript (since it's most probably incomplete)
df_1 = df_1[df_1['tx_id'] != df_1.iloc[-1]['tx_id']]
df_2 = df_2[df_2['tx_id'] != df_2.iloc[-1]['tx_id']]
df_3 = df_3[df_3['tx_id'] != df_3.iloc[-1]['tx_id']]

# use transcript in the intersection of both dataset
s1 = df_1.tx_id.unique()
s2 = df_2.tx_id.unique()
s3 = df_3.tx_id.unique()
tx_common = set(s1).intersection(set(s2)).intersection(set(s3))
print("Number of transcripts to be processed: {}".format(len(tx_common)))

# compare vivo1 and vivo2
min_coverages = [10, 20, 50, 100]
df = compute_corr(df_1, df_2, tx_common, min_coverages).rename(
    columns={'reactivity_1': 'reactivity_vivo1', 'reactivity_2': 'reactivity_vivo2'})
fig = df[['corr_{}'.format(c) for c in min_coverages]].iplot(kind='histogram', bins=100, histnorm='probability',
                                                             asFigure=True)
plot(fig, filename="plot/k562_vivo1_vivo2_correlation_histogram.html", auto_open=False)

result = get_max_median_min(df, df_1, df_2, 20, 100)
for name, _d in result.iteritems():
    tx_id, _df = _d
    _df = _df.rename(columns={'reactivity_1': 'reactivity_vivo1', 'reactivity_2': 'reactivity_vivo2'})
    corr, pval = spearmanr(_df['reactivity_vivo1'], _df['reactivity_vivo2'])
    fig = _df.iplot(kind='scatter', x='reactivity_vivo1', y='reactivity_vivo2', mode='markers', size=3,
                    xTitle='reactivity_vivo1', yTitle='reactivity_vivo2',
                    title="{} corr ({}): {} ({})".format(tx_id, name, corr, pval), asFigure=True)
    plot(fig, filename="plot/k562_vivo1_vivo2_transcript_scatter_{}.html".format(name), auto_open=False)


# compare vivo1 and vitro
min_coverages = [10, 20, 50, 100]
df = compute_corr(df_1, df_3, tx_common, min_coverages).rename(
    columns={'reactivity_1': 'reactivity_vivo1', 'reactivity_2': 'reactivity_vitro'})
fig = df[['corr_{}'.format(c) for c in min_coverages]].iplot(kind='histogram', bins=100, histnorm='probability',
                                                             asFigure=True)
plot(fig, filename="plot/k562_vivo1_vitro_correlation_histogram.html", auto_open=False)

result = get_max_median_min(df, df_1, df_3, 20, 100)
for name, _d in result.iteritems():
    tx_id, _df = _d
    _df = _df.rename(columns={'reactivity_1': 'reactivity_vivo1', 'reactivity_2': 'reactivity_vitro'})
    corr, pval = spearmanr(_df['reactivity_vivo1'], _df['reactivity_vitro'])
    fig = _df.iplot(kind='scatter', x='reactivity_vivo1', y='reactivity_vitro', mode='markers', size=3,
                    xTitle='reactivity_vivo1', yTitle='reactivity_vitro',
                    title="{} corr ({}): {} ({})".format(tx_id, name, corr, pval), asFigure=True)
    plot(fig, filename="plot/k562_vivo1_vitro_transcript_scatter_{}.html".format(name), auto_open=False)
