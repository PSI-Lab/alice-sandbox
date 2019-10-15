import sys
import pandas as pd
import numpy as np
from collections import Counter
from utils import EvalMetric
# import matplotlib.pyplot as plt
# # import seaborn as sns
# import pandas as pd
# # sns.set(color_codes=True)
# import cufflinks as cf
# cf.go_offline()
# cf.set_config_file(theme='ggplot')
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
# import plotly.tools as tls
# import plotly.graph_objs as go
# import plotly.figure_factory as ff
# init_notebook_mode(connected=True)


in_file = sys.argv[1]
# df = pd.read_pickle('result/prediction.rand_seqs_var_len_sample_ensemble_10_100_100.2019_09_12_1.sample100.pkl')
df = pd.read_pickle(in_file)


# convert to tuple so we can use set()
def _to_tuple(idxes):
    x = []
    for _a in idxes:
        a = []
        for i, j in zip(_a[0], _a[1]):
            a.append((i, j))
        if len(a) > 0:
            a = tuple(a)
            x.append(a)
    return x


def _to_mat(x, l):
    # x is tuple of tuple (2 elements)
    # convert to list of 2 lists
    y = [[a[0] for a in x], [a[1] for a in x]]
    z = np.zeros((l, l))
    z[y] = 1
    return z


n_no_struct = 0
df_metric = []

for _, row in df.iterrows():
    seq = row['seq']
    rnafold_struct_count = Counter(_to_tuple(row['one_idx']))
    model_struct_count = Counter(_to_tuple(row['pred_idx']))
    # only keep structures that occurs more than 2 times (2% in sampled structures)
    rnafold_struct_count = {k: v for k, v in rnafold_struct_count.iteritems() if v >=2}
    model_struct_count = {k: v for k, v in model_struct_count.iteritems() if v >=2}
#     # only keep structures that occurs more than 5 times (5% in sampled structures)
#     rnafold_struct_count = {k: v for k, v in rnafold_struct_count.iteritems() if v >=5}
#     model_struct_count = {k: v for k, v in model_struct_count.iteritems() if v >=5}
#     # only keep structures that occurs more than 10 times (10% in sampled structures)
#     rnafold_struct_count = {k: v for k, v in rnafold_struct_count.iteritems() if v >=10}
#     model_struct_count = {k: v for k, v in model_struct_count.iteritems() if v >=10}
    
    # skip the case if RNAfold outputs no structure
    if len(rnafold_struct_count) == 0:
        n_no_struct += 1
        continue
    
    # for each structure from RNAfold
    # evaluate against all structures from the model, pick the closest one, record performance
#     result = []
    for rnafold_struct in rnafold_struct_count.keys():
        r_s = _to_mat(rnafold_struct, len(seq))
        sensitivity = 0
        ppv = 0
        f_measure = 0
        best_m_s = None
        for model_struct in model_struct_count.keys():
            m_s = _to_mat(model_struct, len(seq))
            _sensitivity = EvalMetric.sensitivity(m_s, r_s)
            _ppv = EvalMetric.ppv(m_s, r_s)
            _f_measure = EvalMetric.f_measure(_sensitivity, _ppv)
            if _f_measure > f_measure:
                sensitivity = _sensitivity + 0  # copy
                ppv = _ppv + 0
                f_measure = _f_measure + 0
                best_m_s = list(model_struct)
#         result.append((sensitivity, ppv, f_measure))
        df_metric.append({
            'seq': seq,
            'rnafold_idx': list(rnafold_struct),
            'closest_model_idx': best_m_s,
            'sensitivity': sensitivity,
            'ppv': ppv,
            'f_measure': f_measure,
        })

print("n_no_struct: {}".format(n_no_struct))
df_metric = pd.DataFrame(df_metric)


# fig = df_metric[['sensitivity', 'ppv']].iplot(kind='histogram', bins=20,
#                                                     histnorm='probability', opacity=1,
#                                                      barmode='group', asFigure=True)
# fig = fig.update(layout=dict(
#     paper_bgcolor='rgba(0,0,0,0)',
#     plot_bgcolor='rgba(0,0,0,0)',
#     xaxis=dict(showgrid=False, zeroline=False,
#                tickfont=dict(size=18, color='#000'),
#                titlefont=dict(size=24, color='#000'),
#                showline=True, mirror=True),
#     yaxis=dict(showgrid=True, gridwidth=1, gridcolor='darkgray',
#                tickfont=dict(size=18, color='#000'),
#                titlefont=dict(size=24, color='#000'),
#                title='Percentage', showline=True, mirror=True),
#     title='Performance on test dataset',
#     titlefont=dict(size=24, color='#000'),
#     legend=dict(orientation="h",
#                 font=dict(
# #             family='sans-serif',
#             size=24,
#             color='#000'
#         ),),
# ))


print(len(df_metric))

print(sum(df_metric['ppv']>0.8)/float(len(df_metric)))

# df_metric[['seq', 'sensitivity', 'ppv', 'f_measure']].to_csv('/Users/alicegao/data/tmp/perf_rand_ensemble_metric.csv', index=False)
print df_metric.describe()
