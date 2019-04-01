import os
import pandas as pd


dfs = []
for root, dirs, files in os.walk("result"):
    for file in files:
        if file.endswith(".csv"):
            tf_name = root.replace('result/', '')
            data_name = file.replace('.csv', '')
            _df = pd.read_csv(os.path.join(root, file))
            _df['tf_name'] = tf_name
            _df['data_name'] = data_name
            dfs.append(_df)
df = pd.concat(dfs)

# hpt result
dfs = []
for root, dirs, files in os.walk("result_hyper_param"):
    for file in files:
        if file.endswith(".csv"):
            tf_name = root.replace('result_hyper_param/', '')
            data_name = file.replace('.csv', '')
            _df = pd.read_csv(os.path.join(root, file))
            _df['tf_name'] = tf_name
            _df['data_name'] = data_name
            # param were stored in multiple rows, drop duplicates (until we have different param per each fold)
            _df.drop_duplicates(subset=['task'], inplace=True)
            dfs.append(_df)
df_hpt = pd.concat(dfs)

df_comp = df.copy()
df_comp['name'] = df_comp['tf_name'] + '/' + df_comp['data_name']
df_comp = pd.DataFrame(df_comp.reset_index().pivot('name', 'task', 'val').to_records())

df_hpt_comp = df_hpt.copy()
df_hpt_comp['name'] = df_hpt_comp['tf_name'] + '/' + df_hpt_comp['data_name']
df_hpt_comp = pd.DataFrame(df_hpt_comp.reset_index().pivot('name', 'task', 'val').to_records())

df_comp['training_r2'] = (df_comp['training_fold_0_r2'] + df_comp['training_fold_1_r2'] + df_comp[
    'training_fold_2_r2'] + df_comp['training_fold_3_r2'] + df_comp['training_fold_4_r2']) / 5

df_hpt_comp['training_r2'] = (df_hpt_comp['training_fold_0_r2'] + df_hpt_comp['training_fold_1_r2'] + df_hpt_comp[
    'training_fold_2_r2'] + df_hpt_comp['training_fold_3_r2'] + df_hpt_comp['training_fold_4_r2']) / 5

df_comp[['name', 'cross_validation_pearson_corr', 'test_set_pearson_corr']].to_csv('performance_summary_fixed.csv',
                                                                                   index=False)

df_hpt_comp[['name', 'cross_validation_pearson_corr', 'test_set_pearson_corr']].to_csv('performance_summary_hpt.csv',
                                                                                       index=False)
