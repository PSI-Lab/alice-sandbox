import pandas as pd
from scipy.stats import spearmanr

# load N rows for analysis
N = 2000000

# these data were processed by Omar
# from: /dg-shared/for_omar/data/SS_RNA_mapping/studies/wan2014/o
df_father = pd.read_csv('data/GM12891_renatured.tab.gz', nrows=N, sep='\t')
df_mother = pd.read_csv('data/GM12892_renatured.tab.gz', nrows=N, sep='\t')
df_child_rep1 = pd.read_csv('data/GM12878_native_deproteinized_replicate_1.tab.gz', nrows=N, sep='\t')
df_child_rep2 = pd.read_csv('data/GM12878_native_deproteinized_replicate_2.tab.gz', nrows=N, sep='\t')

for name, df in zip(['father', 'mother', 'child_rep1', 'child_rep2'],
                    [df_father, df_mother, df_child_rep1, df_child_rep2]):
    print(
        "{}: loaded {} rows, with {} rows ({}) missing reactivity".format(name, len(df),
                                                                          df['reactivity'].isna().sum(),
                                                                          df['reactivity'].isna().sum() / float(
                                                                              len(df))))


