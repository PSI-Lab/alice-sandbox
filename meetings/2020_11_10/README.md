

## Compare FE

For predicted best global structure not equal to ground truth,
calculate FE and compare with ground truth.


```
python compare_fe.py ../2020_10_13/data/rand_s1_bb_0p1_global_structs_60.pkl.gz data/rand_s1_bb_0p1_global_structs_60_fe.pkl.gz
```


Result:

- total number of examples: 45885

- examples where predicted best structure has negative free energy (as defined by RNAfold model): 32970

- correlation between free energy of RNAfold MFE structure and predicted best structure: 0.9187

- percent abs difference in free energy: abs((pred_fe - rnafold_fe)/rnafold_fe), distribution:

```
count    32970.000000
mean         0.117612
std          0.244401
min          0.000000
25%          0.000000
50%          0.000000
75%          0.096774
max          1.333333
```



## TODOs

move to top level util

end to end pipeline





