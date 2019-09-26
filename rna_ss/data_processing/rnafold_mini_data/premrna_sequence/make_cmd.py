import sys

parts = int(sys.argv[1])
out_file = sys.argv[2]

with open(out_file, 'w') as f:
    for k in range(1, parts + 1):
        f.write("python compute_structure.py --inf tmp_output/data_{}.csv --out tmp_output/data_{}.pkl.gz\n".format(k, k))




