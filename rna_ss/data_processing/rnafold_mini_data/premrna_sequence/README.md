

setting maxseq, otherwise running for too long...

```bash
mkdir -p tmp_output
mkdir -p data
mkdir -p cmd
python make_sequences.py --minlen 10 --maxlen 200 --maxseq 1000000 --parts 200 --out tmp_output/data
```

above takes around 1 min.

```bash
python make_cmd.py 200 cmd/cmd.txt
```

``bash
parallel --jobs 18 < cmd/cmd.txt
```

wait till the above to finish....

```bash
python combine_data.py data/premrna_seqs_10_200.pkl.gz tmp_output
```

