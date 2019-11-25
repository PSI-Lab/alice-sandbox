# NUM can be overriden
NUM = 50
BATCHSIZE = 10
NBATCH = $(shell expr $(NUM) / $(BATCHSIZE))

MINLEN = 10
MAXLEN = 500

NUMS := $(shell seq 1 $(NBATCH))
VARS = $(addsuffix .pkl.gz, $(addprefix tmp_output/data_, $(NUMS)))


setup: FORCE
	mkdir -p tmp_output; mkdir -p data

FORCE: ;

tmp_output/data_%.pkl.gz: make_data_var_len.py
	python make_data_var_len.py --minlen $(MINLEN) --maxlen $(MAXLEN) --num $(BATCHSIZE) --out tmp_output/data_$*.pkl.gz

data/rand_seqs_var_len_sample_mfe_$(MINLEN)_$(MAXLEN)_$(NUM).pkl.gz: $(VARS) combine_data.py
	python combine_data.py data/rand_seqs_var_len_sample_mfe_$(MINLEN)_$(MAXLEN)_$(NUM).pkl.gz tmp_output

all: data/rand_seqs_var_len_sample_mfe_$(MINLEN)_$(MAXLEN)_$(NUM).pkl.gz

