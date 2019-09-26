# these can be overriden
NBATCH = 10

MINLEN = 10
MAXLEN = 500

NUMS := $(shell seq 1 $(NBATCH))
FSEQ = $(addsuffix .csv, $(addprefix tmp_output/data_, $(NUMS)))
FSTRUT = $(addsuffix .pkl.gz, $(addprefix tmp_output/data_, $(NUMS)))


setup: FORCE
	mkdir -p tmp_output; mkdir -p data

FORCE: ;

PHONY: SEQ


$(FSEQ): make_sequences.py
	python make_sequences.py --minlen $(MINLEN) --maxlen $(MAXLEN) --parts $(NBATCH) --out tmp_output/data

tmp_output/data_%.pkl.gz: compute_structure.py tmp_output/data_%.csv
	python compute_structure.py --inf tmp_output/data_$*.csv --out tmp_output/data_$*.pkl.gz


#all: SEQ

#tmp_output/data_%.pkl.gz: make_data_var_len.py
#	python make_data_var_len.py --minlen $(MINLEN) --maxlen $(MAXLEN) --num $(BATCHSIZE) --out tmp_output/data_$*.pkl.gz
#
data/premrna_seqs_$(MINLEN)_$(MAXLEN).pkl.gz: $(FSTRUCT) combine_data.py
	python combine_data.py data/premrna_seqs_$(MINLEN)_$(MAXLEN).pkl.gz tmp_output

all: $(FSTRUCT) data/premrna_seqs_$(MINLEN)_$(MAXLEN).pkl.gz

