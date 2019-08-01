

## ModSeq data processing

Download `sacCer3` fasta and anno:

```bash
mkdir raw_data/fasta
cd raw_data/fasta
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrI.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrII.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrIII.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrIV.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrIX.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrM.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrV.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrVI.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrVII.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrVIII.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrX.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrXI.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrXII.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrXIII.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrXIV.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrXV.fa.gz
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/chromosomes/chrXVI.fa.gz
cd ../../

mkdir raw_data/annotation
cd raw_data/annotation
curl -O ftp://hgdownload.cse.ucsc.edu/goldenPath/sacCer3/database/ncbiRefSeqCurated.txt.gz
cd ../../
```
