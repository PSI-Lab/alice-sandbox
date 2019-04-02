# uPBM data modelling

## Introduction

Data was kindly provided by Jingkang Zhao.

The objective of this analysis is to get a sense of the baseline performance of conv nets on this type of dataset.

## Data Processing

- primer `GTCTGTGTTCCGTTGTCCGTGCTG` is being stripped off if present

- sequence with different length in the same dataset is being discarded (only a few sequences)

- convert to log intensity, drop NaN's

## Model & Training

- build one model per each dataset

- one-hot encoding of input sequence

- 80% training + validation, 20% test

- 5-fold cross validation

- Model: 3 layer dilated conv, global max pooling at the end

- Tried hyperparameter optimization on number of filters, layers, epochs, etc. but didn't have enough time to finish (partial result available)

## Result

For detailed result on each dataset, see [result/](result/) for the fixed parameter run, and [result_hyper_param/](result_hyper_param/) for the best result from hyperparmaeter optimization (for only selected dataset, still running).
The directory structure mirrors that of the `data/` and is the same as in the Dropbox folder.

For each dataset, there is one `.csv` file that records the training and validation r2 for each fold, as well as the training and test set pearson correlation,
where the training prediction is computed using cross-validation models (e.g. the model used to predict on a certain data point hasn't seen the data during training),
and the test prediction is computed using an ensemble of the 5 models.
There is also one `.html` file with an interactive plot that shows the correlation between prediction and target value. (note that GitHub won't render html files, so you need to save them and reopen using your browser)

To reproduce result one one dataset, for fixed hyperparameter, run:

```bash
python train_one_tf.py {TF_NAME} {DATA_FILE} {INPUT_DIR} {OUTPUT_DIR} {GPU_ID}
# e.g. python train_one_tf.py BHLHE40 "Mus_musculus|M00251_1.94d|Badis09|Bhlhb2_1274.3=v2.txt" data result 0
```

for hyperparameter optimization, run:

```bash
python train_hyper_param.py {TF_NAME} {DATA_FILE} {INPUT_DIR} {OUTPUT_DIR} {GPU_ID}
# e.g. python train_hyper_param.py BHLHE40 "Mus_musculus|M00251_1.94d|Badis09|Bhlhb2_1274.3=v2.txt" data result_hyper_param 2
```

Here we summarize the performance on different dataset:

### Optimized hyperparmeter

| name                                                          | cross_validation_pearson_corr | test_set_pearson_corr |
|---------------------------------------------------------------|-------------------------------|-----------------------|
| BHLHE40/Mus_musculus\|M00251_1.94d\|Badis09\|Bhlhb2_1274.3=v2 | 0.7451570952404395            | 0.7827164724139747    |
| BHLHE40/Mus_musculus\|M00252_1.94d\|Zoo_01\|2878              | 0.8346365761063146            | 0.8701574654009504    |
| BHLHE40/Mus_musculus\|M00253_1.94d\|Zoo_01\|1308              | 0.7536858646162291            | 0.7892722942206171    |
| CEBPB/Mus_musculus\|M00401_1.94d\|Zoo_01\|1586                | 0.722472334200782             | 0.7572544737880601    |
| CEBPB/Mus_musculus\|M00406_1.94d\|Zoo_01\|2537                | 0.7368181887453125            | 0.7586573463582926    |
| CEBPB/Mus_musculus\|M00415_1.94d\|DREAM_contest\|1408         | 0.8383098016454329            | 0.8738197732818648    |
| CEBPB/Mus_musculus\|M00416_1.94d\|Zoo_01\|1598                | 0.7691956975042746            | 0.8151561000175771    |
| CREB1/Mus_musculus\|M00388_1.94d\|Zoo_01\|2502                | 0.639600613531907             | 0.6799032495080467    |
| CREB1/Mus_musculus\|M00392_1.94d\|Zoo_01\|1581                | 0.6130146539367552            | 0.6629919148017454    |
| CREB1/Mus_musculus\|M00420_1.94d\|Zoo_01\|1307                | 0.810391737785821             | 0.8302995267619555    |
| EGR1/Caenorhabditis_elegans\|M00701_1.94d\|CEPD\|7901         | 0.7013276577407939            | 0.7457746730983789    |
| EGR1/Homo_sapiens\|NA\|Unpublished\|EGR1                      | 0.8137741417740282            | 0.8427362418596592    |
| EGR1/Mus_musculus\|M00609_1.94d\|DREAM_contest\|1440          | 0.7514625933215091            | 0.8094788799533804    |
| EGR1/Mus_musculus\|M00611_1.94d\|DREAM_contest\|667           | 0.7941112523810475            | 0.8304835652077647    |
| ELF1/Mus_musculus\|M00995_1.94d\|Wei10\|Elf4                  | 0.6862933854515259            | 0.7357501114923753    |
| ELF1/Mus_musculus\|M00997_1.94d\|Wei10\|Elf2                  | 0.78516152461152              | 0.8214823783785887    |
| ELK1/Homo_sapiens\|NA\|Shen2018\|ELK1                         | 0.6252532765899076            | 0.6760293874512416    |
| ELK1/Mus_musculus\|M00978_1.94d\|Wei10\|Etv1                  | 0.8579031521057902            | 0.8764494490114934    |
| ELK1/Mus_musculus\|M00979_1.94d\|Wei10\|Gm5454                | 0.8301965986020501            | 0.8508644275677111    |
| ELK1/Mus_musculus\|M00981_1.94d\|Wei10\|Elk3                  | 0.9424647931353876            | 0.952407676112936     |
| ELK1/Mus_musculus\|M00984_1.94d\|Wei10\|Elk1                  | 0.8942737836841137            | 0.91308829120732      |
| ETS1/Homo_sapiens\|NA\|Shen2018\|ETS1                         | 0.7860707024192487            | 0.8155344502161378    |

Raw data can be found in [performance_summary_hpt.csv](performance_summary_hpt.csv).


### Fixed hyperparameter

| name                                                          | cross_validation_pearson_corr | test_set_pearson_corr |
|---------------------------------------------------------------|-------------------------------|-----------------------|
| BHLHE40/Mus_musculus\|M00251_1.94d\|Badis09\|Bhlhb2_1274.3=v2 | 0.7306178394596959            | 0.7972783943413743    |
| BHLHE40/Mus_musculus\|M00252_1.94d\|Zoo_01\|2878              | 0.8349057707348844            | 0.8563795934567219    |
| BHLHE40/Mus_musculus\|M00253_1.94d\|Zoo_01\|1308              | 0.7508202126647565            | 0.7814262733112454    |
| CEBPB/Mus_musculus\|M00401_1.94d\|Zoo_01\|1586                | 0.7234387655327106            | 0.7488088620792193    |
| CEBPB/Mus_musculus\|M00406_1.94d\|Zoo_01\|2537                | 0.7234536091900039            | 0.7717770696507772    |
| CEBPB/Mus_musculus\|M00415_1.94d\|DREAM_contest\|1408         | 0.8399575948237681            | 0.875149942208807     |
| CEBPB/Mus_musculus\|M00416_1.94d\|Zoo_01\|1598                | 0.7779948650017559            | 0.8205388284151308    |
| CREB1/Mus_musculus\|M00388_1.94d\|Zoo_01\|2502                | 0.6213930383356862            | 0.6827695699743976    |
| CREB1/Mus_musculus\|M00392_1.94d\|Zoo_01\|1581                | 0.6095123416305136            | 0.6508552539057834    |
| CREB1/Mus_musculus\|M00420_1.94d\|Zoo_01\|1307                | 0.8160272102030118            | 0.8352593253676124    |
| EGR1/Caenorhabditis_elegans\|M00701_1.94d\|CEPD\|7901         | 0.6911383896112053            | 0.7474916243581304    |
| EGR1/Homo_sapiens\|NA\|Unpublished\|EGR1                      | 0.8083756496024712            | 0.8509591293552122    |
| EGR1/Mus_musculus\|M00609_1.94d\|DREAM_contest\|1440          | 0.7356801570186496            | 0.7928860416575919    |
| EGR1/Mus_musculus\|M00611_1.94d\|DREAM_contest\|667           | 0.7831435776864909            | 0.8204229005689322    |
| EGR1/Mus_musculus\|M00613_1.94d\|Zoo_01\|951                  | 0.8506420458604789            | 0.8785830835097644    |
| EGR1/Mus_musculus\|M00614_1.94d\|Berger06\|Zif268             | 0.28093504143980297           | 0.3434691694518807    |
| ELF1/Mus_musculus\|M00995_1.94d\|Wei10\|Elf4                  | 0.6930250772779756            | 0.7361311779954961    |
| ELF1/Mus_musculus\|M00997_1.94d\|Wei10\|Elf2                  | 0.7851429469712623            | 0.8204266070993008    |
| ELK1/Homo_sapiens\|NA\|Shen2018\|ELK1                         | 0.5924031677252595            | 0.6809353507364813    |
| ELK1/Mus_musculus\|M00978_1.94d\|Wei10\|Etv1                  | 0.8336842485079539            | 0.868270280489372     |
| ELK1/Mus_musculus\|M00979_1.94d\|Wei10\|Gm5454                | 0.8147614743676921            | 0.8555843147890081    |
| ELK1/Mus_musculus\|M00981_1.94d\|Wei10\|Elk3                  | 0.9421279523441894            | 0.9578325360418236    |
| ELK1/Mus_musculus\|M00984_1.94d\|Wei10\|Elk1                  | 0.8925080088621106            | 0.9203441121706616    |
| ETS1/Homo_sapiens\|NA\|Shen2018\|ETS1                         | 0.7839414635963861            | 0.8312140644773498    |
| ETS1/Mus_musculus\|M00978_1.94d\|Wei10\|Etv1                  | 0.8496673231655224            | 0.8750591000767217    |
| ETS1/Mus_musculus\|M00979_1.94d\|Wei10\|Gm5454                | 0.8246067894279518            | 0.8496364778807628    |
| ETS1/Mus_musculus\|M00982_1.94d\|Badis09\|Gabpa_2829.2=v2     | 0.7992486651662422            | 0.8313208784382374    |
| ETS1/Mus_musculus\|M00984_1.94d\|Wei10\|Elk1                  | 0.8944071233101947            | 0.9135564113241372    |
| ETS1/Mus_musculus\|M00996_1.94d\|Wei10\|Ets1                  | 0.7503662641593036            | 0.8025495421060618    |
| FOXA1/Mus_musculus\|M01035_1.94d\|Badis09\|Foxa2_2830.2=v1    | 0.8591214304184309            | 0.8874433199401606    |
| GABPA/Mus_musculus\|M00978_1.94d\|Wei10\|Etv1                 | 0.8372362692989179            | 0.8785202696302953    |
| GABPA/Mus_musculus\|M00979_1.94d\|Wei10\|Gm5454               | 0.8181682784213066            | 0.8529567137091221    |
| GABPA/Mus_musculus\|M00982_1.94d\|Badis09\|Gabpa_2829.2=v2    | 0.7941401521309595            | 0.8385599385246382    |
| GABPA/Mus_musculus\|M00983_1.94d\|Wei10\|Gabpa                | 0.7382126609886285            | 0.7909244572410263    |
| GABPA/Mus_musculus\|M00984_1.94d\|Wei10\|Elk1                 | 0.8874023295443169            | 0.9103559184690804    |
| GABPA/Mus_musculus\|M00996_1.94d\|Wei10\|Ets1                 | 0.7522231702602625            | 0.7925049265310654    |
| GATA1/Homo_sapiens\|NA\|Unpublished\|GATA1                    | 0.9500224234549236            | 0.964941896841874     |
| GATA1/Mus_musculus\|M01105_1.94d\|Badis09\|Gata3_1024.3=v1    | 0.9242209330895224            | 0.9383634126023932    |
| GATA1/Mus_musculus\|M01106_1.94d\|Badis09\|Gata5_3768.1=v1    | 0.8399091100514484            | 0.868769712034895     |
| GATA1/Mus_musculus\|M01107_1.94d\|DREAM_contest\|572          | 0.759657588408878             | 0.7899999727778316    |
| GATA3/Homo_sapiens\|NA\|Unpublished\|GATA1                    | 0.9493941522936624            | 0.963054357849587     |
| GATA3/Mus_musculus\|M01105_1.94d\|Badis09\|Gata3_1024.3=v1    | 0.9222712536971428            | 0.9368910787786412    |
| GATA3/Mus_musculus\|M01106_1.94d\|Badis09\|Gata5_3768.1=v1    | 0.8340631135027483            | 0.8682024450974121    |
| GATA3/Mus_musculus\|M01107_1.94d\|DREAM_contest\|572          | 0.7529391347651458            | 0.7681107117135483    |
| IRF4/Mus_musculus\|M01731_1.94d\|Badis09\|Irf4_3476.1=v2      | 0.8734345477431505            | 0.8925937786842701    |
| MAFK/Mus_musculus\|M00383_1.94d\|Badis09\|Mafk_3106.2=v2      | 0.8228170923668979            | 0.8459898368142384    |
| MAFK/Mus_musculus\|M00405_1.94d\|Zoo_01\|205                  | 0.624147110921872             | 0.6881164913839614    |
| MAX/Homo_sapiens\|NA\|Shen2018\|MAX                           | 0.9279550379034562            | 0.9474664973723212    |
| MAX/Mus_musculus\|M00269_1.94d\|Badis09\|Max_3863.1=v2        | 0.8684113552489463            | 0.8913389928058483    |
| MAX/Mus_musculus\|M00270_1.94d\|Badis09\|Max_3864.1=v2        | 0.6537707980679103            | 0.7265209663138049    |
| MAX/Mus_musculus\|M00271_1.94d\|Zoo_01\|855                   | 0.7342998031671544            | 0.7555042508457309    |
| PAX5/Caenorhabditis_elegans\|M02066_1.94d\|CEPD\|10189        | 0.3588517834197407            | 0.5115914790263577    |
| PAX5/Danio_rerio\|M02061_1.94d\|Zoo_01\|5899                  | 0.7128812316935422            | 0.7946158101255625    |
| PAX5/Xenopus_tropicalis\|M02063_1.94d\|Zoo_01\|5729           | 0.38598156378121995           | 0.4394231417381816    |
| PBX3/Homo_sapiens\|M01230_1.94d\|Barrera2016\|PBX4_REF_R2     | 0.6409597603645849            | 0.7107066417109368    |
| POU2F2/Danio_rerio\|M01676_1.94d\|Zoo_01\|6777                | 0.9252672204809624            | 0.9486760868849023    |
| POU2F2/Mus_musculus\|M01689_1.94d\|Berger08\|Pou2f2_3748.1    | 0.8654680656564188            | 0.8901630965256904    |
| POU2F2/Mus_musculus\|M01693_1.94d\|Berger08\|Pou2f1_3081.2    | 0.9259395476852308            | 0.9403519165925596    |
| POU2F2/Mus_musculus\|M01694_1.94d\|DREAM_contest\|1085        | 0.7738534086791576            | 0.8110411123543437    |
| POU2F2/Nasonia_vitripennis\|M01706_1.94d\|Zoo_01\|6832        | 0.9215540575317492            | 0.9364218648861871    |
| POU2F2/Takifugu_rubripes\|M01700_1.94d\|Zoo_01\|6957          | 0.8948954428868477            | 0.91927522681336      |
| RFX5/Mus_musculus\|M02099_1.94d\|DREAM_contest\|1416          | 0.7389200671802982            | 0.7914219974139621    |
| RFX5/Mus_musculus\|M02100_1.94d\|Badis09\|Rfxdc2_3516.1=v1    | 0.7470703436898303            | 0.8066871039208956    |
| RUNX1/Homo_sapiens\|NA\|Shen2018\|RUNX1                       | 0.9152680896647052            | 0.9378692781829666    |
| RUNX1/Homo_sapiens\|NA\|Shen2018\|RUNX2                       | 0.9043145653569564            | 0.9298329352519632    |
| RUNX3/Homo_sapiens\|NA\|Shen2018\|RUNX1                       | 0.9148221119397534            | 0.9321934709140064    |
| RUNX3/Homo_sapiens\|NA\|Shen2018\|RUNX2                       | 0.9061783201897916            | 0.93604502599917      |
| SP1/Mus_musculus\|M00595_1.94d\|Zoo_01\|12                    | 0.7019698117130044            | 0.7310238145780066    |
| SP1/Mus_musculus\|M00596_1.94d\|Badis09\|Sp4_1011.2=v1        | 0.8348302243435297            | 0.8613003858823381    |
| SP1/PBM_CONSTRUCTS\|M00804_1.94d\|Lam11\|953                  | 0.5728883215772297            | 0.6199872153147342    |
| SPI1/Mus_musculus\|M00973_1.94d\|Badis09\|Sfpi1_1034.2=v1     | 0.9033086798693298            | 0.9280656558305932    |
| SPI1/Mus_musculus\|M00974_1.94d\|Wei10\|Sfpi1                 | 0.9211539003241928            | 0.9394681613797884    |
| SRF/Caenorhabditis_elegans\|M01772_1.94d\|CEPD\|10570         | 0.5553684747509984            | 0.6575511088928137    |
| SRF/Dictyostelium_discoideum\|M01770_1.94d\|Zoo_01\|981       | 0.6621376932367121            | 0.6912467097836159    |
| STAT3/Mus_musculus\|NA\|Unpublished\|Stat3                    | 0.6634069932491742            | 0.7789794478811191    |
| TBP/Homo_sapiens\|NA\|Unpublished\|TBP                        | 0.8773664089906181            | 0.8970920701757046    |
| TBP/Mus_musculus\|M02225_1.94d\|Badis09\|Tbp_pr781.1=v2       | 0.7354401248363999            | 0.7697101069898198    |
| TCF12/Homo_sapiens\|NA\|Unpublished\|TCF3                     | 0.9139542135956576            | 0.9407276811676164    |
| TCF12/Mus_musculus\|M00231_1.94d\|DREAM_contest\|1466         | 0.8286570394109216            | 0.8692991991889454    |
| TCF12/Mus_musculus\|M00233_1.94d\|Badis09\|Tcfe2a_3865.1=v2   | 0.8446686844224721            | 0.8802257353361194    |
| TCF12/Mus_musculus\|M00254_1.94d\|Zoo_01\|1362                | 0.7732987137888487            | 0.813871089352188     |
| TCF12/Mus_musculus\|M00266_1.94d\|Zoo_01\|207                 | 0.9284404611857068            | 0.9427925666214536    |
| TCF7L2/Mus_musculus\|M02156_1.94d\|Badis09\|Tcf7_0950.2=v2    | 0.8278569640377726            | 0.8549281759321495    |
| TCF7L2/Mus_musculus\|M02162_1.94d\|Badis09\|Tcf7l2_3461.1=v1  | 0.8231196700433523            | 0.8554661066625096    |
| TCF7L2/Mus_musculus\|M02164_1.94d\|Badis09\|Lef1_3504.1=v1    | 0.8472080410808712            | 0.8776892511971481    |
| TFAP2C/Homo_sapiens\|NA\|Unpublished\|TFAP2A                  | 0.6688873771818175            | 0.6987432618420205    |
| TFAP2C/Mus_musculus\|M00117_1.94d\|Badis09\|Tcfap2a_2337.3=v2 | 0.7868446333325417            | 0.8449423990203028    |
| TFAP2C/Mus_musculus\|M00118_1.94d\|Badis09\|Tcfap2b_3988.1=v2 | 0.7815428426861991            | 0.823402626211613     |
| TFAP2C/Mus_musculus\|M00119_1.94d\|Badis09\|Tcfap2c_2912.2=v2 | 0.8142817148658602            | 0.8652624822211389    |
| USF1/Mus_musculus\|M00268_1.94d\|Zoo_01\|2533                 | 0.4403879980484332            | 0.4903767406172708    |
| USF2/Mus_musculus\|M00268_1.94d\|Zoo_01\|2533                 | 0.4490969921362388            | 0.4937569470100023    |

Raw data can be found in [performance_summary_fixed.csv](performance_summary_fixed.csv).

## Future Work

- Finish running the current hyperparameter optimization

- Add more hyperparameters, optimizing on each fold

- Training one model for all TFs, with shared conv filter weights

