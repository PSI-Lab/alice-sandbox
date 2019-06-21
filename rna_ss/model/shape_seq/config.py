config = {
    # need to use refseq since we need to match the transcript IDs in the processed reactivity data
    'genome_annotation': 'ucsc_refseq.2017-06-25',

    # this is one of the reactivity data processed by Omar
    # /dg-shared/for_omar/data/SS_RNA_mapping/studies/rouskin2014/out/rf_clean_1/K562_vivo1_DMS_300.tab.gz
    'raw_data': 'HkLXTF',

    # dataset
    'gtrack': 'data/data.gtrack',

    # all chromosomes used for training and validation
    # will be split into 5 folds
    'chrom_all': ['chr%d' % x for x in range(1, 23)],
    # chrom sizes, from
    'chrom_sizes': [['chr1', 249250621],
                    ['chr2', 243199373],
                    ['chr3', 198022430],
                    ['chr4', 191154276],
                    ['chr5', 180915260],
                    ['chr6', 171115067],
                    ['chr7', 159138663],
                    ['chr8', 146364022],
                    ['chr9', 141213431],
                    ['chr10', 135534747],
                    ['chr11', 135006516],
                    ['chr12', 133851895],
                    ['chr13', 115169878],
                    ['chr14', 107349540],
                    ['chr15', 102531392],
                    ['chr16', 90354753],
                    ['chr17', 81195210],
                    ['chr18', 78077248],
                    ['chr20', 63025520],
                    ['chr19', 59128983],
                    ['chr22', 51304566],
                    ['chr21', 48129895]],
    # save disjoint intervals for each of the 5 folds
    'interval_folds': [
        'data/intervals_fold_0.txt',
        'data/intervals_fold_1.txt',
        'data/intervals_fold_2.txt',
        'data/intervals_fold_3.txt',
        'data/intervals_fold_4.txt',
    ],

    # model
    'n_filters': 32,  # n filter in first layer
    'residual_conv': [
        {'num_filter': 32, 'filter_width': 11, 'dilation': 1},
        {'num_filter': 32, 'filter_width': 11, 'dilation': 1},
        {'num_filter': 32, 'filter_width': 11, 'dilation': 1},
        {'num_filter': 32, 'filter_width': 11, 'dilation': 1},
        {'num_filter': 32, 'filter_width': 11, 'dilation': 4},
        {'num_filter': 32, 'filter_width': 11, 'dilation': 4},
        {'num_filter': 32, 'filter_width': 11, 'dilation': 4},
        {'num_filter': 32, 'filter_width': 11, 'dilation': 4},
        # {'num_filter': 32, 'filter_width': 21, 'dilation': 10},
        # {'num_filter': 32, 'filter_width': 21, 'dilation': 10},
        # {'num_filter': 32, 'filter_width': 21, 'dilation': 10},
        # {'num_filter': 32, 'filter_width': 21, 'dilation': 10},
        # {'num_filter': 32, 'filter_width': 41, 'dilation': 25},
        # {'num_filter': 32, 'filter_width': 41, 'dilation': 25},
        # {'num_filter': 32, 'filter_width': 41, 'dilation': 25},
        # {'num_filter': 32, 'filter_width': 41, 'dilation': 25},
    ],


    'n_repeat_in_residual_unit': 2,
    'skip_conn_every_n': 4,

    'example_length': 5000,
    'batch_size': 10,
    'learning_rate': 0.00005,
    'residual': True,
    'skipconn': True,
    'gated': False,

    'num_epoch': 200,
    'num_batch_for_validation': 10,
    'es_patience': 10,

    # final models
    'model_dir': 'model/',

}
