config = {
    # need to use refseq since we need to match the transcript IDs in the processed reactivity data
    'genome_annotation': 'ucsc_refseq.2017-06-25',

    # dataset
    # generated by https://github.com/PSI-Lab/alice-sandbox/tree/6f1c6125435aa906c07a71a195cd964cc88eda8c/rna_ss/data_processing/k562_all
    'gtrack': 'data/reactivity_combined.gtrack',
    'all_inervals': 'data/intervals_combined.csv',

    # chromosome for different folds
    'chrom_folds': [['chr1', 'chr10', 'chr15', 'chr20', 'chr22'],
                    ['chr2', 'chr9', 'chr14', 'chr19', 'chr21'],
                    ['chr3', 'chr8', 'chr11', 'chr17'],
                    ['chr4', 'chr7', 'chr12', 'chr18'],
                    ['chr5', 'chr6', 'chr13', 'chr16', 'chrX']],

    # example filter
    'min_log_tpm': 1,   # set to -inf to turn this off
    'min_rep_corr':  0.2,   # set to -inf to turn this off
    'example_reweighting': True,   # see code for details on method

    # # model
    # 'n_filters': 128,  # n filter in first layer
    # 'residual_conv': [
    #     {'num_filter': 128, 'filter_width': 8, 'dilation': 1},
    #     {'num_filter': 128, 'filter_width': 8, 'dilation': 2},
    #     {'num_filter': 128, 'filter_width': 8, 'dilation': 4},
    #     {'num_filter': 128, 'filter_width': 8, 'dilation': 8},
    #     {'num_filter': 128, 'filter_width': 8, 'dilation': 16},
    #     {'num_filter': 128, 'filter_width': 8, 'dilation': 32},
    # ],
    # # 'n_hidden_units': [50, 10, 3],  # "fully connected" (exclude temporal dimension) layers, 3 is number of output
    #
    #
    # 'n_repeat_in_residual_unit': 2,
    # 'skip_conn_every_n': 4,

    'dense_conv': [
        {'num_filter': 32, 'filter_width': 8, 'dilation': 1},
        {'num_filter': 32, 'filter_width': 8, 'dilation': 2},
        {'num_filter': 32, 'filter_width': 8, 'dilation': 4},
        {'num_filter': 64, 'filter_width': 8, 'dilation': 8},
        {'num_filter': 64, 'filter_width': 8, 'dilation': 16},
        {'num_filter': 64, 'filter_width': 8, 'dilation': 32},
    ],

    'example_length': 500,
    'batch_size': 10,
    'learning_rate': 0.0001,
    'residual': True,
    'skipconn': True,
    'gated': False,

    'num_epoch': 200,
    'num_batch_for_validation': 200,  # use more mini batched so it's more robust
    'es_patience': 10,

    # final models
    'model_dir': 'model/',

}
