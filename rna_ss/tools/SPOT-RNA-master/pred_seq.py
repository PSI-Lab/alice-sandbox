import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
from utils import get_data, _bytes_feature, _int64_feature, _float_feature, output_mask, multiplets_free_bp
import time
start = time.time()


def prob_to_secondary_structure(ensemble_outputs, label_mask, seq, name, args):
    #save_result_path = 'outputs'
    Threshold = 0.335
    test_output = ensemble_outputs
    mask = output_mask(seq)
    inds = np.where(label_mask == 1)
    y_pred = np.zeros(label_mask.shape)
    for i in range(test_output.shape[0]):
        y_pred[inds[0][i], inds[1][i]] = test_output[i]
    y_pred = np.multiply(y_pred, mask)

    tri_inds = np.triu_indices(y_pred.shape[0], k=1)

    out_pred = y_pred[tri_inds]
    outputs = out_pred[:, None]
    seq_pairs = [[tri_inds[0][j], tri_inds[1][j], ''.join([seq[tri_inds[0][j]], seq[tri_inds[1][j]]])] for j in
                 range(tri_inds[0].shape[0])]

    outputs_T = np.greater_equal(outputs, Threshold)
    pred_pairs = [i for I, i in enumerate(seq_pairs) if outputs_T[I]]
    pred_pairs = [i[:2] for i in pred_pairs]
    pred_pairs, save_multiplets = multiplets_free_bp(pred_pairs, y_pred)

    return pred_pairs


def create_tfr_files_new(seq_dict):
    print('\nPreparing tfr records file for SPOT-RNA:')
    path_tfrecords = os.path.join('input_tfr_files', "test_data" + ".tfrecords")
    # with open(all_seq) as file:
    #     input_data = [line.strip() for line in file.read().splitlines() if line.strip()]
    #
    # count = int(len(input_data) / 2)
    #
    # ids = [input_data[2 * i][1:].strip() for i in range(count)]

    with tf.io.TFRecordWriter(path_tfrecords) as writer:
        # for i in tqdm(range(len(ids))):
        for name, sequence in seq_dict.items():
            # name = input_data[2 * i].replace(">", "")
            # sequence = input_data[2 * i + 1].replace(" ", "").replace("T", "U")
            # print(sequence[-1])

            # print(len(sequence), name)
            seq_len, feature, zero_mask, label_mask, true_label = get_data(sequence)

            example = tf.train.Example(features=tf.train.Features(feature={
                'rna_name': _bytes_feature(name),
                'seq_len': _int64_feature(seq_len),
                'feature': _float_feature(feature),
                'zero_mask': _float_feature(zero_mask),
                'label_mask': _float_feature(label_mask),
                'true_label': _float_feature(true_label)}))

            writer.write(example.SerializeToString())

    writer.close()


argparser = argparse.ArgumentParser()
argparser.add_argument(
    '--in_file', type=str,
    help='path to input csv file (with column "seq")'
)
argparser.add_argument(
    '--out_file', type=str,
    help='path to output csv file'
)
argparser.add_argument(
    '--format', type=str,
    help='input format, csv or pkl'
)
args = argparser.parse_args()

in_format = args.format
if in_format == 'csv':
    df_in = pd.read_csv(args.in_file)
elif in_format == 'pkl':
    df_in = pd.read_pickle(args.in_file)
else:
    raise ValueError(in_format)
# add an arbitrary unique ID since it'll get used as part of their inference pipeline O_O
# they also use byte encoding, so has to be converted to str
df_in['tmp_id'] = [str(x) for x in range(len(df_in))]
seq_dict = {row['tmp_id']: row['seq'] for _, row in df_in.iterrows()}
# sequences = df_in['seq'].tolist()

create_tfr_files_new(seq_dict)

# os.environ["CUDA_VISIBLE_DEVICES"]= str(args.gpu)
NUM_MODELS = 5

test_loc = ["input_tfr_files/test_data.tfrecords"]

outputs = {}
mask = {}


def sigmoid(x):
    return 1 / (1 + np.exp(-np.array(x, dtype=np.float128)))


for MODEL in range(NUM_MODELS):
    config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(config=session_conf)
    print('\nPredicting for SPOT-RNA model ' + str(MODEL))
    with tf.compat.v1.Session(config=config) as sess:
        saver = tf.compat.v1.train.import_meta_graph('SPOT-RNA-models' + '/model' + str(MODEL) + '.meta')
        saver.restore(sess, 'SPOT-RNA-models' + '/model' + str(MODEL))
        graph = tf.compat.v1.get_default_graph()
        init_test = graph.get_operation_by_name('make_initializer_2')
        tmp_out = graph.get_tensor_by_name('output_FC/fully_connected/BiasAdd:0')
        name_tensor = graph.get_tensor_by_name('tensors_2/component_0:0')
        RNA_name = graph.get_tensor_by_name('IteratorGetNext:0')
        label_mask = graph.get_tensor_by_name('IteratorGetNext:4')
        sess.run(init_test, feed_dict={name_tensor: test_loc})

        pbar = tqdm(total=len(seq_dict))
        while True:
            try:
                out = sess.run([tmp_out, RNA_name, label_mask], feed_dict={'dropout:0': 1})
                out[1] = out[1].decode()
                mask[out[1]] = out[2]

                if MODEL == 0:
                    outputs[out[1]] = [sigmoid(out[0])]
                else:
                    outputs[out[1]].append(sigmoid(out[0]))
                # print('RNA name: %s'%(out[1]))
                pbar.update(1)
            except:
                break
        pbar.close()
    tf.compat.v1.reset_default_graph()

RNA_ids = [i for i in list(outputs.keys())]
ensemble_outputs = {}

df_out = []
for i in RNA_ids:
    ensemble_outputs[i] = np.mean(outputs[i],0)
    pred_pairs = prob_to_secondary_structure(ensemble_outputs[i], mask[i], seq_dict[i], i, args)
    df_out.append({'seq': seq_dict[i],
                   'pred_idx': [[x[0] for x in pred_pairs],
                                [x[1] for x in pred_pairs]]})

df_out = pd.DataFrame(df_out)
# TODO need to pickle since output contains list
df_out.to_pickle(args.out_file, protocol=2)
