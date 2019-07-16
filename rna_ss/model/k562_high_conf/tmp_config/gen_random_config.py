import sys
import yaml
import numpy as np

config_ref_file = sys.argv[1]
n = int(sys.argv[2])

with open(config_ref_file, 'r') as fi:
    config_ref = yaml.load(fi)

for i in range(n):
    # dense_conv
    n_layer = np.random.randint(1, 10)
    filter_width = np.random.choice([4, 8, 16, 32])

    num_filter_base = np.random.choice([8, 16, 32, 64])
    if np.random.randint(3) == 0:
        num_filters = [num_filter_base for _ in range(n_layer)]
    elif np.random.randint(3) == 1:
        num_filters = [num_filter_base * (k + 1) for k in range(n_layer)]
    else:
        num_filters = [num_filter_base * (k // 2 + 1) for k in range(n_layer)]

    if np.random.randint(3) == 0:
        dilations = [2 ** (k + 1) for k in range(n_layer)]
    else:
        dilations = [2 ** (k // 2 + 1) for k in range(n_layer)]

    dense_conv = [{"dilation": int(d), "filter_width": int(filter_width), "num_filter": int(f)} for d, f in
                  zip(dilations, num_filters)]

    # lr
    learning_rate = float(np.random.choice([0.01, 0.001, 0.005, 0.0001]))

    # update config
    config_new = config_ref.copy()
    config_new['dense_conv'] = dense_conv
    config_new['learning_rate'] = learning_rate

    # output
    with open('config_{}.yml'.format(i), 'w') as fo:
        yaml.dump(config_new, fo)
