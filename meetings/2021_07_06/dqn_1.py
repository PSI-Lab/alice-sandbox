"""
adopted from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
import argparse
import os
import logging
import numpy as np
import pandas as pd
import random
from pprint import pprint
from itertools import count
from collections import namedtuple, deque
from sklearn.metrics import f1_score
from utils.util_global_struct import process_bb_old_to_new
from utils.rna_ss_utils import arr2db, one_idx2arr, compute_fe
from utils.inference_s2 import Predictor, process_row_bb_combo, stem_bbs2arr
from utils_s2_tree_search import bb_conflict
from utils.misc import add_column
import torch
import torch.nn as nn
import torch.nn.functional as F


BoundingBox = namedtuple("BoundingBox", ['bb_x', 'bb_y', 'siz_x', 'siz_y'])


DataExample = namedtuple('DataExample',
                        ('seq', 'seq_arr', 'bbs', 'bb_arrs', 'bb_conflict'))


Transition = namedtuple('Transition',
                        ('example_id', 'bb_id_inc', 'bbs_inc_arr', 'valid_bb_ids', 'bb_id_next', 'reward'))



def find_valid_bb_ids(bb_id_inc, bb_id_next, valid_bb_ids, bb_conflict):
    # bb_id_inc: list of IDs included
    # bb_id_next: ID of bb to be included next
    # valid_bb_ids: list of IDs, current list of valid bbs (contains bb_id_next)
    # bb_conflict: NxN binary matrix with 1 indicating conflit

    # check
    assert bb_id_next in valid_bb_ids

    # we're removing invalid IDs from valid_bb_ids due to the inclusion of bb_id_next
    id_conflict = np.where(bb_conflict[bb_id_next, :])[0]
    new_valid_bb_ids = [i for i in valid_bb_ids if i not in id_conflict]

    return new_valid_bb_ids


class AllDataExamples(object):

    def __init__(self, df):
        self.data = dict()
        self.single_seq_enc = SingleSeqEncoder()

        # reindex to make sure df idx is sequential
        df = df.reset_index(drop=True)

        for data_idx, row in df.iterrows():
            seq = row.seq

            if data_idx % 100 == 0:
                print(data_idx)

            # bbs
            df_stem = pd.DataFrame(row.pred_stem_bb)
            # we use df index, make sure it's contiguous
            assert df_stem.iloc[-1].name == len(df_stem) - 1
            bbs = {}
            for idx, r in df_stem.iterrows():
                bbs[idx] = BoundingBox(bb_x=r['bb_x'],
                                       bb_y=r['bb_y'],
                                       siz_x=r['siz_x'],
                                       siz_y=r['siz_y'])

            # list of arr
            bb_arrs = {bb_id: stem_bbs2arr([bb], len(seq)) for bb_id, bb in bbs.items()}
            # bb conflict arr
            bb_conflict_arr = np.zeros((len(bbs), len(bbs)))
            for i in range(len(bbs)):
                for j in range(i, len(bbs)):  # only need to go through half
                    if bb_conflict(bbs[i], bbs[j]):
                        bb_conflict_arr[i, j] = 1
                        bb_conflict_arr[j, i] = 1
            # seq
            seq = row.seq
            seq_arr = self.single_seq_enc.encode_single(seq)

            # add to data
            self.data[data_idx] = DataExample(seq=seq,
                                              seq_arr=seq_arr,
                                              bbs=bbs,
                                              bb_arrs=bb_arrs,
                                              bb_conflict=bb_conflict_arr)


class SingleSeqEncoder(object):
    DNA_ENCODING = np.asarray([[0, 0, 0, 0],
                               [1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

    def __init__(self):
        pass

    def _encode_seq(self, seq):
        seq = seq.upper().replace('A', '1').replace('C', '2').replace('G', '3').replace('T', '4').replace('U',
                                                                                                          '4').replace(
            'N', '0')
        x = np.asarray([int(x) for x in list(seq)])
        x = self.DNA_ENCODING[x.astype('int8')]
        return x

    def tile_and_stack(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 4
        l = x.shape[0]
        x1 = x[:, np.newaxis, :]
        x2 = x[np.newaxis, :, :]
        x1 = np.repeat(x1, l, axis=1)
        x2 = np.repeat(x2, l, axis=0)
        return np.concatenate([x1, x2], axis=2)

    def encode_single(self, seq):
        x = self._encode_seq(seq)
        x = self.tile_and_stack(x)
        return x  # LxLx8


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ValueNetwork(nn.Module):

    def __init__(self, h=60, w=60):
        super(ValueNetwork, self).__init__()
        # FIXME hard-coded for now
        self.conv1 = nn.Conv2d(10, 16, kernel_size=5, stride=2)  # input ch = 10 = 8 + 1 + 1
#         self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.out = nn.Linear(linear_input_size, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
#         x = x.to(device)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return self.out(x.view(x.size(0), -1))


def encode_all_actions(seq_arr, inc_bbs_arr, next_bbs_arrs, n_actions):
    seq_arr = np.tile(seq_arr[np.newaxis, :, :, :], [n_actions, 1, 1, 1])  # kxLxLx8
    inc_bbs_arr = np.tile(inc_bbs_arr[np.newaxis, :, :, np.newaxis], [n_actions, 1, 1, 1])  # kxLxLx8

    # FIXME unify data format
    if len(next_bbs_arrs[0].shape) == 3:
        next_bbs_arrs = [x[np.newaxis, :, :, :] for x in next_bbs_arrs]
    elif len(next_bbs_arrs[0].shape) == 2:
        next_bbs_arrs = [x[np.newaxis, :, :, np.newaxis] for x in next_bbs_arrs]
    else:
        raise ValueError
    next_bbs_arrs = np.concatenate(next_bbs_arrs, axis=0)

    batch_data = np.concatenate([seq_arr, inc_bbs_arr, next_bbs_arrs], axis=3)  # kxLxLx10
    batch_data = torch.from_numpy(batch_data).float()
    return batch_data.permute(0, 3, 1, 2)


# steps_done = 0

def select_action(global_counter, policy_net, seq_arr, inc_bbs_arr, next_bbs_arrs):  # works on a single example
    # seq_arr: LxLx4
    # inc_bbs_arr: LxLx1
    # next_bbs_arrs: list of k items: LxLx1
    
    # TODO check dimensions

    # FIXME hard-coded
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    
    n_actions = len(next_bbs_arrs)
    next_bbs_arrs = [x[np.newaxis, :, :, np.newaxis] for x in next_bbs_arrs]
    next_bbs_arrs = np.concatenate(next_bbs_arrs, axis=0)
    
    # global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * global_counter.ct / EPS_DECAY)
    global_counter.step()
    if sample > eps_threshold:
        with torch.no_grad():
            # predict on all actions as a batch
            # kxLxLx6
            batch_data = encode_all_actions(seq_arr, inc_bbs_arr, next_bbs_arrs, n_actions)
            pred = policy_net(batch_data)  # TODO check dimension kx1?
            # index of action following greedy policy
            idx_action = pred.argmax(0)  # argmax along dim=0 (actions)
            
#             # t.max(1) will return largest column value of each row.
#             # second column on max result is index of where max element was
#             # found, so we pick action with the larger expected reward.
#             return policy_net(state).max(1)[1].view(1, 1)
        return idx_action
    else:
        return np.random.randint(0, n_actions)  # [0, n_actions)
#         return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)



def optimize_model(optimizer, policy_net, target_net, all_data_examples, replay_memory, batch_size):

    # FIXME hard-coded
    # discount factor
    GAMMA = 0.999

    if len(replay_memory) < batch_size:
        return
    transitions = replay_memory.sample(batch_size)  # list of transitions
    
    # Compute Q(s_t, a)
    # running all examples as a single batch, since we only need to evaluate one action per example
    state_action_batch = []
    for transition in transitions:
        example_id = transition.example_id
        data_example = all_data_examples.data[example_id]
        seq_arr = data_example.seq_arr
        bb_arr_next = data_example.bb_arrs[transition.bb_id_next]
        data = np.concatenate([seq_arr, 
                               transition.bbs_inc_arr[:, :, np.newaxis],
                               bb_arr_next[:, :, np.newaxis]], axis=2)  # LxLx10
        data = data[np.newaxis, :, :, :]  # 1xLxLx10
        state_action_batch.append(data)
    state_action_batch = np.concatenate(state_action_batch, axis=0)  # shape: batch x h x w x channel
    state_action_values = policy_net(torch.from_numpy(state_action_batch).float().permute(0, 3, 1, 2))  # torch expects batch x channel x h x w
    
    # Compute V(s_{t+1}) for all next states.
    # next state value: take max over all valid actions at t+2
    # runnin each example as their own batch, since we'll be evaluating all valid actions per example
    # in the future we can combine different examples and make it more efficient
    # also deal with cases where t+1 is the final state, set to 0 otherwise
    expected_state_action_values = torch.zeros(len(transitions))
    for idx_val, transition in enumerate(transitions):
        example_id = transition.example_id
        data_example = all_data_examples.data[example_id]
        seq_arr = data_example.seq_arr
        bp_arr_t1 = transition.bbs_inc_arr + data_example.bb_arrs[transition.bb_id_next]
        # find all valid actions from t1 to t2
        bb_id_inc_t1 = transition.bb_id_inc + [transition.bb_id_next]
        valid_bb_ids = find_valid_bb_ids(transition.bb_id_inc, 
                                         transition.bb_id_next, 
                                         transition.valid_bb_ids, 
                                         data_example.bb_conflict)
        if len(valid_bb_ids) == 0:  # no valid action after t+1, i.e. final state
            pass  # default is 0
        else:
            batch_data = encode_all_actions(seq_arr, 
                                            bp_arr_t1, 
                                            [data_example.bb_arrs[bb_id] for bb_id in valid_bb_ids],
                                           len(valid_bb_ids))
            next_state_action_values = target_net(batch_data)
            next_state_value = next_state_action_values.max()
            expected_state_action_values[idx_val] = (next_state_value * GAMMA) + transition.reward
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)
#     print(state_action_values.shape)
#     print(expected_state_action_values.shape)
    print(loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)  # TODO do we need clamping?
    optimizer.step()


class GlobalCounter():
    def __init__(self):
        self.ct = 0

    def step(self):
        """Increment the counter"""
        self.ct += 1


def main(path_data, num_episodes, lr, batch_size, memory_size):
    # debug dataset
    # df = pd.read_pickle('../2021_06_22/data/data_len60_test_1000_s1_stem_bb_combos_s1s100.pkl.gz')
    df = pd.read_pickle(path_data)
    # build examples
    all_data_examples = AllDataExamples(df)

    # # sequence encoder
    # single_seq_enc = SingleSeqEncoder()

    # initialize value network
    policy_net = ValueNetwork(60, 60)  # FIXME hard-coded seq len
    # make a copy for computing target value in the Bellman update
    # this network will be updated once in a while
    target_net = ValueNetwork(60, 60)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # optimizer and replay memory
    optimizer = torch.optim.RMSprop(policy_net.parameters(), lr)
    replay_memory = ReplayMemory(memory_size)

    # constants
    # batch_size = 4
    # GAMMA = 0.999
    # EPS_START = 0.9
    # EPS_END = 0.05
    # EPS_DECAY = 200
    TARGET_UPDATE = 10  # FIXME hard-coded

    # global counter
    global_counter = GlobalCounter()

    # num_episodes = 10  # debug
    for i_episode in range(num_episodes):
        # get one random example
        example_id = np.random.randint(0, len(all_data_examples.data))
        data_example = all_data_examples.data[example_id]

        # init
        bb_id_inc = []
        inc_bbs_arr = np.zeros((len(data_example.seq), len(data_example.seq)))
        valid_bb_ids = list(data_example.bbs.keys())
        current_neg_fe = 0  # inital neg fe 0

        for t in count():
            # Select and perform an action
            idx_action = select_action(global_counter, policy_net, data_example.seq_arr,
                                   inc_bbs_arr, [data_example.bb_arrs[bb_id] for bb_id in valid_bb_ids])
            next_bb_id = valid_bb_ids[idx_action]

            # backup current state before update, so we can store in memory
            old_bb_id_inc = bb_id_inc
            old_inc_bbs_arr = inc_bbs_arr
            old_valid_bb_ids = valid_bb_ids

            # update
            valid_bb_ids = find_valid_bb_ids(bb_id_inc,
                                     next_bb_id,
                                     valid_bb_ids,
                                     data_example.bb_conflict)
            bb_id_inc.append(next_bb_id)
            inc_bbs_arr += data_example.bb_arrs[next_bb_id]

            # fe & reward
            bp_arr = stem_bbs2arr([data_example.bbs[bb_id] for bb_id in bb_id_inc], len(data_example.seq))
            db_str, result_ambiguous = arr2db(bp_arr)  # TODO check
            new_neg_fe = - compute_fe(data_example.seq, db_str)
            reward = new_neg_fe - current_neg_fe

            # update current fe
            current_neg_fe = new_neg_fe + 0  # TODO copy?

            # Store the transition in memory
            replay_memory.push(example_id,
                        old_bb_id_inc,
                        old_inc_bbs_arr,
                        old_valid_bb_ids,
                        next_bb_id,
                        reward)

    #         _, reward, done, _ = env.step(action.item())
    #         reward = torch.tensor([reward], device=device)

            # Perform one step of the optimization (on the policy network)
            optimize_model(optimizer, policy_net, target_net, all_data_examples, replay_memory, batch_size)

            # check if we're at final state
            if len(valid_bb_ids) == 0:
                break

    #         if done:
    #             episode_durations.append(t + 1)
    #             plot_durations()
    #             break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')


def set_up_logging(path_result):
    # make result dir if non existing
    if not os.path.isdir(path_result):
        os.makedirs(path_result)

    log_format = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    file_logger = logging.FileHandler(os.path.join(path_result, 'run.log'))
    file_logger.setFormatter(log_format)
    root_logger.addHandler(file_logger)
    console_logger = logging.StreamHandler()
    console_logger.setFormatter(log_format)
    root_logger.addHandler(console_logger)
    root_logger.addHandler(console_logger)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        help='Path to training data file, should be in pkl.gz format')
    parser.add_argument('--result', type=str, help='Path to output result')
    parser.add_argument('--episode', type=int, default=10, help='Number of episodes, each episode is one sequence')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Mini batch size')
    parser.add_argument('--memory_size', type=int, default=100, help='replay memory size')

    args = parser.parse_args()
    set_up_logging(args.result)

    main(args.data, args.episode, args.lr, args.batch_size, args.memory_size)


