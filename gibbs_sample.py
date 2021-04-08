import numpy as np
import lstm_model
import torch
import copy
import random
import data_generator
import pprint
import matplotlib.pyplot as plt

np.random.seed(1234567890)
torch.manual_seed(1234567890)
random.seed(1234567890)

pp = pprint.PrettyPrinter(indent=4, width=180)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model_probs(input_seq):
    for i in range(len(input_seq)):
        if input_seq[i] is None:
            input_seq[i] = 0

    input_seq_f = list(input_seq)
    input_seq_b = list(reversed(list(input_seq)))

    input_seq_length = [len(input_seq)]
    input_seq_length = torch.LongTensor(input_seq_length).to(device)

    input_seq_batch_f = [input_seq_f]
    input_seq_batch_f = np.array(input_seq_batch_f)
    input_seq_batch_f = np.transpose(input_seq_batch_f)
    input_seq_batch_f = torch.LongTensor(input_seq_batch_f).to(device)

    model_fwd = lstm_model.get_forward_lstm_model(load_from_file=True)
    output_prb_f = torch.softmax(model_fwd(input_seq_batch_f, input_seq_length), dim=-1)
    output_prb_f = output_prb_f.cpu().detach().numpy()

    input_seq_batch_b = [input_seq_b]
    input_seq_batch_b = np.array(input_seq_batch_b)
    input_seq_batch_b = np.transpose(input_seq_batch_b)
    input_seq_batch_b = torch.LongTensor(input_seq_batch_b).to(device)

    model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
    output_prb_b = torch.softmax(model_bwd(input_seq_batch_b, input_seq_length), dim=-1)
    output_prb_b = output_prb_b.cpu().detach().numpy()

    return output_prb_f, output_prb_b


def get_prob_in_idx(input_seq, I):
    S = len(list(input_seq))
    output_prb_f, output_prb_b = get_model_probs(input_seq)

    final_op = np.zeros((output_prb_f.shape[1]))

    for i in range(len(final_op)):
        final_op[i] = (I * output_prb_f[I - 1][i] + (S - I - 1) * output_prb_b[S - I - 2][i]) / (S - 1)

    candidates_replacement = (np.argsort(-final_op))

    return final_op, candidates_replacement


def get_traj_perplexity(input_seq):
    if len(input_seq) == 0:
        return 1000, 0.0

    output_prb_f, output_prb_b = get_model_probs(input_seq)
    fwd_perplexity = 0

    raw_prob_fwd = 1
    raw_prob_bwd = 1

    for i in range(1, len(input_seq)):
        prob_idx = output_prb_f[i - 1][input_seq[i]]
        raw_prob_fwd *= prob_idx
        if prob_idx != 0:
            fwd_perplexity += np.log(prob_idx)
        else:
            fwd_perplexity += -100

    fwd_perplexity = (1 / len(input_seq) ** 0.75) * (-1.0) * fwd_perplexity

    bwd_perplexity = 0
    input_seq_b = list(reversed(list(input_seq)))
    for i in range(1, len(input_seq_b)):
        prob_idx = output_prb_b[i - 1][input_seq_b[i]]
        raw_prob_bwd *= prob_idx
        if prob_idx != 0:
            bwd_perplexity += np.log(prob_idx)
        else:
            bwd_perplexity += -100

    bwd_perplexity = (1 / len(input_seq) ** 0.75) * (-1.0) * bwd_perplexity

    perplexity = 0.5 * (fwd_perplexity + bwd_perplexity)

    return perplexity


def normalize(x):
    tem = copy.copy(x)
    tem = np.array(tem)
    if np.max(tem) == 0:
        return tem
    return list(tem / np.sum(tem))


def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))


def choose_action(c):
    r = np.random.random()
    c = np.array(c)
    for i in range(1, len(c)):
        c[i] = c[i] + c[i - 1]
    for i in range(len(c)):
        if c[i] >= r:
            return i


def sampling_algo(barebone_seq, N_max=10, N_min=3):
    prominent_poi = barebone_seq[1]

    ans_traj = None
    ans_traj_perp = 1000.0

    if N_min <= len(barebone_seq) <= N_max:
        ans_traj = barebone_seq.copy()
        ans_traj_perp = get_traj_perplexity(barebone_seq)

    current_traj = barebone_seq.copy()

    def insert_poi(input_seq, idx):

        original_input_seq = input_seq.copy()

        input_seq = list(input_seq)
        input_seq.insert(idx, 0)

        final_prob, candidate_poi = get_prob_in_idx(input_seq, idx)

        for poi in candidate_poi:
            if poi not in original_input_seq:
                input_seq[idx] = poi
                break

        return input_seq

    for l in range(len(barebone_seq), N_max + 1):
        min_perp_traj = current_traj
        min_perp = 100.0
        for i in range(1, len(current_traj)):
            current_traj_candidate = insert_poi(current_traj.copy(), i)
            current_traj_perp = get_traj_perplexity(current_traj_candidate)

            if i == 1 or current_traj_perp < min_perp:
                min_perp = current_traj_perp
                min_perp_traj = current_traj_candidate

        if (N_min <= len(min_perp_traj) <= N_max) and (min_perp < ans_traj_perp or ans_traj is None):
            ans_traj_perp = min_perp
            ans_traj = min_perp_traj

        current_traj = min_perp_traj

    return ans_traj

# sampling_algo([6, 11, 0], 10)
#
#
# def get_diverse_traj(poi_start, poi_end, no_diverse_traj):
#     barebone_traj = [poi_start, 0, poi_end]
#     final_prob, candidates = get_prob_in_idx(barebone_traj, 1)
#     print(candidates)
#     print(final_prob)
#
#     plt.plot(np.arange(final_prob.shape[0]), sorted(final_prob, reverse=True), color='red', label='probs')
#     plt.show()
#
#     return
#
#
#
#
# get_diverse_traj(6, 0, 3)

# inflated_tr_ds = [[10, 12, 1, 0, 3],
#                   [10, 4, 12, 5, 9, 1, 0, 3],
#                   [10, 12, 1, 2, 7, 0, 9, 3],
#                   [10, 12, 1, 2, 15, 19, 0, 3],
#                   [10, 4, 3],
#                   [10, 7, 4, 15, 3],
#                   [10, 5, 4, 15, 3],
#                   [10, 4, 12, 3],
#                   [10, 4, 15, 3],
#                   [10, 4, 15, 9, 0, 3],
#                   [10, 4, 6, 8, 15, 3],
#                   [10, 1, 4, 2, 15, 3],
#                   [10, 13, 4, 16, 3],
#                   [10, 5, 4, 9, 3],
#                   [10, 5, 4, 2, 3]]
#
# inflated_tr_ds_perp = []
#
# for i in range(len(inflated_tr_ds)):
#     perp = get_traj_perplexity(inflated_tr_ds[i])
#     inflated_tr_ds_perp.append([inflated_tr_ds[i], perp])
#
# print(inflated_tr_ds_perp)
