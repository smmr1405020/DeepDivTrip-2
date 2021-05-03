import numpy as np
import lstm_model
import torch
import copy
import random
import pprint

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

    input_seq_sd = [input_seq[0], input_seq[-1]]
    input_seq_sd = torch.LongTensor(input_seq_sd).to(device)

    input_seq_batch_f = [input_seq_f]
    input_seq_batch_f = np.array(input_seq_batch_f)
    input_seq_batch_f = torch.LongTensor(input_seq_batch_f).to(device)

    model_fwd = lstm_model.get_forward_lstm_model(load_from_file=True)
    output_prb_f = torch.softmax(model_fwd(input_seq_batch_f, input_seq_length, input_seq_sd), dim=-1)
    output_prb_f = output_prb_f.cpu().detach().numpy()

    input_seq_batch_b = [input_seq_b]
    input_seq_batch_b = np.array(input_seq_batch_b)
    input_seq_batch_b = torch.LongTensor(input_seq_batch_b).to(device)

    model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
    output_prb_b = torch.softmax(model_bwd(input_seq_batch_b, input_seq_length, input_seq_sd), dim=-1)
    output_prb_b = output_prb_b.cpu().detach().numpy()

    return output_prb_f, output_prb_b


def get_model_probs_f(input_seq):
    for i in range(len(input_seq)):
        if input_seq[i] is None:
            input_seq[i] = 0

    input_seq_f = list(input_seq)

    input_seq_length = [len(input_seq)]
    input_seq_length = torch.LongTensor(input_seq_length).to(device)

    input_seq_sd = [input_seq[0], input_seq[-1]]
    input_seq_sd = torch.LongTensor(input_seq_sd).to(device)

    input_seq_batch_f = [input_seq_f]
    input_seq_batch_f = np.array(input_seq_batch_f)
    input_seq_batch_f = torch.LongTensor(input_seq_batch_f).to(device)

    model_fwd = lstm_model.get_forward_lstm_model(load_from_file=True)
    output_prb_f = torch.softmax(model_fwd(input_seq_batch_f, input_seq_length, input_seq_sd), dim=-1)
    output_prb_f = output_prb_f.cpu().detach().numpy()

    return output_prb_f


def get_model_probs_b(input_seq):
    for i in range(len(input_seq)):
        if input_seq[i] is None:
            input_seq[i] = 0

    input_seq_b = list(reversed(list(input_seq)))

    input_seq_length = [len(input_seq)]
    input_seq_length = torch.LongTensor(input_seq_length).to(device)

    input_seq_sd = [input_seq[0], input_seq[-1]]
    input_seq_sd = torch.LongTensor(input_seq_sd).to(device)

    input_seq_batch_b = [input_seq_b]
    input_seq_batch_b = np.array(input_seq_batch_b)
    input_seq_batch_b = torch.LongTensor(input_seq_batch_b).to(device)

    model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
    output_prb_b = torch.softmax(model_bwd(input_seq_batch_b, input_seq_length, input_seq_sd), dim=-1)
    output_prb_b = output_prb_b.cpu().detach().numpy()

    return output_prb_b


def get_prob_in_idx(input_seq, I):
    S = len(list(input_seq))
    output_prb_f, output_prb_b = get_model_probs(input_seq)

    final_op = np.zeros((output_prb_f.shape[1]))

    for i in range(len(final_op)):
        final_op[i] = (I * output_prb_f[I - 1][i] + (S - I - 1) * output_prb_b[S - I - 2][i]) / (S - 1)

    candidates_replacement = (np.argsort(-final_op))

    return final_op, candidates_replacement


def get_prob_in_idx_2(input_seq, I):
    S = len(list(input_seq))

    if I > S / 2:
        output_prb_f = get_model_probs_f(input_seq)
        final_op = output_prb_f[I - 1]
    else:
        output_prb_b = get_model_probs_b(input_seq)
        final_op = output_prb_b[S - I - 2]

    return final_op


def get_traj_perplexity(input_seq):
    if len(input_seq) == 0:
        return 1000

    output_prb_f = get_model_probs_f(input_seq)
    fwd_perplexity = 0

    for i in range(1, len(input_seq)):
        prob_idx = output_prb_f[i - 1][input_seq[i]]
        if prob_idx != 0:
            fwd_perplexity += np.log(prob_idx)
        else:
            fwd_perplexity += -100

    fwd_perplexity = (-1.0) * fwd_perplexity

    return fwd_perplexity


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

        if (N_min <= len(min_perp_traj) <= N_max) and (min_perp > ans_traj_perp or ans_traj is None):
            ans_traj_perp = min_perp
            ans_traj = min_perp_traj

        current_traj = min_perp_traj

    return ans_traj
