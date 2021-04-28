import numpy as np
import lstm_model
import torch
import random

np.random.seed(1234567890)
torch.manual_seed(1234567890)
random.seed(1234567890)

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


def get_backward_sequence(middle_poi, start_poi, max_length):
    input_seq = [middle_poi]
    got_startp = False
    startp_prob = [0.0]

    while input_seq[-1] != start_poi and len(input_seq) < max_length:
        input_seq_batch = [input_seq]
        input_seq_batch_len = [len(input_seq)]

        input_seq_batch_n = np.transpose(np.array(input_seq_batch).astype(int))
        input_seq_batch_len_n = np.array(input_seq_batch_len).astype(int)

        input_seq_batch_t = torch.LongTensor(input_seq_batch_n).to(device)
        input_seq_batch_len_t = torch.LongTensor(input_seq_batch_len_n).to(device)

        model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
        output_prb = torch.softmax(model_bwd(input_seq_batch_t, input_seq_batch_len_t), dim=-1)
        output_prb = output_prb.cpu().detach().numpy()

        if len(input_seq) == 1:
            selected_poi = np.argmax(output_prb, axis=-1)
            startp_prob.append(output_prb[start_poi])
        else:
            selected_poi = np.argmax(output_prb, axis=-1)[-1]
            startp_prob.append(output_prb[-1][start_poi])

        input_seq.append(selected_poi)
        if selected_poi == start_poi:
            got_startp = True

    if got_startp == False:
        startp_position = int(np.argmax(np.array(startp_prob)))
        input_seq[startp_position] = start_poi
        input_seq = input_seq[:startp_position + 1]

    return input_seq


def get_sequence(start_poi, middle_poi, end_poi, max_bwd_seq_len, max_seq_length):
    backward_seq = get_backward_sequence(middle_poi, start_poi, max_bwd_seq_len)

    partial_front_seq = list(reversed(backward_seq))

    input_seq = partial_front_seq
    got_endp = False
    endp_prob = [0.0] * len(partial_front_seq)

    while input_seq[-1] != end_poi and len(input_seq) < max_seq_length:
        input_seq_batch = [input_seq]
        input_seq_batch_len = [len(input_seq)]

        input_seq_batch_n = np.transpose(np.array(input_seq_batch))
        input_seq_batch_len_n = np.array(input_seq_batch_len)

        input_seq_batch_t = torch.LongTensor(input_seq_batch_n).to(device)
        input_seq_batch_len_t = torch.LongTensor(input_seq_batch_len_n).to(device)

        model_bwd = lstm_model.get_backward_lstm_model(load_from_file=True)
        output_prb = torch.softmax(model_bwd(input_seq_batch_t, input_seq_batch_len_t), dim=-1)
        output_prb = output_prb.cpu().detach().numpy()

        if len(input_seq) == 1:
            selected_poi = np.argmax(output_prb, axis=-1)
            endp_prob.append(output_prb[end_poi])
        else:
            selected_poi = np.argmax(output_prb, axis=-1)[-1]
            endp_prob.append(output_prb[-1][end_poi])

        input_seq.append(selected_poi)
        if selected_poi == end_poi:
            got_endp = True

    if not got_endp:
        endp_position = int(np.argmax(np.array(endp_prob)))
        input_seq[endp_position] = end_poi
        input_seq = input_seq[:endp_position + 1]

    return input_seq
