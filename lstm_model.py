import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import parameters
import data_generator
import graph_embedding
import numpy as np

torch.manual_seed(1234567890)
np.random.seed(1234567890)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TrajPredictor(nn.Module):
    def __init__(self, pretrained_node_embeddings, hidden_size):
        super(TrajPredictor, self).__init__()

        self.embedding = nn.Embedding(len(data_generator.vocab_to_int) - 3,
                                      embedding_dim=pretrained_node_embeddings.shape[1]).from_pretrained(
            pretrained_node_embeddings,
            freeze=True)

        '''

        '''
        self.embedding_dim = pretrained_node_embeddings.shape[1]
        self.vocab_size = len(data_generator.vocab_to_int) - 3
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(input_size=self.embedding_dim,
                               hidden_size=self.hidden_size, num_layers=1)

        self.linear_inp_size = self.hidden_size
        self.fc1 = nn.Linear(self.linear_inp_size, self.vocab_size)

    def forward(self, seq, seq_lengths):
        embeds = self.embedding(seq)
        lstm_input = nn.utils.rnn.pack_padded_sequence(embeds, seq_lengths.cpu(), batch_first=False)
        output, (hidden, _) = self.encoder(lstm_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        out_sq = self.fc1(output)
        out_sq = torch.squeeze(out_sq)

        return out_sq


def loss_fn(pred, target):
    return torch.nn.CrossEntropyLoss()(torch.transpose(pred, 1, 2), target)


def print_all(model, backward_model):
    if not backward_model:
        tr_s, _ = data_generator.get_trajectory_dataset()
    else:
        _, tr_s = data_generator.get_trajectory_dataset()

    dataset_trajectory = tr_s

    for i in range(dataset_trajectory.no_training_batches(parameters.lstm_model_batch_size)):
        inputs, seq_lengths, targets = dataset_trajectory(i, parameters.lstm_model_batch_size)
        inputs = torch.LongTensor(inputs).to(device)
        seq_lengths = torch.LongTensor(seq_lengths).to(device)

        targets = torch.LongTensor(targets).to(device)
        output = torch.softmax(model(inputs, seq_lengths), dim=2)

        op = torch.transpose(output, 0, 1)
        op = op.cpu().detach().numpy()
        op = np.argmax(op, axis=2)

        tgt = torch.transpose(targets, 0, 1)
        tgt = tgt.cpu().detach().numpy()

        print("PRED:")
        print(op)
        print("GT:")
        print(tgt)
        print("\n")


def train(model, optimizer, loss_fn_, epochs=100, backward_model=False):
    train_loss_min = 100000.0
    validation_loss_min = 100000.0
    for epoch in range(epochs):
        training_loss = 0.0

        if not backward_model:
            tr_s, _ = data_generator.get_trajectory_dataset()
        else:
            _, tr_s = data_generator.get_trajectory_dataset()

        model.train()

        no_batches = 0
        for i in range(int(0.8 * tr_s.no_training_batches(parameters.lstm_model_batch_size))):
            no_batches += 1
            inputs, seq_lengths, targets = tr_s(i, parameters.lstm_model_batch_size)
            inputs = torch.LongTensor(inputs).to(device)
            seq_lengths = torch.LongTensor(seq_lengths).to(device)
            targets = torch.LongTensor(targets).to(device)
            optimizer.zero_grad()
            output = model(inputs, seq_lengths)
            loss = loss_fn_(output, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.2)
            optimizer.step()
            training_loss += loss.data.item()

        training_loss /= no_batches

        validation_loss = 0.0
        model.eval()

        no_batches = 0
        for i in range(int(0.8 * tr_s.no_training_batches(parameters.lstm_model_batch_size))
                , tr_s.no_training_batches(parameters.lstm_model_batch_size)):
            no_batches += 1
            inputs, seq_lengths, targets = tr_s(i, parameters.lstm_model_batch_size)
            inputs = torch.LongTensor(inputs).to(device)
            seq_lengths = torch.LongTensor(seq_lengths).to(device)
            targets = torch.LongTensor(targets).to(device)

            output = model(inputs, seq_lengths)
            loss = loss_fn_(output, targets)

            validation_loss += loss.data.item()

        validation_loss /= no_batches

        if validation_loss < validation_loss_min:
            validation_loss_min = validation_loss
            if not backward_model:
                torch.save(model.state_dict(),
                           os.path.join("model_files", "LSTM_net_1_f_" + data_generator.embedding_name))
            else:
                torch.save(model.state_dict(),
                           os.path.join("model_files", "LSTM_net_1_b_" + data_generator.embedding_name))

        if epoch % 1 == 0 or epoch == epochs - 1:
            print('Epoch: {}, Training Loss: {:.3f}, Validation Loss: {:.3f}'.format
                  (epoch, training_loss, validation_loss))


def get_forward_lstm_model(load_from_file=True):
    pretrained_embeddings = graph_embedding.get_POI_embeddings(load_from_file=True)
    pretrained_embeddings = torch.FloatTensor(pretrained_embeddings).to(device)
    if not load_from_file:
        trajpredictor_forward = TrajPredictor(pretrained_embeddings, parameters.lstm_model_hidden_size).to(device)
        optimizer_forward = optim.Adam(trajpredictor_forward.parameters(), lr=0.001)
        print("\nForward")
        train(trajpredictor_forward, optimizer_forward, loss_fn, epochs=80)
        print("\n")
    forward_lstm_model = TrajPredictor(pretrained_embeddings, parameters.lstm_model_hidden_size).to(device)
    fwd_model_state_dict = torch.load(os.path.join("model_files", "LSTM_net_1_f_" + data_generator.embedding_name))
    forward_lstm_model.load_state_dict(fwd_model_state_dict)
    # print_all(forward_lstm_model, backward_model=False)
    return forward_lstm_model


# get_forward_lstm_model(False)


def get_backward_lstm_model(load_from_file=True):
    pretrained_embeddings = graph_embedding.get_POI_embeddings(load_from_file=True)
    pretrained_embeddings = torch.FloatTensor(pretrained_embeddings).to(device)
    if not load_from_file:
        trajpredictor_backward = TrajPredictor(pretrained_embeddings, parameters.lstm_model_hidden_size).to(device)
        optimizer_backward = optim.Adam(trajpredictor_backward.parameters(), lr=0.001)
        print("\nBackward")
        train(trajpredictor_backward, optimizer_backward, loss_fn, epochs=80, backward_model=True)
        print("\n")
    backward_lstm_model = TrajPredictor(pretrained_embeddings, parameters.lstm_model_hidden_size).to(device)
    bwd_model_state_dict = torch.load(os.path.join("model_files", "LSTM_net_1_b_" + data_generator.embedding_name))
    backward_lstm_model.load_state_dict(bwd_model_state_dict)
    # print_all(backward_lstm_model, backward_model=True)
    return backward_lstm_model

# get_backward_lstm_model(False)
