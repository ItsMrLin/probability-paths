from torch import nn
import torch
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout=0.5):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(input_dim, emb_dim)

        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            n_layers,
            dropout=dropout,
            batch_first=True)
        self.rnn.flatten_parameters()
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):

        # src = [src len, batch size]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]
        self.rnn.flatten_parameters()
        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [batch size, src len, hid dim * n directions]
        # hidden = [batch size, n layers * n directions, hid dim]
        # cell = [batch size, n layers * n directions, hid dim]

        # outputs are always from the top hidden layer

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_dim, emb_dim)

        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            n_layers,
            dropout=dropout,
            batch_first=True)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):

        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        #input = input.unsqueeze(1)

        # input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]
        self.rnn.flatten_parameters()
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output)

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(
            self,
            input_dim=6,
            output_dim=6,
            emb_dim=32,
            hid_dim=128,
            n_layers=2,
            dropout=0.5,
            device="cpu"):
        super().__init__()

        self.encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
        self.device = device

    def forward(self, src, trg_len, teacher_forcing_ratio=0.5):

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75%
        # of the time

        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        preds = torch.zeros(batch_size, trg_len, 1).to(self.device).float()

        # last hidden state of the encoder is used as the initial hidden state
        # of the decoder
        hidden, cell = self.encoder(src)

        input = torch.zeros(
            batch_size, 1, trg_vocab_size).to(
            self.device).float()

        for t in range(trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            preds[:, t, :] = torch.sigmoid(output[:, :, -1])

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = output

        return preds.squeeze(-1)

class MQSeq2Seq(nn.Module):
    def __init__(
            self,
            input_dim=6,
            output_dim=6,
            emb_dim=32,
            hid_dim=128,
            n_layers=2,
            dropout=0.5,
            quantiles=None,
            device="cpu"):
        super().__init__()

        self.encoder = Encoder(input_dim, emb_dim, hid_dim, n_layers, dropout)
        self.decoder = Decoder(output_dim, emb_dim, hid_dim, n_layers, dropout)
        self.device = device
        self.quantiles = quantiles
        self.output_dim = output_dim

    def forward(self, src, trg_len, teacher_forcing_ratio=0.5, sampling=False):

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75%
        # of the time

        batch_size = src.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        preds = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device).float()
        paths = []

        # last hidden state of the encoder is used as the initial hidden state
        # of the decoder
        hidden, cell = self.encoder(src)

        input = torch.zeros(
            batch_size, 1, trg_vocab_size).to(
            self.device).float()

        for t in range(trg_len):

            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            N = output.size(0) # output.shape = Batch size x 1 x n_quantiles
            if sampling is True:
                idx = np.searchsorted(self.quantiles, np.random.rand(N)*0.98+0.01)
                chosen = torch.stack([output[i, :, quantile_i] 
                                        for i, quantile_i in enumerate(idx)])
                #print(f"selected shape: {selected.shape}")
                paths.append(torch.sigmoid(chosen))
                chosen_expanded = chosen.unsqueeze(-1).expand_as(output)
                #print(f"output shape: {output.shape}")
                #print(f"chosen shape: {chosen.shape}")
                #print(f"chosen expanded shape: {chosen_expanded.shape}")
                 
                output = (output + chosen_expanded) / 2

            preds[:, t, :] = torch.sigmoid(output.squeeze(1))
            input = output
        if sampling:
            paths = torch.stack(paths, 1)[...,0]
            #print(f"paths shape: {paths.shape}")
            return paths
        else:
            return preds
