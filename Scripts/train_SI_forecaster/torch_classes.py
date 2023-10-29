import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTM_ED(torch.nn.Module):
    def __init__(self, input_size_e, hidden_size_lstm, input_size_d,input_size_past_t,input_size_fut_t,output_dim,dev):
        super(LSTM_ED, self).__init__()
        self.input_size_e = input_size_e  # input size
        self.input_size_d = input_size_d  # input size

        self.hidden_size_lstm = hidden_size_lstm  # hidden state

        self.output_dim = output_dim

        self.num_layers_e = 1
        self.num_layers_d = 1

        self.dev = dev

        #self.nn_past = torch.nn.Linear(in_features = input_size_past_t, output_features = hidden_size_lstm)
        #self.nn_fut = torch.nn.Linear(in_features = input_size_fut_t)

        self.lstm_e = torch.nn.LSTM(input_size=input_size_e, hidden_size=hidden_size_lstm, num_layers=1,
                                    batch_first=True,bidirectional=False).to(dev)  # Encoder
        self.lstm_d = torch.nn.LSTM(input_size=input_size_d, hidden_size=hidden_size_lstm, num_layers=1,
                                    batch_first=True,bidirectional=False).to(dev)  # Decoder
        self.fc = torch.nn.Linear(hidden_size_lstm, output_dim).to(dev) # fully connected 1

    def forward(self, list_data,dev_type='NA'):
        x_e = list_data[0]
        x_d = list_data[1]

        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        h_0 = Variable(torch.zeros(self.num_layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # internal state
        # Propagate input through LSTM
        output_e, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0))  # lstm with input, hidden, and internal state

        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_e, c_e))
        out = torch.squeeze(self.fc(output_d))  # Final Output
        return out

class Loss_pinball(torch.nn.Module):

    def __init__(self,list_quantiles,dev):
        super(Loss_pinball,self).__init__()

        self.dev=dev
        self.list_quantiles = list_quantiles
        self.n_quantiles = len(list_quantiles)
        self.quantile_tensor = torch.from_numpy(np.asarray(list_quantiles)).to(dev)

    def forward(self,labels,fc_SI_quant):
        actual_SI = labels[0]
        diff = actual_SI - fc_SI_quant

        mask_pos = diff>=0

        diff_pos = torch.mul(mask_pos,diff)
        diff_neg = torch.mul(~mask_pos,diff)


        loss = torch.sum(torch.mul(diff_pos,self.quantile_tensor) - torch.mul(diff_neg,1-self.quantile_tensor))

        return loss

#Bi-attention

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size_ctxt, input_size_temp, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding_ctxt = nn.Embedding(input_size_ctxt, hidden_size)
        self.embedding_temp = nn.Embedding(input_size_temp, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size)

    def forward(self, past_ctxt, past_temp):
        embedded_ctxt = self.embedding_ctxt(past_ctxt)
        embedded_temp = self.embedding_temp(past_temp)
        combined = torch.cat((embedded_ctxt, embedded_temp), 2)
        output, hidden = self.gru(combined)
        return output, hidden


class BiAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BiAttention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(3 * hidden_size, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        attn_energies = torch.zeros(encoder_outputs.size(0), encoder_outputs.size(1)).to(hidden.device)

        for i in range(encoder_outputs.size(1)):
            attn_energies[:, i] = self.score(hidden, encoder_outputs[:, i])

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        combined = torch.cat((hidden, encoder_output), 2)
        energy = self.attn(combined)
        return torch.sum(self.v * torch.tanh(energy), dim=2)


class DecoderRNN(nn.Module):
    def __init__(self, input_size_ctxt, input_size_temp, output_size, hidden_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding_ctxt = nn.Embedding(input_size_ctxt, hidden_size)
        self.embedding_temp = nn.Embedding(input_size_temp, hidden_size)
        self.gru = nn.GRU(hidden_size * 3, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.biattention = BiAttention(hidden_size)

    def forward(self, fut_ctxt, fut_temp, encoder_outputs):
        embedded_ctxt = self.embedding_ctxt(fut_ctxt)
        embedded_temp = self.embedding_temp(fut_temp)

        attn_weights = self.biattention(embedded_ctxt, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))

        rnn_input = torch.cat((embedded_temp, context), 2)
        output, hidden = self.gru(rnn_input)

        output = self.out(output.squeeze(0))
        return output, hidden


# Define your input sizes
input_size_past_ctxt = ...  # Define the appropriate size
input_size_past_temp = ...  # Define the appropriate size
input_size_fut_ctxt = ...  # Define the appropriate size
input_size_fut_temp = ...  # Define the appropriate size
hidden_size = ...  # Define the hidden size
output_size = ...  # Define the output size

# Encoder and Decoder initialization
encoder = EncoderRNN(input_size_past_ctxt, input_size_past_temp, hidden_size)
decoder = DecoderRNN(input_size_fut_ctxt, input_size_fut_temp, output_size, hidden_size)

# Training loop (a basic example)
for i in range(epochs):
    encoder_outputs, encoder_hidden = encoder(past_ctxt, past_temp)
    decoder_output, decoder_hidden = decoder(fut_ctxt, fut_temp, encoder_outputs)
    # Compute loss, backpropagate, update weights, etc.
