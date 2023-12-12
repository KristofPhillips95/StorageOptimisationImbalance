import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy

#Loss functions
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

class Loss_profit(nn.Module):

    def __init__(self):
        super(Loss_profit, self).__init__()

    def forward(self, output, prices):
        schedules = output
        profit = torch.sum(torch.mul(schedules, prices))

        return -profit

class Loss_smoothing(nn.Module):

    def __init__(self,obj):
        super(Loss_smoothing, self).__init__()
        self.obj = obj

    def forward(self, y_hat,labels):

        if self.obj == "profit":
            loss = -torch.sum(torch.mul(y_hat, labels))
        elif self.obj == "mse_sched":
            loss = torch.sum(torch.square(y_hat-labels))

        return loss

""" 
Basic unidirectional encoder-decoder without attention. 
"""
class LSTM_ED(torch.nn.Module):
    def __init__(self, input_size_e,layers_e, hidden_size_lstm, input_size_d,layers_d,output_dim,dev):
        super(LSTM_ED, self).__init__()
        self.input_size_e = input_size_e  # input size
        self.input_size_d = input_size_d  # input size
        self.layers_e = layers_e
        self.layers_d = layers_d

        self.hidden_size_lstm = hidden_size_lstm  # hidden state

        self.output_dim = output_dim


        self.dev = dev

        #self.nn_past = torch.nn.Linear(in_features = input_size_past_t, output_features = hidden_size_lstm)
        #self.nn_fut = torch.nn.Linear(in_features = input_size_fut_t)

        self.lstm_e = torch.nn.LSTM(input_size=input_size_e, hidden_size=hidden_size_lstm, num_layers=layers_e,
                                    batch_first=True,bidirectional=False).to(dev)  # Encoder
        self.lstm_d = torch.nn.LSTM(input_size=input_size_d, hidden_size=hidden_size_lstm, num_layers=layers_d,
                                    batch_first=True,bidirectional=False).to(dev)  # Decoder
        self.fc = torch.nn.Linear(hidden_size_lstm, output_dim).to(dev) # fully connected 1

    def forward(self, list_data,dev_type='NA'):
        x_e = list_data[0]
        x_d = list_data[1]

        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        h_0 = Variable(torch.zeros(self.layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # internal state
        # Propagate input through LSTM
        output_e, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0))  # lstm with input, hidden, and internal state


        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_e, c_e))
        out = torch.squeeze(self.fc(output_d))  # Final Output
        return out

    def set_device(self,dev):
        self.dev = dev
        self.lstm_e.to(dev)
        self.lstm_d.to(dev)
        self.fc.to(dev)


"""
Bi-directional RNNs with attention by chatgpt
"""

class LSTMLayerNorm(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMLayerNorm, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        # Apply Layer Normalization to the output of LSTM
        output = self.layer_norm(output)
        return output, (hidden, cell)

class AttentionModule(torch.nn.Module):
    def __init__(self, hidden_size_lstm,dev):
        super(AttentionModule, self).__init__()
        self.attn = torch.nn.Linear(2 * hidden_size_lstm, hidden_size_lstm).to(dev)
        self.v = torch.nn.Parameter(torch.rand(hidden_size_lstm)).to(dev)

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (batch_size, hidden_size_lstm)
        # encoder_outputs shape: (batch_size, seq_length_e, hidden_size_lstm)

        # Expand hidden to match the sequence length of encoder_outputs
        #hidden_expanded = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        # hidden_expanded shape: (batch_size, seq_length_e, hidden_size_lstm)

        # Calculate energies
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy shape: (batch_size, seq_length_e, hidden_size_lstm)

        energy = energy.transpose(1, 2)  # (batch_size, hidden_size_lstm, seq_length_e)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # (batch_size, 1, hidden_size_lstm)

        # Calculate attention scores
        attn_scores = torch.bmm(v, energy).squeeze(1)  # (batch_size, seq_length_e)
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, seq_length_e)

        return attn_weights


class LSTM_ED_Attention(torch.nn.Module):
    def __init__(self, input_size_e,layers_e, hidden_size_lstm, input_size_d,layers_d,output_dim,bidir_e=True,dev='cuda'):
        super(LSTM_ED_Attention, self).__init__()
        self.input_size_e = input_size_e  # input size
        self.input_size_d = input_size_d  # input size
        self.layers_e = layers_e
        self.layers_d = layers_d
        self.bidir_e = bidir_e

        self.hidden_size_lstm = hidden_size_lstm  # hidden state

        self.output_dim = output_dim


        self.dev = dev

        #self.nn_past = torch.nn.Linear(in_features = input_size_past_t, output_features = hidden_size_lstm)
        #self.nn_fut = torch.nn.Linear(in_features = input_size_fut_t)

        self.lstm_e = torch.nn.LSTM(input_size=input_size_e, hidden_size=hidden_size_lstm, num_layers=layers_e,
                                    batch_first=True,bidirectional=bidir_e).to(dev)  # Encoder
        self.lstm_d = torch.nn.LSTM(input_size=input_size_d+hidden_size_lstm, hidden_size=hidden_size_lstm, num_layers=layers_d,
                                    batch_first=True,bidirectional=False).to(dev)  # Decoder
        self.fc = torch.nn.Linear(hidden_size_lstm, output_dim).to(dev) # fully connected 1

        self.attn = AttentionModule(hidden_size_lstm,dev)

        if bidir_e:
            self.mul_bidir_e = 2
        else:
            self.mul_bidir_e = 1


    def forward(self, list_data,dev_type='NA'):
        x_e = list_data[0]
        x_d = list_data[1]

        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        h_0 = Variable(torch.zeros(self.layers_e * self.mul_bidir_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.layers_e * self.mul_bidir_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # internal state
        # Propagate input through LSTM
        output_e, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0))  # lstm with input, hidden, and internal state

        if self.bidir_e:
            output_e = self.aggregate_bidirectional(tensors=output_e,dim=2)
            h_e,c_e = self.aggregate_bidirectional(tensors=[h_e,c_e],dim=0)

        output_d = []

        for t in range(x_d.size(1)):
            input_d_step = x_d[:,t,:]
            # Reshape input_d_step for seq_len=1
            input_d_step = input_d_step.unsqueeze(1)  # Shape: (batch_size, 1, input_size_d + hidden_size_lstm)

            # Expand h_e[-1] to match the sequence length of output_e
            h_e_expanded = h_e[-1].unsqueeze(1).repeat(1, output_e.size(1),
                                                       1) # Shape: (batch_size, seq_length_e, hidden_size_lstm)

            # Compute attention weights
            attn_weights = self.attn(h_e_expanded, output_e)

            # Compute context vector
            context = attn_weights.unsqueeze(1).bmm(output_e).squeeze(1)  # Shape: (batch_size, hidden_size_lstm)

            # Concatenate context vector with input_d_step
            input_d_step = torch.cat((input_d_step.squeeze(1), context),
                                     dim=1)  # Shape: (batch_size, input_size_d + 2 * hidden_size_lstm)

            # Pass the concatenated vector to the LSTM
            output_d_step, (h_e, c_e) = self.lstm_d(input_d_step.unsqueeze(1), (h_e, c_e))

            # Collect outputs
            output_d.append(output_d_step.squeeze(1))

        stacked_output = torch.stack(output_d,dim=1)
        out = torch.squeeze(self.fc(stacked_output))

        return out

    def aggregate_bidirectional(self, tensors,dim):

        def aggregate_tensor(t,dim):
            t_1, t_2 = torch.split(t, int(t.shape[dim] / 2), dim)
            t_agg = (t_1 + t_2) / 2
            return t_agg

        if isinstance(tensors,list):
            aggregated_tensors = []
            for t in tensors:
                aggregated_tensors.append(aggregate_tensor(t,dim))
            return aggregated_tensors
        else:
            return aggregate_tensor(tensors,dim)


    def set_device(self,dev):
        self.dev = dev
        self.lstm_e.to(dev)
        self.lstm_d.to(dev)
        self.fc.to(dev)

"""
Transformer class, as produced by chatgpt
"""

import torch
import torch.nn as nn
from torch.nn import Transformer
import math


class TransformerModel(nn.Module):

    def __init__(self, input_size, hidden_size, nhead, num_encoder_layers, num_decoder_layers, output_size,
                 dropout=0.5):
        super(TransformerModel, self).__init__()

        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_size, dropout)

        encoder_layers = TransformerEncoderLayer(input_size, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        decoder_layers = TransformerDecoderLayer(input_size, nhead, hidden_size, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        self.encoder = nn.Linear(input_size, input_size)
        self.decoder = nn.Linear(input_size, output_size)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src, self.src_mask)

        tgt = self.decoder(tgt)
        output = self.transformer_decoder(tgt, memory, self.src_mask)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


"""
Bi-attention, supposedly as translated by chatgpt from the tf code of Jeremie to torch code
"""

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


# # Define your input sizes
# input_size_past_ctxt = ...  # Define the appropriate size
# input_size_past_temp = ...  # Define the appropriate size
# input_size_fut_ctxt = ...  # Define the appropriate size
# input_size_fut_temp = ...  # Define the appropriate size
# hidden_size = ...  # Define the hidden size
# output_size = ...  # Define the output size
#
# # Encoder and Decoder initialization
# encoder = EncoderRNN(input_size_past_ctxt, input_size_past_temp, hidden_size)
# decoder = DecoderRNN(input_size_fut_ctxt, input_size_fut_temp, output_size, hidden_size)
#
# # Training loop (a basic example)
# for i in range(epochs):
#     encoder_outputs, encoder_hidden = encoder(past_ctxt, past_temp)
#     decoder_output, decoder_hidden = decoder(fut_ctxt, fut_temp, encoder_outputs)
#     # Compute loss, backpropagate, update weights, etc.





