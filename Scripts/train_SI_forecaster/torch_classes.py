import torch
from torch.autograd import Variable
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

