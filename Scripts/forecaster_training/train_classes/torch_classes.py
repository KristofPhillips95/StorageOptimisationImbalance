import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

experiment_dir = os.path.join(current_dir, '..', '..', 'experiment')
ml_proxy_dir = os.path.join(current_dir, '..', '..', 'ML_proxy')
sys.path.insert(0,experiment_dir)
sys.path.insert(0,ml_proxy_dir)

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp
import copy
from cvxpylayers.torch import CvxpyLayer
#from functions_support import opti_problem_mu
#from model import Storage_model





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
        if type(prices) is list:
            prices = prices[0]
        schedules = output
        profit = torch.sum(torch.mul(schedules, prices))

        return -profit

class Loss(nn.Module):

    def __init__(self,obj,params):
        super(Loss, self).__init__()
        self.obj = obj
        self.params = params

    def forward(self,preds,labels):

        """
        + [None]*n adds n times None to the list to ensure it has the minimum required elements for unpacking
        We  only use the first n elements to recover the actual inputs if they are there
        """
        #pred_price, pred_sched = (preds + [None]*2)[:2]
        #act_price,opt_schedules,weights = (labels + [None]*3)[:3]

        if self.obj == "profit":
            loss = -torch.sum(torch.mul(preds[1], labels[0]))
        elif self.obj == "mse_sched":
            loss = torch.sum(torch.square(preds[1]-labels[1]))
        elif self.obj == "mse_sched_weighted":
            loss = torch.sum(torch.sum(torch.square(preds[1]-labels[1]),axis=1)*self.params['weights'])
        elif self.obj == "mse_price":
            loss = torch.mean(torch.square(preds[0]-labels[0]))
        elif self.obj == "mae_price":
            loss = torch.mean(torch.abs(preds[0]-labels[0]))
        elif self.obj == 'pinball':
            diff = labels[0] - preds[0]
            mask_pos = diff >= 0
            diff_pos = torch.mul(mask_pos, diff)
            diff_neg = torch.mul(~mask_pos, diff)

            loss = torch.sum(torch.mul(diff_pos, self.params['quantile_tensor']) - torch.mul(diff_neg, 1 - self.params['quantile_tensor']))

        else:
            ValueError(f"{self.obj} unsupported loss function")

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


"""
Bi-directional RNNs with attention by chatgpt
"""
class AttentionModule(torch.nn.Module):
    """
    Implements the attention mechanism.
    """
    def __init__(self, hidden_size_lstm):
        """
        Args:
            hidden_size_lstm (int): Size of the LSTM hidden state.
        """
        super(AttentionModule, self).__init__()
        self.attn = torch.nn.Linear(2*hidden_size_lstm + hidden_size_lstm, hidden_size_lstm)
        self.v = torch.nn.Parameter(torch.rand(hidden_size_lstm))

    def forward(self, hidden, encoder_outputs):
        """
        Forward propagate the attention mechanism.

        Args:
            hidden (Tensor): The previous hidden state of the decoder LSTM.
            encoder_outputs (Tensor): The output sequences from the encoder LSTM.

        Returns:
            Tensor: Attention weights.
        """
        # Calculate attention weights (energies)
        attn_energies = self.score(hidden, encoder_outputs)
        attn_energies = attn_energies.t().unsqueeze(1)

        return F.softmax(attn_energies, dim=2)

    def score(self, hidden, encoder_outputs):
        """
        Compute attention scores.

        Args:
            hidden (Tensor): The previous hidden state of the decoder LSTM.
            encoder_outputs (Tensor): The output sequences from the encoder LSTM.

        Returns:
            Tensor: Attention scores.
        """
        energy = self.attn(torch.cat((hidden.repeat(encoder_outputs.size(0), 1, 1), encoder_outputs), 2))
        energy = energy.transpose(1, 2)
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy = torch.bmm(v, energy)
        return energy.squeeze(1)

class LSTM_ED_Attention(torch.nn.Module):
    """
    LSTM-based Encoder-Decoder model with Attention mechanism.
    """
    def __init__(self, input_size_e, hidden_size_lstm, input_size_d, output_dim, dev):
        """
        Args:
            input_size_e (int): Feature dimension of input data for the encoder.
            hidden_size_lstm (int): Size of the LSTM hidden state.
            input_size_d (int): Feature dimension of input data for the decoder.
            output_dim (int): Dimensionality of the model output.
            dev (str): Device to deploy the model to ('cpu' or 'cuda').
        """
        super(LSTM_ED_Attention, self).__init__()

        # Bidirectional LSTM for the encoder
        self.lstm_e = torch.nn.LSTM(input_size=input_size_e, hidden_size=hidden_size_lstm, num_layers=1,
                                    batch_first=True, bidirectional=True).to(dev)

        # Attention mechanism
        self.attn = AttentionModule(hidden_size_lstm)

        # LSTM for the decoder
        self.lstm_d = torch.nn.LSTM(input_size=input_size_d + 2*hidden_size_lstm, hidden_size=hidden_size_lstm,
                                    num_layers=1, batch_first=True, bidirectional=False).to(dev)

        # Fully connected layer for the final output
        self.fc = torch.nn.Linear(hidden_size_lstm, output_dim).to(dev)

    def forward(self, list_data, dev_type='NA'):
        """
        Forward propagate the model.

        Args:
            list_data (list): A list containing encoder and decoder input data.
            dev_type (str, optional): Device type, if different from initialized device. Defaults to 'NA'.

        Returns:
            Tensor: Model's output predictions.
        """
        x_e = list_data[0]  # Encoder input
        x_d = list_data[1]  # Decoder input

        # Determine device type
        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        # Initialize hidden and cell states for the encoder LSTM
        h_0 = Variable(torch.zeros(2, x_e.size(0), self.hidden_size_lstm)).to(dev)  # 2 for bidirectionality
        c_0 = Variable(torch.zeros(2, x_e.size(0), self.hidden_size_lstm)).to(dev)

        # Pass encoder input through bidirectional LSTM
        output_e, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0))

        # Combine bidirectional LSTM outputs
        output_e = (output_e[:, :, :self.hidden_size_lstm] + output_e[:, :, self.hidden_size_lstm:])

        # Compute attention weights and get context
        attn_weights = self.attn(h_e[-1], output_e)
        context = attn_weights.bmm(output_e.transpose(0, 1))

        # Concatenate context to decoder input
        x_d = torch.cat((x_d, context.transpose(0, 1)), 2)

        # Pass decoder input through LSTM
        output_d, _ = self.lstm_d(x_d, (h_e.view(1, h_e.size(1), -1), c_e.view(1, c_e.size(1), -1)))

        out = torch.squeeze(self.fc(output_d))
        return out

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




""" Classes for training with optimization """

class NeuralNet(torch.nn.Module):
    def __init__(self, input_feat, output_dim,list_units=[],list_act=[]):
        super(NeuralNet, self).__init__()

        self.input_feat = input_feat
        self.output_dim = output_dim
        self.list_units = list_units
        self.list_act = list_act

        dict_act_fcts = {
            'relu': F.relu,
            'elu': F.elu,
            'softplus': F.softplus
        }

        # Define layers

        self.hidden_layers = []
        self.act_fcts = []

        for i,units in enumerate(list_units):
            if i == 0:
                self.hidden_layers.append(torch.nn.Linear(input_feat,units))
            else:
                self.hidden_layers.append(torch.nn.Linear(list_units[i-1],units))

            self.act_fcts.append(dict_act_fcts[list_act[i]])

        if len(list_units)>0:
            self.final_layer = torch.nn.Linear(list_units[-1], output_dim)
        else:
            self.final_layer = torch.nn.Linear(input_feat, output_dim)

    def regularization(self,pow=1):

        reg = 0
        for layer in self.hidden_layers:
            reg+=torch.sum(torch.pow(torch.abs(layer.weight),pow))

        reg+= torch.sum(torch.pow(torch.abs(self.final_layer.weight),pow))

        return reg

    def forward(self, x):

        x = x[0]

        for i,act in enumerate(self.act_fcts):

            x = act(self.hidden_layers[i](x))

        out = self.final_layer(x)


        return out

class OptiLayer(torch.nn.Module):
    def __init__(self, params_dict):
        super(OptiLayer, self).__init__()
        prob, params, vars = opti_problem_mu(params_dict)
        self.layer = CvxpyLayer(prob, params, vars)

    def forward(self, x):
        if isinstance(x,list):
            x=x[0]
        #return self.layer(x, solver_args={'max_iters': 10000, 'solve_method': 'ECOS'})[0]  # 'eps': 1e-4,'mode':'dense'
        try:
            result = self.layer(x,solver_args={"solve_method": "ECOS"})[0] #"n_jobs_forward": 1
        except:
            print("solvererror occured")
            result = self.layer(x,solver_args={"solve_method": "SCS"})[0]


        return result

class NeuralNetWithOpti(torch.nn.Module):
    def __init__(self, price_gen,params_dict):
        super(NeuralNetWithOpti, self).__init__()
        self.price_generator = price_gen
        self.convex_opti_layer = OptiLayer(params_dict)
        self.params_dict = params_dict

    def forward(self, x):
        prices = self.price_generator(x)  # .is_leaf_(True)
        try:
            prices.retain_grad()
        except:
            prices[0].retain_grad()
        schedule = self.convex_opti_layer(prices)
        return schedule

    def regularization(self):

        return self.price_generator.regularization()

    def get_intermediate_price_prediction(self,x):
        return self.price_generator(x)

class NeuralNetWithOpti_new():

    def __init__(self,price_gen,params_dict,framework,reg_type='quad',reg_val=0):
        self.params_dict = params_dict
        self.price_generator = price_gen
        self.framework = framework
        self.sc = Schedule_Calculator(params_dict,framework)
        self.reg_val = reg_val
        if reg_type in ['abs', 'quad']:
            self.reg_type = reg_type
        else:
            ValueError(f"{reg_type} unsupported type of regularization")


        if framework == "ID":
            self.diff_sched_calc = OptiLayer(params_dict)
        elif framework[0:2] == "GS":
            self.diff_sched_calc = self.sc
        else:
            raise ValueError("Invalid differentiable schedule calculator framework")

    def __call__(self,x):
        prices = self.price_generator(x)  # .is_leaf_(True)
        prices.retain_grad()
        schedule = self.diff_sched_calc(prices)
        return schedule

    def calc_prices(self,features):
        return self.price_generator(features)

    def calc_sched_linear(self,features=None,prices=None):
        if prices==None:
            prices = self.price_generator(features)

        mu,d,c = self.sc.calc_linear(prices)
        net_sched = d*self.params_dict['eff_d'] - c/self.params_dict['eff_c']
        return net_sched

    def calc_reg(self):
        reg = 0.0
        for name,p in self.price_generator.named_parameters():
            if 'weight' in name: #Only count weights (not biases) in regularzation
                if self.reg_type == 'quad':
                    reg += torch.sum(torch.square(p))
                elif self.reg_type == 'abs':
                    reg += torch.sum(torch.abs(p))
        return self.reg_val*reg





##### CLASSES FOR SELF-DEFINING THE GRADIENT ####

import torch
import torch.nn as nn
import torch.optim as optim

# Create a simple neural network
class MyModel(nn.Module):
    def __init__(self,input_size,h_size,output_size):
        super(MyModel, self).__init__()
        #self.fc_1 = nn.Linear(input_size,h_size,bias=False)
        self.fc_2 = nn.Linear(h_size, output_size,bias=False)

    def forward(self, x):
        #hid = F.relu(self.fc_1(x))
        out = self.fc_2(x)
        return out

# Define a custom autograd function for computing gradients w.r.t. prices
class CustomGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model_output,true_values,gc):
        ctx.save_for_backward(model_output,true_values)
        ctx.gc = gc

        return model_output

    @staticmethod
    def backward(ctx, grad_output):
        model_output,true_values = ctx.saved_tensors
        #custom_grad = torch.ones_like(input_tensor) * 3  # Set the custom gradient to 3
        gc = ctx.gc
        custom_grad = gc(model_output,true_values)
        return custom_grad*grad_output, None, None

class Grad_Calculator():
    def __init__(self, OP_params_dict, sm, sm_val):
        super(Grad_Calculator,self).__init__()

        self.OP_params_dict = copy.deepcopy(OP_params_dict)
        self.OP_params_dict['gamma'] = 0

        op,params,vars = opti_problem_mu(OP_params_dict)

        self.op = op
        self.params = params[0]
        self.sm = sm
        self.sm_value = sm_val

    def __call__(self,y_hat,y):
        mu = self.retrieve_mu(y_hat)
        dxdp = self.retrieve_grad_opti(y_hat,mu) #Gradient of optimal decision w.r.t. input price
        dLdp = -y * dxdp #Gradient of Loss w.r.t. input price; Notice that as such we make the assumption that the loss function is -y * net_discharge

        mu_np = mu.detach().numpy()
        y_hat_np = y_hat.detach().numpy()
        y_np = y.detach().numpy()
        dLdp_np = dLdp.detach().numpy()
        dxdp_np = dxdp.detach().numpy()

        return dLdp

    def retrieve_mu(self, y_hat):

        mu = torch.zeros_like(y_hat)
        y_hat_np = y_hat.detach().numpy()

        for i in range(y_hat.size()[0]):

            try:
                self.params.value = y_hat_np[i,:]
            except:
                self.params.value = y_hat_np[:]

            self.op.solve(solver=cp.GUROBI)

            try:
                mu[i,0] = self.op.constraints[6].dual_value[0]
                for j in range(self.OP_params_dict['lookahead']):
                    if j == 0:
                        mu[i,j] = self.op.constraints[6].dual_value[j]
                    else:
                        mu[i,j] = self.op.constraints[7].dual_value[j]
            except:

                #self.soc[:] = self.op.var_dict['var173'].value

                mu[0] = self.op.constraints[6].dual_value[0]
                for j in range(self.OP_params_dict['lookahead']):
                    if j == 0:
                        mu[j] = self.op.constraints[6].dual_value[j]
                    else:
                        mu[j] = self.op.constraints[7].dual_value[j]

        return mu

    def retrieve_grad_opti(self, y_hat, mu):
        def calc_dddp_quadr(y_hat,mu):
            # Calculate the conditions
            condition1 = mu < y_hat * self.OP_params_dict['eff_d']
            condition2 = y_hat * self.OP_params_dict['eff_d']< mu + self.sm_value * self.OP_params_dict['max_discharge']
            final_condition = condition1 & condition2

            # Convert boolean tensor to integer tensor (1 where True, 0 where False)
            mask = final_condition.int()
            dddp = mask * self.OP_params_dict['eff_d'] / self.sm_value

            return dddp

        def calc_dcdp_quadr(y_hat,mu):
            # Calculate the conditions
            condition1 = mu > y_hat / self.OP_params_dict['eff_c']
            condition2 = y_hat / self.OP_params_dict['eff_c']> mu - self.sm_value * self.OP_params_dict['max_discharge']
            final_condition = condition1 & condition2

            # Convert boolean tensor to integer tensor (1 where True, 0 where False)
            mask = final_condition.int()
            dcdp = - mask / self.OP_params_dict['eff_c'] / self.sm_value

            return dcdp

        def calc_dddp_logBarrier(y_hat,mu,sm_val):
            x=1
        def calc_dcdp_logBarrier(y_hat,mu,sm_val):
            x=1

        if self.sm == "quadratic":
            dddp = calc_dddp_quadr(y_hat,mu)
            dcdp = calc_dcdp_quadr(y_hat,mu)
        elif self.sm == "logBarrier":
            dddp = self.calc_dddp_quadr(y_hat,mu)
            dcdp = self.calc_dcdp_quadr(y_hat,mu)
        else:
            sys.exit(f"{self.sm} is an invalid smoothing style")

        return dddp * self.OP_params_dict['eff_d'] - dcdp / self.OP_params_dict['eff_c']

    def set_sm_val(self,val):
        self.sm_value = val

    def forward_pass(self,y_hat):
        d = np.zeros_like(y_hat.detach().numpy())
        c = np.zeros_like(y_hat.detach().numpy())
        s = np.zeros_like(y_hat.detach().numpy())

        y_hat_np = y_hat.detach().numpy()

        for i in range(y_hat.size()[0]):

            try:
                self.params.value = y_hat_np[i,:]
            except:
                self.params.value = y_hat_np[:]

            self.op.solve(solver=cp.GUROBI)

            list_keys = list(self.op.var_dict.keys())

            try:
                d[i,:] = self.op.var_dict[list_keys[0]].value
                c[i,:] = self.op.var_dict[list_keys[1]].value
                s[i,:] = self.op.var_dict[list_keys[2]].value

            except:
                d = self.op.var_dict[list_keys[0]].value
                c = self.op.var_dict[list_keys[1]].value
                s = self.op.var_dict[list_keys[2]].value

        return [d,c,s]

class Schedule_Calculator():
    def __init__(self, OP_params_dict,fw):
        super(Schedule_Calculator,self).__init__()

        self.sm = OP_params_dict['smoothing']
        self.sm_value = OP_params_dict['gamma']
        self.fw = fw
        self.OP_params_dict = copy.deepcopy(OP_params_dict)
        self.OP_params_dict['gamma'] = 0 #The optimization program to be used does not include the smoothing term: calculations based on actual linear problem

        op,params,vars = opti_problem_mu(self.OP_params_dict)

        self.op = op
        self.params = params[0]

        if self.fw == 'GS_proxy':
            loc = '../ML_proxy/trained_models/smoothing_training/'
            config = 4
            m = Storage_model.load_model(loc=loc,config=config)
            self.mu_calculator = m.best_net
            self.mu_calculator.set_dev("cpu")
            #Fix the mu_calculator, i.e. don't allow it to get updated in the training process
            for param in self.mu_calculator.parameters():
                param.requires_grad=False

    def __call__(self,y_hat):

        if self.fw == 'GS':
            mu,_,_ = self.calc_linear(y_hat)
        elif self.fw == 'GS_proxy':
            mu = self.mu_calculator(y_hat.unsqueeze(2))[0]
        else:
            raise ValueError(f"{self.fw} is an unsupported framework for gradient smoothing")

        d_sm = self.calc_sm_d(y_hat,mu)
        c_sm = self.calc_sm_c(y_hat,mu)

        net_sched_sm = d_sm*self.OP_params_dict['eff_d'] - c_sm/self.OP_params_dict['eff_c']

        #net_sched = d*self.OP_params_dict['eff_d'] - c/self.OP_params_dict['eff_c']
        #return net_sched_sm,net_sched,[mu,d,c]
        return net_sched_sm

    def calc_sm_d(self,y_hat,mu):

        def kahan_summation(input_list):
            sum_ = 0.0
            c = 0.0  # A running compensation for lost low-order bits.
            for x in input_list:
                y = x - c  # So far, so good: c is zero.
                t = sum_ + y  # Alas, sum_ is big, y small, so low-order digits of y are lost.
                c = (t - sum_) - y  # (t - sum_) recovers the high-order part of y; subtracting y recovers -(low part of y)
                sum_ = t  # Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
            return sum_

        def calc_d_quadr(y_hat,mu):
            condition1 = mu < y_hat * self.OP_params_dict['eff_d']
            condition2 = y_hat * self.OP_params_dict['eff_d']< mu + self.sm_value * self.OP_params_dict['max_discharge']
            final_condition = condition1 & condition2

            d_1 = 0 * (1-condition1.int())
            d_2 = (self.OP_params_dict["eff_d"]*y_hat - mu)/self.sm_value * final_condition.int()
            d_3 = self.OP_params_dict["max_discharge"] * (1-condition2.int())

            return d_1+d_2+d_3

        def calc_d_logBar(y_hat, mu):
            P = self.OP_params_dict['max_discharge']
            A = y_hat * self.OP_params_dict['eff_d'] - mu


            x = kahan_summation([A * P, - 2 * self.sm_value])
            x_squared = kahan_summation([torch.square(A)*P**2, - 4*A*P*self.sm_value, 4*self.sm_value**2])

            epsilon = 1e-4
            mask_zero = torch.abs(A) < epsilon
            mask_nonzero = ~mask_zero

            cutoff_y, slope = self.get_linear_interp(epsilon,P)

            d = torch.zeros_like(A)
            d[mask_zero] = self.OP_params_dict['max_discharge']/2
            d[mask_nonzero] = (x[mask_nonzero] + torch.sqrt(x_squared[mask_nonzero] + 4 * A[mask_nonzero] * self.sm_value * P)) / (2 * A[mask_nonzero])

            return d

        if self.sm == "quadratic":
            d = calc_d_quadr(y_hat,mu)
        elif self.sm == "logBar":
            d = calc_d_logBar(y_hat,mu)

        return d

    def calc_sm_c(self,y_hat,mu):
        def calc_c_quadr(y_hat,mu):
            condition1 = mu > y_hat / self.OP_params_dict['eff_c']
            condition2 = y_hat / self.OP_params_dict['eff_c']> mu - self.sm_value * self.OP_params_dict['max_discharge']
            final_condition = condition1 & condition2

            c_1 = 0 * (1-condition1.int())
            c_2 = (mu - y_hat/self.OP_params_dict['eff_c'])/self.sm_value * final_condition.int()
            c_3 = self.OP_params_dict["max_charge"] * (1-condition2.int())

            return c_1 + c_2 + c_3

        def calc_c_logBar(y_hat,mu):
            P = self.OP_params_dict['max_charge']
            A = mu - y_hat / self.OP_params_dict['eff_c']

            x = A * self.OP_params_dict['max_charge'] - 2 * self.sm_value

            epsilon = 1e-4
            mask_zero = torch.abs(A) < epsilon
            mask_nonzero = ~mask_zero

            cutoff_y, slope = self.get_linear_interp(epsilon,P)

            c = torch.zeros_like(A)
            c[mask_zero] = cutoff_y + slope*(A[mask_zero]+epsilon)
            c[mask_nonzero] = (x[mask_nonzero] + torch.sqrt(torch.square(x[mask_nonzero]) + 4 * A[mask_nonzero] * self.sm_value * self.OP_params_dict['max_charge']))/(2*A[mask_nonzero])

            return c


        if self.sm == "quadratic":
            c = calc_c_quadr(y_hat,mu)
        elif self.sm == "logBar":
            c = calc_c_logBar(y_hat,mu)

        return c

    def get_linear_interp(self,epsilon,P):
        y_2 = (epsilon * P - 2*self.sm_value + np.sqrt((epsilon*P - 2*self.sm_value)**2 + 4 * epsilon * self.sm_value * P))/(2*epsilon)
        y_1 = (-epsilon * P - 2*self.sm_value + np.sqrt((-epsilon*P - 2*self.sm_value)**2 - 4 * epsilon * self.sm_value * P))/(-2*epsilon)

        slope = (y_2-y_1)/(2*epsilon)

        return y_1,slope

    def set_sm_val(self,val):
        self.sm_value = val

    def calc_linear(self,y_hat):
        if isinstance(y_hat,list):
            y_hat = y_hat[0]
        d = torch.zeros_like(y_hat,requires_grad=False)
        c = torch.zeros_like(y_hat,requires_grad=False)
        mu = torch.zeros_like(y_hat,requires_grad=False)

        y_hat_np = y_hat.detach().numpy()

        try:
            _ = y_hat.size()[1]

            for i in range(y_hat.size()[0]):

                self.params.value = y_hat_np[i, :]
                self.op.solve(solver=cp.GUROBI)

                list_keys = list(self.op.var_dict.keys())

                d[i, :] = torch.from_numpy(self.op.var_dict[list_keys[0]].value)
                c[i, :] = torch.from_numpy(self.op.var_dict[list_keys[1]].value)
                mu[i, 0] = self.op.constraints[6].dual_value[0]
                for j in range(self.OP_params_dict['lookahead']):
                    if j == 0:
                        mu[i, j] = self.op.constraints[6].dual_value[j]
                    else:
                        mu[i, j] = self.op.constraints[7].dual_value[j]



        except:

            self.params.value = y_hat_np[:]
            self.op.solve(solver=cp.GUROBI)

            list_keys = list(self.op.var_dict.keys())

            d[:] = torch.from_numpy(self.op.var_dict[list_keys[0]].value)
            c[:] = torch.from_numpy(self.op.var_dict[list_keys[1]].value)
            mu[0] = self.op.constraints[6].dual_value[0]
            for j in range(self.OP_params_dict['lookahead']):
                if j == 0:
                    mu[j] = self.op.constraints[6].dual_value[j]
                else:
                    mu[j] = self.op.constraints[7].dual_value[j]


        return mu,d,c









