import copy
import numpy as np
import cvxpy as cp
import torch
from torch.autograd import Variable
from cvxpylayers.torch import CvxpyLayer
import torch_classes as ct
from opti_problem import OptiProblem
import torch.nn.functional as F
import torch.nn as nn
import random

import sys
sys.path.insert(0,"../../ML_proxy/")
#from model import Storage_model


class Forecaster():

    def __init__(self,nn_params,OP_params,training_params,reg_type='quad',reg_val=0):
        self.nn_params = nn_params
        self.OP_params = OP_params
        self.training_params = training_params
        self.framework = training_params['framework']
        self.reg_type = reg_type
        self.reg_val = reg_val

        self.init_net(nn_params)
        self.init_diff_sched(training_params['framework'], OP_params)
        self.init_reg(reg_type,reg_val)

    def __call__(self,x):
        prices = self.nn(x)  # .is_leaf_(True)
        #prices.retain_grad()
        if self.include_opti:
            schedule = self.diff_sched_calc(prices)
            return [prices,schedule]
        else:
            return [prices]

    def init_net(self, nn_params):

        if nn_params['type'] == "vanilla":
            self.nn = NeuralNet(nn_params)
        elif nn_params['type'] == "vanilla_separate":
            self.nn = NeuralNetSep(nn_params)
        elif nn_params['type'] == "RNN_decoder":
            self.nn = Decoder(nn_params)
        elif nn_params['type'] == 'RNN_M2M':
            self.nn = RNN_M2M(nn_params)
        elif nn_params['type'] == 'LSTM_ED':
            self.nn = LSTM_ED(nn_params)
        elif nn_params['type'] == 'LSTM_ED_sep':
            self.nn = LSTM_ED_sep(nn_params)
        elif nn_params['type'] == 'LSTM_ED_attn':
            self.nn = LSTM_ED_Attention(nn_params)
        elif nn_params['type'] == 'ED_Transformer':
            self.nn = ED_Transformer(nn_params)
        else:
            ValueError(f"{type} is not supported as neural network type")

        if nn_params['warm_start']:
            self.nn = self.nn.warm_start(nn_params)

    def init_diff_sched(self, fw, OP_params):

        if fw == "ID":
            self.sc = Schedule_Calculator(OP_params, self.framework)
            self.diff_sched_calc = OptiLayer(OP_params)
            self.include_opti = True
        elif fw[0:2] == "GS":
            self.sc = Schedule_Calculator(OP_params, self.framework)
            self.diff_sched_calc = self.sc
            self.include_opti = True
        elif fw == "NA":
            self.diff_sched_calc = None
            self.include_opti = False
        else:
            raise ValueError("Invalid differentiable schedule calculator framework")

    def init_reg(self,r_type,val):
        self.reg_val = val

        if type in ['abs', 'quad']:
            self.reg_type = r_type
        else:
            ValueError(f"{r_type} unsupported type of regularization")

    def get_trainable_params(self):
        return self.nn.parameters()

    def set_nn(self,nn):
        self.nn = nn

    def calc_prices(self,features):
        return self.nn(features)

    def calc_sched_linear(self,features=None,prices=None):
        if prices==None:
            prices = self.nn(features)

        mu,d,c = self.sc.calc_linear(prices)
        net_sched = d*self.OP_params['eff_d'] - c/self.OP_params['eff_c']
        return net_sched

    def calc_reg(self):
        reg = 0.0
        for name,p in self.nn.named_parameters():
            if 'weight' in name: #Only count weights (not biases) in regularzation
                if self.reg_type == 'quad':
                    reg += torch.sum(torch.square(p))
                elif self.reg_type == 'abs':
                    reg += torch.sum(torch.abs(p))
        return self.reg_val*reg

    def make_clone(self):

        clone_nn = type(self.nn)(self.nn_params)
        clone_nn.load_state_dict(self.nn.state_dict())

        clone_fc = type(self)(self.nn_params, self.OP_params, self.training_params, self.reg_type, self.reg_val)
        clone_fc.set_nn(clone_nn)

        return clone_fc

# class NeuralNetWithOpti():
#
#     def __init__(self,price_gen,nn_params,params_dict,reg_type='quad',reg_val=0):
#         self.nn_params = nn_params
#         self.params_dict = params_dict
#         self.price_generator = price_gen
#         self.framework = nn_params['framework']
#         self.sc = Schedule_Calculator(params_dict,self.framework)
#         self.reg_val = reg_val
#         if reg_type in ['abs', 'quad']:
#             self.reg_type = reg_type
#         else:
#             ValueError(f"{reg_type} unsupported type of regularization")
#
#
#         if self.framework == "ID":
#             self.diff_sched_calc = OptiLayer(params_dict)
#         elif self.framework[0:2] == "GS":
#             self.diff_sched_calc = self.sc
#         else:
#             raise ValueError("Invalid differentiable schedule calculator framework")
#
#     def __call__(self,x):
#         prices = self.price_generator(x)  # .is_leaf_(True)
#         prices.retain_grad()
#         schedule = self.diff_sched_calc(prices)
#         return [schedule,prices]
#
#     def get_trainable_params(self):
#         return self.price_generator.parameters()
#
#     def calc_prices(self,features):
#         return self.price_generator(features)
#
#     def calc_sched_linear(self,features=None,prices=None):
#         if prices==None:
#             prices = self.price_generator(features)
#
#         mu,d,c = self.sc.calc_linear(prices)
#         net_sched = d*self.params_dict['eff_d'] - c/self.params_dict['eff_c']
#         return net_sched
#
#     def calc_reg(self):
#         reg = 0.0
#         for name,p in self.price_generator.named_parameters():
#             if 'weight' in name: #Only count weights (not biases) in regularzation
#                 if self.reg_type == 'quad':
#                     reg += torch.sum(torch.square(p))
#                 elif self.reg_type == 'abs':
#                     reg += torch.sum(torch.abs(p))
#         return self.reg_val*reg
#
#     def make_clone(self):
#
#         clone_price_gen = type(self.price_generator)(self.nn_params)
#         clone_price_gen.load_state_dict(self.price_generator.state_dict())
#         clone_nn_with_opti = type(self)(clone_price_gen, self.nn_params, self.params_dict, self.reg_type, self.reg_val)
#         return clone_nn_with_opti

class OptiLayer(torch.nn.Module):
    def __init__(self, params_dict):
        super(OptiLayer, self).__init__()
        self.OP = OptiProblem(params_dict)
        prob, params, vars = self.OP.get_opti_problem()
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

class Schedule_Calculator():
    def __init__(self, OP_params_dict,fw):
        super(Schedule_Calculator,self).__init__()

        self.sm = OP_params_dict['smoothing']
        self.sm_value = OP_params_dict['gamma']
        self.fw = fw
        self.OP_params_dict = copy.deepcopy(OP_params_dict)
        self.OP_params_dict['gamma'] = 0 #The optimization program to be used does not include the smoothing term: calculations based on actual linear problem
        self.op_RN = OptiProblem(self.OP_params_dict) #TODO: check if order of setting gamme to 0 correct

        op,params,vars = self.op_RN.get_opti_problem()

        self.params = params[0]

        if self.fw == 'GS_proxy':
            loc = '../../ML_proxy/trained_models/smoothing_training/'
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
                self.op_RN.solve(solver=cp.GUROBI)

                list_keys = list(self.op_RN.var_dict.keys())

                d[i, :] = torch.from_numpy(self.op_RN.var_dict[list_keys[0]].value)
                c[i, :] = torch.from_numpy(self.op_RN.var_dict[list_keys[1]].value)
                mu[i, 0] = self.op_RN.constraints[6].dual_value[0]
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

class NeuralNet(torch.nn.Module):
    def __init__(self, nn_params):
        super(NeuralNet, self).__init__()

        self.input_feat = nn_params['input_feat']
        self.output_dim = nn_params['output_dim']
        self.list_units = nn_params['list_units']
        self.list_act = nn_params['list_act']

        dict_act_fcts = {
            'relu': F.relu,
            'elu': F.elu,
            'softplus': F.softplus
        }

        # Define layers

        self.hidden_layers = torch.nn.ModuleList()
        self.act_fcts = []

        for i,units in enumerate(self.list_units):
            if i == 0:
                self.hidden_layers.append(torch.nn.Linear(self.input_feat,units))
            else:
                self.hidden_layers.append(torch.nn.Linear(self.list_units[i-1],units))

            self.act_fcts.append(dict_act_fcts[self.list_act[i]])

        if len(self.list_units)>0:
            self.final_layer = torch.nn.Linear(self.list_units[-1], self.output_dim)
        else:
            self.final_layer = torch.nn.Linear(self.input_feat, self.output_dim)

    def forward(self, x):

        x = x[0]

        for i,act in enumerate(self.act_fcts):

            x = act(self.hidden_layers[i](x))

        out = self.final_layer(x)


        return out

    def warm_start(self,nn_params):
        #TODO: generalize for any amount of hidden layers
        if nn_params['list_units'] == []:
            idx_fc = self.OP_params['feat_cols'].index('y_hat')
            with torch.no_grad():
                for i in range(self.final_layer.weight.shape[0]):
                    self.final_layer.bias[i] = 0
                    for j in range(self.final_layer.weight.shape[1]):
                        if j == self.OP_params['n_diff_features'] * i + idx_fc:
                            self.final_layer.weight[i, j] = 1
                        else:
                            self.final_layer.weight[i, j] = 0

        else:

            loc = "../../data/pretrained_fc/model_softplus_wd_nlfr_genfc_yhat_scaled_pos/"
            loc = nn_params['loc_ws']

            weights_layer_1 = torch.tensor(np.load(loc + 'weights_layer_1.npz'), dtype=torch.float32)
            biases_layer_1 = torch.tensor(np.load(loc + 'biases_layer_1.npz'), dtype=torch.float32)
            weights_layer_2 = torch.tensor(np.load(loc + 'weights_layer_2.npz'), dtype=torch.float32)
            biases_layer_2 = torch.tensor(np.load(loc + 'biases_layer_2.npz'), dtype=torch.float32)

            with torch.no_grad():
                self.hidden_layers[0].weight.copy_(weights_layer_1)
                self.hidden_layers[0].bias.copy_(biases_layer_1)
                self.final_layer.weight.copy_(weights_layer_2)
                self.final_layer.bias.copy_(biases_layer_2)

    def regularization(self,pow=1):

        reg = 0
        for layer in self.hidden_layers:
            reg+=torch.sum(torch.pow(torch.abs(layer.weight),pow))

        reg+= torch.sum(torch.pow(torch.abs(self.final_layer.weight),pow))

        return reg

class NeuralNetSep(torch.nn.Module):
    def __init__(self, nn_params):
        super(NeuralNetSep, self).__init__()

        self.input_feat = nn_params['input_feat']
        self.output_dim = nn_params['output_dim']
        self.list_units = nn_params['list_units']
        self.list_act = nn_params['list_act']

        dict_act_fcts = {
            'relu': F.relu,
            'elu': F.elu,
            'softplus': F.softplus
        }

        # Define layers

        self.hidden_layers = torch.nn.ModuleList()
        self.act_fcts = []

        for i,units in enumerate(self.list_units):
            if i == 0:
                self.hidden_layers.append(torch.nn.Linear(self.input_feat,units))
            else:
                self.hidden_layers.append(torch.nn.Linear(self.list_units[i-1],units))

            self.act_fcts.append(dict_act_fcts[self.list_act[i]])

        if len(self.list_units)>0:
            self.final_layer = torch.nn.Linear(self.list_units[-1], self.output_dim)
        else:
            self.final_layer = torch.nn.Linear(self.input_feat, self.output_dim)

    def forward(self, x_all):

        x_all = x_all[0]

        batch_size = x_all.size(0)
        la = x_all.size(1)

        tensor_dimensions = (batch_size,la,self.output_dim)
        out = torch.zeros(tensor_dimensions)

        for l in range(la):

            x = x_all[:,l,:]

            for i, act in enumerate(self.act_fcts):
                x = act(self.hidden_layers[i](x))

            out_la = self.final_layer(x)
            out[:,l,:] = out_la

        return torch.squeeze(out)

    def warm_start(self,nn_params):
        #TODO: generalize for any amount of hidden layers
        if nn_params['list_units'] == []:
            idx_fc = self.OP_params['feat_cols'].index('y_hat')
            with torch.no_grad():
                for i in range(self.final_layer.weight.shape[0]):
                    self.final_layer.bias[i] = 0
                    for j in range(self.final_layer.weight.shape[1]):
                        if j == self.OP_params['n_diff_features'] * i + idx_fc:
                            self.final_layer.weight[i, j] = 1
                        else:
                            self.final_layer.weight[i, j] = 0

        else:

            loc = "../../data/pretrained_fc/model_softplus_wd_nlfr_genfc_yhat_scaled_pos/"
            loc = nn_params['loc_ws']

            weights_layer_1 = torch.tensor(np.load(loc + 'weights_layer_1.npz'), dtype=torch.float32)
            biases_layer_1 = torch.tensor(np.load(loc + 'biases_layer_1.npz'), dtype=torch.float32)
            weights_layer_2 = torch.tensor(np.load(loc + 'weights_layer_2.npz'), dtype=torch.float32)
            biases_layer_2 = torch.tensor(np.load(loc + 'biases_layer_2.npz'), dtype=torch.float32)

            with torch.no_grad():
                self.hidden_layers[0].weight.copy_(weights_layer_1)
                self.hidden_layers[0].bias.copy_(biases_layer_1)
                self.final_layer.weight.copy_(weights_layer_2)
                self.final_layer.bias.copy_(biases_layer_2)

    def regularization(self,pow=1):

        reg = 0
        for layer in self.hidden_layers:
            reg+=torch.sum(torch.pow(torch.abs(layer.weight),pow))

        reg+= torch.sum(torch.pow(torch.abs(self.final_layer.weight),pow))

        return reg

class Decoder(torch.nn.Module):
    def __init__(self, nn_params):
        super(Decoder, self).__init__()

        self.input_size_d = nn_params['input_size_d']  # input size
        self.layers_d = nn_params['layers_d']
        self.hidden_size_lstm = nn_params['hidden_size_lstm']  # hidden state
        self.out_dim_per_neuron = nn_params['out_dim_per_neuron']
        self.output_dim = nn_params['output_dim']
        self.dev = nn_params['dev']
        self.act_last = nn_params['act_last']

        self.lstm_d = torch.nn.LSTM(input_size=self.input_size_d, hidden_size=self.hidden_size_lstm, num_layers=self.layers_d,
                                    batch_first=True,bidirectional=False).to(self.dev)  # Decoder
        self.fc_neuron = torch.nn.Linear(self.hidden_size_lstm, self.out_dim_per_neuron).to(self.dev) # fully connected 1

        #self.act_last = self.retrieve_act_fun_string(self.act_last)

        #self.fc_final = torch.nn.Linear(self.out_dim_per_neuron,1)

    def forward(self, list_data,dev_type='NA'):
        x_d = list_data[0]

        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        h_0 = Variable(torch.zeros(self.layers_d, x_d.size(0), self.hidden_size_lstm)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.layers_d, x_d.size(0), self.hidden_size_lstm)).to(dev)  # internal state

        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_0, c_0))
        #out_neurons = self.act_last(self.fc_neuron(output_d))
        #out = torch.squeeze(self.fc_final(out_neurons))
        out = torch.squeeze(self.fc_neuron(output_d))  # Final Output
        return out

    def set_device(self,dev):
        self.dev = dev
        self.lstm_e.to(dev)
        self.lstm_d.to(dev)
        self.fc.to(dev)

    def retrieve_act_fun_string(self,act_fun_str):
        if act_fun_str == 'relu':
            fun = torch.nn.ReLU()
        else:
            raise ValueError(f"{act_fun_str} unsupported activation function")

        return fun

class RNN_M2M(torch.nn.Module):
    def __init__(self, rnn_params):
        super(RNN_M2M, self).__init__()

        torch.manual_seed(73)

        self.rnn_params = rnn_params
        self.input_size = rnn_params['input_size']  # input size = n_features per instance
        self.output_size = rnn_params['output_size'] #n_features per output instance
        self.output_length = rnn_params['output_length'] #n of output instances
        self.layers = rnn_params['layers']
        self.hidden_size_lstm = rnn_params['hidden_size_lstm']  # hidden state
        self.dev = rnn_params['dev']

        # Encoder
        self.encoder_lstm = torch.nn.LSTM(self.input_size, self.hidden_size_lstm, self.layers,batch_first=True,bidirectional=True).to(self.dev)

        # Decoder
        self.decoder_lstm = torch.nn.LSTM(self.output_size, 2 * self.hidden_size_lstm, self.layers,batch_first=True).to(self.dev)
        self.decoder_fc = torch.nn.Linear(2*self.hidden_size_lstm, self.output_size).to(self.dev)

    def set_dev(self,dev):
        self.dev = dev
        self.encoder_lstm.to(dev)
        self.decoder_lstm.to(dev)
        self.decoder_fc.to(dev)

    def forward(self, input_seq, target_seq=None, tf_ratio=0.3):
        # Initialize hidden and cell states with zeros
        input_seq = input_seq[0]
        if target_seq is not None:
            target_seq = target_seq[0]

        h_0 = torch.zeros(self.layers*2, input_seq.size(0), self.hidden_size_lstm).to(self.dev)
        c_0 = torch.zeros(self.layers*2, input_seq.size(0), self.hidden_size_lstm).to(self.dev)

        # Encoding
        _, (hidden, cell) = self.encoder_lstm(input_seq, (h_0, c_0))

        # Adjusting the hidden state for unidirectional decoder
        hidden = self._cat_hidden(hidden)
        cell = self._cat_hidden(cell)

        # Decoding
        decoder_input = torch.zeros(input_seq.size(0), 1, self.output_size).to(self.dev)
        outputs = []
        for t in range(self.output_length):
            decoder_output, (hidden, cell) = self.decoder_lstm(decoder_input, (hidden, cell))
            prediction = self.decoder_fc(decoder_output)
            outputs.append(prediction)

            # Determine if we will use teacher forcing
            if target_seq is not None and random.random() < tf_ratio:
                # Using the actual target output as the next input
                decoder_input = target_seq[:, t:t+1, :]
            else:
                # Using the model's prediction as the next input
                decoder_input = prediction

        pred = torch.cat(outputs, dim=1)

        return pred.squeeze(dim=2)

    def _cat_hidden(self, hidden):
        # Concatenate the hidden states from both directions
        hidden_forward = hidden[0:hidden.size(0):2]
        hidden_backward = hidden[1:hidden.size(0):2]
        return torch.cat((hidden_forward, hidden_backward), dim=2)

class LSTM_ED(torch.nn.Module):
    def __init__(self, nn_params):

        super(LSTM_ED, self).__init__()
        self.input_size_e = nn_params['input_size_e']  # input size
        self.input_size_d = nn_params['input_size_d']  # input size
        self.layers_e = nn_params['layers_e']
        self.layers_d = nn_params['layers_d']
        self.hidden_size_lstm = nn_params['hidden_size_lstm']  # hidden state
        self.output_dim = nn_params['output_dim']
        self.dev = nn_params['dev']

        self.lstm_e = torch.nn.LSTM(input_size=self.input_size_e, hidden_size=self.hidden_size_lstm, num_layers=self.layers_e,
                                    batch_first=True,bidirectional=False).to(self.dev)  # Encoder
        self.lstm_d = torch.nn.LSTM(input_size=self.input_size_d, hidden_size=self.hidden_size_lstm, num_layers=self.layers_d,
                                    batch_first=True,bidirectional=False).to(self.dev)  # Decoder
        self.fc = torch.nn.Linear(self.hidden_size_lstm, self.output_dim).to(self.dev) # fully connected 1


    def forward(self, list_data,dev_type='NA'):
        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        x_e = list_data[0].to(dev)
        x_d = list_data[1].to(dev)

        h_0 = Variable(torch.zeros(self.layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # internal state
        # Propagate input through LSTM
        output_e, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0))  # lstm with input, hidden, and internal state


        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_e, c_e))
        out = torch.squeeze(self.fc(output_d))  # Final Output
        return out

class LSTM_ED_sep(torch.nn.Module):
    def __init__(self, nn_params):

        super(LSTM_ED_sep, self).__init__()
        self.input_size_e = nn_params['input_size_e']  # input size
        self.input_size_d = nn_params['input_size_d']  # input size
        self.layers_e = nn_params['layers_e']
        self.layers_d = nn_params['layers_d']
        self.hidden_size_lstm = nn_params['hidden_size_lstm']  # hidden state
        self.output_dim = nn_params['output_dim']
        self.dev = nn_params['dev']

        self.encoders = torch.nn.ModuleList()
        self.decoders = torch.nn.ModuleList()
        self.fully_connected = torch.nn.ModuleList()

        for i in range(self.output_dim):
            self.encoders.append(
                torch.nn.LSTM(input_size=self.input_size_e, hidden_size=self.hidden_size_lstm, num_layers=self.layers_e,
                                    batch_first=True,bidirectional=False).to(self.dev)
            )
            self.decoders.append(torch.nn.LSTM(input_size=self.input_size_d, hidden_size=self.hidden_size_lstm, num_layers=self.layers_d,
                                    batch_first=True,bidirectional=False).to(self.dev)
            )
            self.fully_connected.append(
                torch.nn.Linear(self.hidden_size_lstm, 1).to(self.dev)
            )

    def forward(self, list_data,dev_type='NA'):
        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        x_e = list_data[0].to(dev)
        x_d = list_data[1].to(dev)

        h_0 = Variable(torch.zeros(self.layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # hidden state
        c_0 = Variable(torch.zeros(self.layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)  # internal state
        # Propagate input through LSTM
        list_out = []

        for i in range(self.output_dim):
            out_e,(h_e,c_e) = self.encoders[i](x_e,(h_0,c_0))
            out_d, _ = self.decoders[i](x_d,(h_e,c_e))
            out_fc = torch.squeeze(self.fully_connected[i](out_d))
            list_out.append(out_fc)

        out = torch.stack(list_out,dim=2)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_size,dev):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.dev = dev
        self.attn = nn.Linear(hidden_size * 2, hidden_size).to(dev)
        self.v = nn.Parameter(torch.rand(hidden_size)).to(dev)
        self.v.data.normal_(mean=0, std=1. / hidden_size**0.5)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        h_t = hidden[-1].repeat(seq_len, 1, 1).transpose(0, 1)
        #encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]

        attn_energies = self.score(h_t, encoder_outputs)

        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = F.tanh(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        v = self.v.view(1, -1, 1)
        v_rep = torch.tile(v,(energy.shape[0],1,1))
        energy = torch.bmm(energy, v_rep)  # [B*T*1]
        return energy.squeeze(2)

class LSTM_ED_Attention(nn.Module):
    def __init__(self, nn_params):
        super(LSTM_ED_Attention, self).__init__()
        self.input_size_e = nn_params['input_size_e']  # input size
        self.input_size_d = nn_params['input_size_d']  # input size
        self.layers_e = nn_params['layers_e']
        self.layers_d = nn_params['layers_d']
        self.hidden_size_lstm = nn_params['hidden_size_lstm']  # hidden state
        self.output_dim = nn_params['output_dim']
        self.dev = nn_params['dev']
        #self.seq_length_e = nn_params['seq_length_e']
        #self.seq_length_d = nn_params['seq_length_d']

        self.lstm_e = nn.LSTM(input_size=self.input_size_e, hidden_size=self.hidden_size_lstm, num_layers=self.layers_e,
                              batch_first=True, bidirectional=False).to(self.dev)  # Encoder
        self.lstm_d = nn.LSTM(input_size=self.input_size_d+self.hidden_size_lstm, hidden_size=self.hidden_size_lstm, num_layers=self.layers_d,
                              batch_first=True, bidirectional=False).to(self.dev)  # Decoder
        self.attention = Attention(self.hidden_size_lstm,self.dev)


        self.list_units = nn_params['list_units']
        self.list_act = nn_params['list_act']


        dict_act_fcts = {
            'relu': F.relu,
            'elu': F.elu,
            'softplus': F.softplus
        }

        # Define layers

        self.hidden_layers = torch.nn.ModuleList()
        self.act_fcts = []

        for i, units in enumerate(self.list_units):
            if i == 0:
                self.hidden_layers.append(torch.nn.Linear(self.hidden_size_lstm, units).to(self.dev))
            else:
                self.hidden_layers.append(torch.nn.Linear(self.list_units[i - 1], units).to(self.dev))

            self.act_fcts.append(dict_act_fcts[self.list_act[i]])

        if len(self.list_units) > 0:
            self.final_layer = torch.nn.Linear(self.list_units[-1], self.output_dim).to(self.dev)
        else:
            self.final_layer = torch.nn.Linear(self.hidden_size_lstm, self.output_dim).to(self.dev)

    def forward(self, list_data, dev_type='NA'):
        x_e = list_data[0]
        x_d = list_data[1]

        if dev_type == 'NA':
            dev = self.dev
        else:
            dev = dev_type

        h_0 = Variable(torch.zeros(self.layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)
        c_0 = Variable(torch.zeros(self.layers_e, x_e.size(0), self.hidden_size_lstm)).to(dev)
        encoder_outputs, (h_e, c_e) = self.lstm_e(x_e, (h_0, c_0)) #dim encoder_outputs: [bs,la,hidden]

        # Attention mechanism
        attn_weights = self.attention(h_e, encoder_outputs) #dim attn_weights: [bs,la]
        context = attn_weights.bmm(encoder_outputs)  # dim context: [bs,1,hidden]
        context = context.repeat(1, x_d.size(1), 1)

        # Concatenate the context vector with the decoder input
        x_d = torch.cat([x_d, context], dim=2)

        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_e, c_e))

        for i,act in enumerate(self.act_fcts):

            output_d = act(self.hidden_layers[i](output_d))

        out = torch.squeeze(self.final_layer(output_d))



        return out


class ED_Transformer(nn.Module):
    def __init__(self,nn_params):
        super(ED_Transformer, self).__init__()

        self.encoder_seq_length = nn_params['encoder_seq_length']
        self.decoder_seq_length = nn_params['decoder_seq_length']
        self.decoder_size = nn_params['decoder_size']
        self.encoder_size = nn_params['encoder_size']
        self.num_heads = nn_params['num_heads']
        self.num_layers = nn_params['num_layers']
        self.ff_dim = nn_params['ff_dim']
        self.dropout = nn_params['dropout']
        self.dev = nn_params['dev']
        self.output_size = nn_params['output_size']


        # Transformer Encoder
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=self.encoder_size, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=self.num_layers).to(self.dev)

        # Transformer Decoder
        self.decoder_layers = nn.TransformerDecoderLayer(d_model=self.decoder_size, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=self.num_layers).to(self.dev)

        # Fully connected layer for output
        self.fc = nn.Linear(self.decoder_size, self.output_size).to(self.dev)

    def forward(self, x):
        encoder_input = x[0]
        decoder_input = x[1]
        # Ensure the input sequence lengths match the specified lengths
        assert encoder_input.size(1) == self.encoder_seq_length, "Encoder input sequence length mismatch"
        assert decoder_input.size(1) == self.decoder_seq_length, "Decoder input sequence length mismatch"

        # Forward pass through the transformer encoder
        encoder_output = self.transformer_encoder(encoder_input)

        # Forward pass through the transformer decoder
        decoder_output = self.transformer_decoder(decoder_input, encoder_output)

        # Apply fully connected layer for final prediction
        output = self.fc(decoder_output)

        return output