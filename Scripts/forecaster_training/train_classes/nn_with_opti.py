import copy
import numpy as np
import cvxpy as cp
import torch
torch.autograd.set_detect_anomaly(True)

import math
from torch.autograd import Variable
from cvxpylayers.torch import CvxpyLayer
import torch_classes as ct
from opti_problem import OptiProblem,OptiProblemNew
import torch.nn.functional as F
import torch.nn as nn
import random
import pickle

import sys
#sys.path.insert(0,"../../ML_proxy/")
#from model import Storage_model

class Forecaster():

    def __init__(self,nn_params,OP_params,training_params,reg_type='quad',reg_val=0):
        self.nn_params = nn_params
        self.OP_params = OP_params
        self.training_params = training_params
        self.framework = training_params['framework']
        self.reg_type = reg_type
        self.reg_val = reg_val

        self.nn = self.init_net(nn_params)
        self.init_diff_sched(training_params['framework'], OP_params)
        self.init_reg(reg_type,reg_val)

    def __call__(self,x):
        prices = self.get_price(x)  # .is_leaf_(True)
        #prices.retain_grad()
        if self.include_opti:
            if self.training_params['MPC']:
                opti_input = [prices,x[2]]
            else:
                opti_input = prices
            net_schedule,separate_schedules,*_ = self.diff_sched_calc(opti_input)
            return [prices,net_schedule,separate_schedules]
        else:
            return [prices]

    def get_price(self,x):
        return self.nn(x)

    def init_net(self, nn_params):

        if not nn_params['warm_start']:

            if nn_params['type'] == "vanilla":
                nn = NeuralNet(nn_params)
            elif nn_params['type'] == "vanilla_separate":
                nn = NeuralNetSep(nn_params)
            elif nn_params['type'] == "RNN_decoder":
                nn = Decoder(nn_params)
            elif nn_params['type'] == 'RNN_M2M':
                nn = RNN_M2M(nn_params)
            elif nn_params['type'] == 'LSTM_ED':
                nn = LSTM_ED(nn_params)
            elif nn_params['type'] == 'LSTM_ED_Attention':
                nn = LSTM_ED_Attention(nn_params)
            elif nn_params['type'] == 'ED_Transformer':
                nn = ED_Transformer(nn_params)
            else:
                ValueError(f"{type} is not supported as neural network type")

        else:

            try: #new way
                loc_state = nn_params['warm_start'] + "_dict_model.pkl"
                loc_model = nn_params['warm_start'] + "_best_model.pth"
                with open(loc_state, 'rb') as file:
                    model_state = pickle.load(file)  # Gives a dict

                params = model_state['nn_params']
                params['dev'] = self.nn_params['dev']
                nn = self.init_net(params) #Creates a neural net without warm start based with the same structure as pre-trained model
                nn.load_state_dict(torch.load(loc_model))
            except: #old way
                'loading model in old way'
                nn = torch.load(nn_params['warm_start']+"_best_model.pth",map_location=nn_params['dev']).nn
                nn.set_device(nn_params['dev'])

        return nn



    def init_diff_sched(self, fw, OP_params):

        if fw == "ID":
            self.sc = Schedule_Calculator(OP_params, self.framework, self.nn_params['dev'])
            self.diff_sched_calc = OptiLayer(OP_params)
            self.include_opti = True
        elif fw in ['GS', 'GS_proxy', 'proxy_direct', 'proxy_direct_linear']:
            self.sc = Schedule_Calculator(OP_params, self.framework, self.nn_params['dev'])
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

    def set_device(self,dev):
        self.nn.set_device(dev)

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
        self.params_dict = params_dict
        #check_OP = OptiProblem(params_dict)
        #check_prob, check_params, check_vars = check_OP.get_opti_problem(params_dict['MPC'])
        self.OP = OptiProblemNew(params_dict)
        self.prob, self.params, self.vars = self.OP.get_opti_problem()


        self.layer = CvxpyLayer(self.prob, self.params, self.vars)
        #check_layer = CvxpyLayer(check_prob,check_params,check_vars)

        x=1

    def forward(self, x):
        if (not isinstance(x,list)) or len(x)<2:
            if len(x) == 1:
                x=x[0]
            soc_0 = torch.full((x.size(0),),self.params_dict['soc_0']).to(x.device).type(x.dtype)
            try:
                result = self.layer(soc_0,x, solver_args={"solve_method": "ECOS"})  # "n_jobs_forward": 1
            except:
                print("solvererror occured")
                result = self.layer(soc_0,x, solver_args={"solve_method": "SCS"})
        elif len(x) == 2:
            try:
                result = self.layer(x[1].squeeze(1),x[0], solver_args={"solve_method": "ECOS"})  # "n_jobs_forward": 1
            except:
                print("solvererror occured")
                result = self.layer(x[1].squeeze(1),x[0], solver_args={"solve_method": "SCS"})

        net_dis = result[0]
        sched_separate = [result[1],result[2],result[3]]
        return net_dis,sched_separate

    def rev_eng_mu(self,d,c):
        """Function to calculate mu from optimized schedule"""
        pass

class Schedule_Calculator():
    def __init__(self, OP_params_dict,fw,dev,bs=1,mu_calculator=None):
        super(Schedule_Calculator,self).__init__()

        self.sm = OP_params_dict['smoothing']
        self.sm_value = OP_params_dict['gamma']
        self.fw = fw
        self.dev = dev
        self.bs = bs #get rid of this?
        self.OP_params_dict = copy.deepcopy(OP_params_dict)
        self.op_sm = OptiProblemNew(OP_params_dict,bs) #smoothened optimization problem
        self.OP_params_dict['gamma'] = 0 #The optimization program to be used does not include the smoothing term: calculations based on actual linear problem
        self.op_RN = OptiProblemNew(self.OP_params_dict,bs) #linear optimization problem
        #self.feas_repair = False
        self.feas_repair = OP_params_dict['repair_proxy_feasibility']

        if self.fw in ['GS_proxy','proxy_direct','proxy_direct_linear']:
            if mu_calculator is None:
                self.initialize_proxy()
            else:
                self.mu_calculator = mu_calculator


        print(f"Repair: {self.feas_repair}")


    def __call__(self,y_hat,smooth=True):

        if not isinstance(y_hat, list):
            mu_calc_input = [y_hat.unsqueeze(2)]
            soc = None
        else:
            mu_calc_input = [y_hat[0].unsqueeze(2)] + y_hat[1:]
            soc = y_hat[1]
            y_hat = y_hat[0]

        if smooth:
            if self.fw[0:2] == "GS":
                if self.fw == 'GS':
                    mu = self.get_opti_outcome(y_hat,smooth=False)[2]
                elif self.fw == 'GS_proxy':
                    mu = self.mu_calculator(mu_calc_input)[0]
                else:
                    raise ValueError(f"{self.fw} is an unsupported framework for gradient smoothing")

                d_sm = self.calc_sm_d(y_hat,mu)
                c_sm = self.calc_sm_c(y_hat,mu)

                if self.feas_repair:
                    d_sm,c_sm = self.repair_decisions(d_sm,c_sm,y_hat,soc)

            elif self.fw in ["proxy_direct","proxy_direct_linear"]:

                #[mu,d_sm,c_sm] = self.mu_calculator(mu_calc_input)
                out = self.mu_calculator(mu_calc_input)
                mu = out[0]
                d_sm = out[1]
                c_sm = out[2]
                #TODO: re-scale d and c based on (i) max (dis)charge and (ii) what the model was trained for

                if self.feas_repair:
                    d_sm,c_sm = self.repair_decisions(d_sm,c_sm,y_hat,soc)


            elif self.fw == "ID":

                d_sm,c_sm,mu = self.get_opti_outcome(y_hat,smooth=True)

            net_sched_sm = d_sm * self.OP_params_dict['eff_d'] - c_sm / self.OP_params_dict['eff_c']

            return net_sched_sm, [d_sm,c_sm], mu

        else:

            d,c,mu = self.get_opti_outcome(y_hat,smooth=False)

            net_sched = d * self.OP_params_dict['eff_d'] - c / self.OP_params_dict['eff_c']

            return net_sched, [d,c], mu

    def initialize_proxy(self):
        loc = self.OP_params_dict['loc_proxy_model']
        m = Storage_model.load_model(loc=loc)
        self.mu_calculator = m.best_net
        self.mu_calculator.set_dev(self.dev)
        # Fix the mu_calculator, i.e. don't allow it to get updated in the training process
        for param in self.mu_calculator.parameters():
            param.requires_grad = True

    def kahan_summation(self,input_list):
        sum_ = 0.0
        c = 0.0  # A running compensation for lost low-order bits.
        for x in input_list:
            y = x - c  # So far, so good: c is zero.
            t = sum_ + y  # Alas, sum_ is big, y small, so low-order digits of y are lost.
            c = (t - sum_) - y  # (t - sum_) recovers the high-order part of y; subtracting y recovers -(low part of y)
            sum_ = t  # Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
        return sum_

    def calc_d_quadr(self,y_hat, mu):
        condition1 = mu < y_hat * self.OP_params_dict['eff_d']
        condition2 = y_hat * self.OP_params_dict['eff_d'] < mu + self.sm_value * self.OP_params_dict['max_discharge']
        final_condition = condition1 & condition2

        d_1 = 0 * (1 - condition1.int())
        d_2 = ((self.OP_params_dict["eff_d"] * y_hat - mu) / self.sm_value * final_condition.int())
        d_3 = (self.OP_params_dict["max_discharge"] * (1 - condition2.int()))

        return d_1 + d_2 + d_3

    def calc_d_quadr_symm(self,y_hat, mu):
        delta = (self.sm_value * self.OP_params_dict['max_discharge'])/2
        condition1 = y_hat > (mu - delta)/ self.OP_params_dict['eff_d']
        condition2 = y_hat < (mu + delta) / self.OP_params_dict['eff_d']
        final_condition = condition1 & condition2

        d_1 = 0 * (1 - condition1.int())
        d_2 = ((self.OP_params_dict["eff_d"] * y_hat - (mu-delta)) / self.sm_value * final_condition.int())
        d_3 = (self.OP_params_dict["max_discharge"] * (1 - condition2.int()))

        # check_y_hat = y_hat.cpu().detach().numpy()
        # check_mu = mu.cpu().detach().numpy()
        #
        # check_d_1 = d_1.cpu().detach().numpy()
        # check_d_2 = d_2.cpu().detach().numpy()
        # check_d_3 = d_3.cpu().detach().numpy()

        return d_1 + d_2 + d_3

    def calc_d_logistic(self,y_hat, mu, k=1):
        delta = (self.sm_value * self.OP_params_dict['max_discharge'])/2

        condition1 = y_hat > (mu - delta)/ self.OP_params_dict['eff_d']
        condition2 = y_hat < (mu + delta) / self.OP_params_dict['eff_d']
        final_condition = condition1 & condition2

        A = (1+math.exp(k*delta))/(math.exp(k*delta)-1)
        B = 1/2*(1-A)
        x_0 = mu / self.OP_params_dict['eff_d']

        d_1 = torch.zeros_like(y_hat) * (~condition1)
        d_2 = self.OP_params_dict['max_charge'] *((A/(1+torch.exp(-k*(y_hat-x_0)))) + B) * final_condition
        d_3 = self.OP_params_dict["max_discharge"] * torch.ones_like(y_hat) * (~condition2)

        # check_y_hat = y_hat.cpu().detach().numpy()
        # check_mu = mu.cpu().detach().numpy()
        #
        # check_d_1 = d_1.cpu().detach().numpy()
        # check_d_2 = d_2.cpu().detach().numpy()
        # check_d_3 = d_3.cpu().detach().numpy()

        return d_1 + d_2 + d_3

    def calc_d_piecewise(self,y_hat, mu):
        delta = (self.sm_value * self.OP_params_dict['max_discharge'])/2
        x_0 = mu / self.OP_params_dict['eff_d']


        condition1 = y_hat > (mu - delta)/ self.OP_params_dict['eff_d']
        condition2 = y_hat < (mu + delta) / self.OP_params_dict['eff_d']
        condition3 = y_hat < x_0

        condition_left = condition1 & condition3
        condition_right = condition2 &(~condition3)

        a = self.OP_params_dict['max_discharge']/(2*delta**2)
        b = self.OP_params_dict['max_discharge']/delta
        c = self.OP_params_dict['max_discharge']/2


        d_1 = torch.zeros_like(y_hat) * (~condition1)
        d_2 = (a*torch.square(y_hat-x_0) + b*(y_hat-x_0) + c) * condition_left
        d_3 = (-a*torch.square(y_hat-x_0) + b*(y_hat-x_0) + c) * condition_right
        d_4 = self.OP_params_dict["max_discharge"] * torch.ones_like(y_hat) * (~condition2)

        return d_1 + d_2 + d_3 + d_4

    def calc_d_logBar(self,y_hat, mu):
        P = self.OP_params_dict['max_discharge']
        A = y_hat * self.OP_params_dict['eff_d'] - mu

        x = self.kahan_summation([A * P, - 2 * self.sm_value])
        x_squared = self.kahan_summation([torch.square(A) * P ** 2, - 4 * A * P * self.sm_value, 4 * self.sm_value ** 2])

        epsilon = 1e-4
        mask_zero = torch.abs(A) < epsilon
        mask_nonzero = ~mask_zero

        cutoff_y, slope = self.get_linear_interp(epsilon, P)

        d = torch.zeros_like(A)
        d[mask_zero] = self.OP_params_dict['max_discharge'] / 2
        d[mask_nonzero] = (x[mask_nonzero] + torch.sqrt(
            x_squared[mask_nonzero] + 4 * A[mask_nonzero] * self.sm_value * P)) / (2 * A[mask_nonzero])

        return d

    def calc_sm_d(self,y_hat,mu):

        if self.sm == "quadratic":
            d = self.calc_d_quadr(y_hat,mu)
        elif self.sm == "quadratic_symm":
            d = self.calc_d_quadr_symm(y_hat,mu)
        elif self.sm == "logistic":
            d = self.calc_d_logistic(y_hat,mu)
        elif self.sm == "piecewise":
            d = self.calc_d_piecewise(y_hat, mu)
        elif self.sm == "logBar":
            d = self.calc_d_logBar(y_hat,mu)

        return d

    def calc_c_quadr(self,y_hat, mu):
        condition1 = mu > y_hat / self.OP_params_dict['eff_c']
        condition2 = y_hat / self.OP_params_dict['eff_c'] > mu - self.sm_value * self.OP_params_dict['max_discharge']
        final_condition = condition1 & condition2

        c_1 = 0 * (1 - condition1.int())
        c_2 = (mu - y_hat / self.OP_params_dict['eff_c']) / self.sm_value * final_condition.int()
        c_3 = self.OP_params_dict["max_charge"] * (1 - condition2.int())

        return c_1 + c_2 + c_3

    def calc_c_quadr_symm(self,y_hat, mu):

        delta = (self.sm_value * self.OP_params_dict['max_discharge'])/2

        condition1 = y_hat / self.OP_params_dict['eff_c'] < mu + delta
        condition2 = y_hat / self.OP_params_dict['eff_c'] > mu - delta
        final_condition = condition1 & condition2

        c_1 = 0 * (1 - condition1.int())
        c_2 = (mu + delta - y_hat / self.OP_params_dict['eff_c']) / self.sm_value * final_condition.int()
        c_3 = self.OP_params_dict["max_charge"] * (1 - condition2.int())

        # check_y_hat = y_hat.cpu().detach().numpy()
        # check_mu = mu.cpu().detach().numpy()
        #
        # check_c_1 = c_1.cpu().detach().numpy()
        # check_c_2 = c_2.cpu().detach().numpy()
        # check_c_3 = c_3.cpu().detach().numpy()


        return c_1 + c_2 + c_3

    def calc_c_logistic(self,y_hat, mu, k=1):

        delta = (self.sm_value * self.OP_params_dict['max_discharge'])/2

        condition1 = y_hat / self.OP_params_dict['eff_c'] < mu + delta
        condition2 = y_hat / self.OP_params_dict['eff_c'] > mu - delta
        final_condition = condition1 & condition2

        A = (1+math.exp(-k*delta))/(math.exp(-k*delta)-1)
        B = 1/2*(1-A)
        x_0 = mu * self.OP_params_dict['eff_c']

        c_1 = torch.zeros_like(y_hat) * (~condition1)
        c_2 = self.OP_params_dict['max_charge'] *((A/(1+torch.exp(-k*(y_hat-x_0)))) + B) * final_condition
        c_3 = self.OP_params_dict["max_discharge"] * torch.ones_like(y_hat) * (~condition2)

        check_y_hat = y_hat.cpu().detach().numpy()
        check_mu = mu.cpu().detach().numpy()

        check_c_1 = c_1.cpu().detach().numpy()
        check_c_2 = c_2.cpu().detach().numpy()
        check_c_3 = c_3.cpu().detach().numpy()


        return c_1 + c_2 + c_3

    def calc_c_piecewise(self, y_hat, mu):
        delta = (self.sm_value * self.OP_params_dict['max_discharge']) / 2
        x_0 = mu * self.OP_params_dict['eff_c']

        condition1 = y_hat > (mu - delta) * self.OP_params_dict['eff_d']
        condition2 = y_hat < (mu + delta) * self.OP_params_dict['eff_d']
        condition3 = y_hat < x_0

        condition_left = condition1 & condition3
        condition_right = condition2 & (~condition3)

        a = self.OP_params_dict['max_discharge'] / (2 * delta ** 2)
        b = self.OP_params_dict['max_discharge'] / delta
        c = self.OP_params_dict['max_discharge'] / 2

        c_1 = torch.zeros_like(y_hat) * (~condition2)
        c_2 = (-a * torch.square(y_hat - x_0) - b * (y_hat - x_0) + c) * condition_left
        c_3 = (a * torch.square(y_hat - x_0) - b * (y_hat - x_0) + c) * condition_right
        c_4 = self.OP_params_dict["max_discharge"] * torch.ones_like(y_hat) * (~condition1)

        return c_1 + c_2 + c_3 + c_4

    def calc_c_logBar(self,y_hat, mu):
        P = self.OP_params_dict['max_charge']
        A = mu - y_hat / self.OP_params_dict['eff_c']

        x = A * self.OP_params_dict['max_charge'] - 2 * self.sm_value

        epsilon = 1e-4
        mask_zero = torch.abs(A) < epsilon
        mask_nonzero = ~mask_zero

        cutoff_y, slope = self.get_linear_interp(epsilon, P)

        c = torch.zeros_like(A)
        c[mask_zero] = cutoff_y + slope * (A[mask_zero] + epsilon)
        c[mask_nonzero] = (x[mask_nonzero] + torch.sqrt(
            torch.square(x[mask_nonzero]) + 4 * A[mask_nonzero] * self.sm_value * self.OP_params_dict[
                'max_charge'])) / (2 * A[mask_nonzero])

        return c

    def calc_sm_c(self,y_hat,mu):

        if self.sm == "quadratic":
            c = self.calc_c_quadr(y_hat,mu)
        elif self.sm == "quadratic_symm":
            c = self.calc_c_quadr_symm(y_hat,mu)
        elif self.sm == "logistic":
            c = self.calc_c_logistic(y_hat,mu)
        elif self.sm == "piecewise":
            c = self.calc_c_piecewise(y_hat,mu)
        elif self.sm == "logBar":
            c = self.calc_c_logBar(y_hat,mu)

        return c

    def get_linear_interp(self,epsilon,P):
        y_2 = (epsilon * P - 2*self.sm_value + np.sqrt((epsilon*P - 2*self.sm_value)**2 + 4 * epsilon * self.sm_value * P))/(2*epsilon)
        y_1 = (-epsilon * P - 2*self.sm_value + np.sqrt((-epsilon*P - 2*self.sm_value)**2 - 4 * epsilon * self.sm_value * P))/(-2*epsilon)

        slope = (y_2-y_1)/(2*epsilon)

        return y_1,slope

    def set_sm_val(self,val):
        self.sm_value = val

    def calc_linear(self,y_hat,fix_soc=True):

        lambda_hat = y_hat[0]

        d = np.zeros_like(lambda_hat.cpu().detach().numpy())
        c = np.zeros_like(lambda_hat.cpu().detach().numpy())
        mu = np.zeros_like(lambda_hat.cpu().detach().numpy())

        lambda_hat_np = lambda_hat.cpu().detach().numpy()


        try:
            _ = lambda_hat.size()[1]

            for i in range(lambda_hat.size()[0]):

                if fix_soc:
                    self.op_RN.params[0].value = y_hat[1]
                self.op_RN.params[1].value = lambda_hat_np[i, :]
                self.op_RN.prob.solve(solver=cp.GUROBI)

                d[i,:] = self.op_RN.vars[1].value
                c[i,:] = self.op_RN.vars[2].value

                for j in range(self.OP_params_dict['lookahead']):
                    if j == 0:
                        mu[i, j] = self.op_RN.constraints[6].dual_value[j]
                    else:
                        mu[i, j] = self.op_RN.constraints[7].dual_value[j]

        except:

            self.params.value = lambda_hat_np[:]
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
        #

        return [d,c,mu]

    def calc_linear_batched(self, y_hat, fix_soc=True):
        lambda_hat = y_hat[0]
        net_sched = np.zeros_like(lambda_hat.cpu().detach().numpy())
        mu = np.zeros_like(lambda_hat.cpu().detach().numpy())

        lambda_hat_np = lambda_hat.cpu().detach().numpy()
        batch_size = lambda_hat.size()[0]

        if fix_soc:
            self.op_RN.params[0].value = y_hat[1]
        self.op_RN.params[1].value = lambda_hat_np

            # Solve the optimization problem
        self.op_RN.prob.solve(solver=cp.GUROBI)

        # Extract net_sched
        net_sched = self.op_RN.vars[0].value

        # Extract dual values
        for i in range(self.bs):

            mu[i, 0] = self.op_RN.constraints[2*i+7].dual_value[0]
            for j in range(1, self.OP_params_dict['lookahead']):
                mu[i, j] = self.op_RN.constraints[2*i+8].dual_value[j]

        return [net_sched, mu]

    def repair_decisions(self,d,c,price,soc=None):

        #overshoot, undershoot, cyclic_bc_inf = self.calculate_infeasibility(d.cpu(), c.cpu())
        #print(f"Before repair: Overshoot: {overshoot}, undershoot: {undershoot}, cyclic_bc: {cyclic_bc_inf}")

        s_prov = torch.zeros_like(d)
        s = torch.zeros_like(d)

        s_prov[:,0] = self.OP_params_dict['soc_0'] + c[:,0] - d[:,0]
        d[:,0],c[:,0] = self.constrain_decisions(s_prov[:,0],d[:,0],c[:,0])
        if soc is None:
            s[:,0] = self.OP_params_dict['soc_0'] + c[:,0] - d[:,0]
        else:
            s[:,0] = soc[:,0] + c[:,0] - d[:,0]

        if self.OP_params_dict['constrain_decisions'] == 'greedy':
            for i in range(s.shape[1]-1):
                s_prov[:,i+1] = s[:,i] + c[:,i+1] - d[:,i+1]
                d[:,i+1], c[:,i+1] = self.constrain_decisions(s_prov[:,i+1], d[:,i+1], c[:,i+1])
                s[:,i+1] = s[:,i] + c[:,i+1] - d[:,i+1]
        elif self.OP_params_dict['constrain_decisions'] == 'avg':
            last_extr = -np.ones(s.shape[0])
            for i in range(s.shape[1]-1):
                s_prov[:,i+1] = s_prov[:,i] + c[:,i+1] - d[:,i+1]
                d,c,s_prov,last_extr = self.constrain_decisions_average(s_prov,d,c,last_extr,i)
            s = s_prov
        elif self.OP_params_dict['constrain_decisions'] in ['priority','rescale']:
            s_prov = torch.ones((d.shape[0],d.shape[1]+1)).to(d.device) * self.OP_params_dict['soc_0']
            last_extr = np.zeros(d.shape[0])
            for i in range(d.shape[1]):
                s_prov[:,i+1] = s_prov[:,i] + c[:,i] - d[:,i]
                if i > 0:
                    d,c,s_prov,last_extr = self.constrain_decisions_priority(s_prov.cpu(),d.cpu(),c.cpu(),last_extr,i,price,self.OP_params_dict['constrain_decisions'])

            s = s_prov[:,1:]


        else:
            for i in range(s.shape[1]-1):
                s[:,i+1] = s[:,i] + c[:,i+1] - d[:,i+1]

        #overshoot, undershoot, cyclic_bc_inf = self.calculate_infeasibility(d.cpu(), c.cpu())
        #print(f"After repair: Overshoot: {overshoot}, undershoot: {undershoot}, cyclic_bc: {cyclic_bc_inf}")

        if self.OP_params_dict['cyclic_bc']:
            if self.OP_params_dict['restore_cyclic_bc'] == 'rescale':
                d,c = self.rescale_decisions_cyclicBC(d,c)
            elif self.OP_params_dict['restore_cyclic_bc'] == 'greedy':
                d,c = self.restore_decisions_vectorized(d,c)
            else:
                raise ValueError(f"{self.OP_params_dict['restore_cyclic_bc']} unsupported way of restoring cyclic boundary conditions")

        #overshoot, undershoot, cyclic_bc_inf = self.calculate_infeasibility(d.cpu(), c.cpu())
        #print(f"After rescale 2: Overshoot: {overshoot}, undershoot: {undershoot}, cyclic_bc: {cyclic_bc_inf}")

        return d,c

    def calculate_infeasibility(self,d,c):
        s = torch.zeros_like(d)
        overshoot = torch.zeros(d.shape[0])
        undershoot = torch.zeros(d.shape[0])
        cyclic_bc_inf = torch.zeros(d.shape[0])

        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if j == 0:
                    s[i,j] = self.OP_params_dict['soc_0'] + c[i,j] - d[i,j]
                else:
                    s[i,j] = s[i,j-1] + c[i,j] - d[i,j]

                if s[i,j] > self.OP_params_dict['max_soc']:
                    overshoot[i] += s[i,j] - self.OP_params_dict['max_soc']
                elif s[i,j] < self.OP_params_dict['min_soc']:
                    undershoot[i] +=  self.OP_params_dict['min_soc'] - s[i,j]

            if self.OP_params_dict['cyclic_bc']:
                cyclic_bc_inf[i] = torch.abs(s[i,j]-self.OP_params_dict['soc_0'])

        return torch.sum(overshoot).item(), torch.sum(undershoot).item(), torch.sum(cyclic_bc_inf).item()

    def constrain_decisions_average(self,s_prov, d, c, last_extr, la):

        epsilon = self.OP_params_dict['max_charge']/1000

        def rescale_decisions(d, c, s_prov):
            # c and d are the decisions taken between the last extreme value and this one

            if s_prov[-1] > self.OP_params_dict['max_soc']:
                overshoot = s_prov[-1] - self.OP_params_dict['max_soc']
                rescale_factor = (torch.sum(c) - overshoot) / torch.sum(c)
                c *= rescale_factor
            else:
                overshoot = self.OP_params_dict['min_soc'] - s_prov[-1]
                rescale_factor = (torch.sum(d) - overshoot) / torch.sum(d)
                d *= rescale_factor

            return d, c

        def recalculate_soc(d, c, s,e):
            # tensors d,c: length N; tensor s: lenght s, starting with the value of the previous extreme
            if e == -1:
                s = torch.zeros(d.shape[0]+1)
                s[0] = self.OP_params_dict['soc_0']
            for la in range(d.shape[0]):
                s[la + 1] = s[la] + c[la] - d[la]  # Notice that the arrays have shifted here
            return s[1:]

        for i in range(s_prov.shape[0]):
            extreme_below_min_soc = (s_prov[i, la] < self.OP_params_dict['min_soc'] - epsilon) & (s_prov[i, la + 1] >= s_prov[i, la])
            extreme_above_max_soc = (s_prov[i, la] > self.OP_params_dict['max_soc'] + epsilon) & (s_prov[i, la + 1] <= s_prov[i, la])
            if extreme_above_max_soc or extreme_below_min_soc:
                e = int(last_extr[i])
                last_extr[i] = la
                d[i, e + 1:la+1], c[i, e + 1:la+1] = rescale_decisions(d[i, e + 1:la+1], c[i, e + 1:la+1],s_prov[i,e+1:la+1])
                s_prov[i, e + 1:la+1] = recalculate_soc(d[i, e + 1:la+1], c[i, e + 1:la+1], s_prov[i, e:la+1],e)
                s_prov[i, la+1] = s_prov[i,la] + c[i,la+1] - d[i,la+1]

        return d,c,s_prov,last_extr

    def constrain_decisions_priority(self,s_prov, d, c, last_extr, la, price,type):

        epsilon = self.OP_params_dict['max_charge']/10000

        def rescale_decisions(d, c, s_prov):
            # c and d are the decisions taken between the last extreme value and this one

            epsilon = 1e-6

            if s_prov[-1] > self.OP_params_dict['max_soc']:
                overshoot = s_prov[-1] - self.OP_params_dict['max_soc']
                rescale_factor = (torch.sum(c) - overshoot) / (torch.sum(c) + epsilon)
                c *= rescale_factor
            else:
                overshoot = self.OP_params_dict['min_soc'] - s_prov[-1]
                rescale_factor = (torch.sum(d) - overshoot) / (torch.sum(d) + epsilon)
                d *= rescale_factor

            return d, c

        def recalculate_soc(d, c, s):
            # tensors d,c: length N; tensor s: lenght s, starting with the value of the previous extreme

            for la in range(d.shape[0]):
                s[la + 1] = s[la] + c[la] - d[la]  # Notice that the arrays have shifted here
            return s

        def prioritize_decisions(d, c, s_prov, price):
            epsilon = 1e-6
            # Calculate overshoot or undershoot
            if s_prov[-1] > self.OP_params_dict['max_soc']:
                overshoot = s_prov[-1] - self.OP_params_dict['max_soc']
                # Sort p to get indices for prioritizing d adjustments (higher price first for discharging)
                sorted_indices = torch.argsort(price,descending=True)
                for idx in sorted_indices:
                    # Calculate the adjustment needed
                    adjustment = torch.min(c[idx].clone(), overshoot.clone())
                    c[idx] = c[idx] -  adjustment
                    overshoot = overshoot - adjustment
                    # Break if no more adjustment is needed
                    if overshoot <= epsilon:
                        break
            elif s_prov[-1] < self.OP_params_dict['min_soc']:
                undershoot = self.OP_params_dict['min_soc'] - s_prov[-1]
                # Sort p to get indices for prioritizing c adjustments (lower price first for charging)
                sorted_indices = torch.argsort(price)
                for idx in sorted_indices:
                    # Calculate the adjustment needed
                    adjustment = torch.min(d[idx].clone(), undershoot.clone())
                    d[idx] = d[idx] - adjustment
                    undershoot = undershoot - adjustment
                    # Break if no more adjustment is needed
                    if undershoot <= epsilon:
                        break

            return d, c

        for i in range(s_prov.shape[0]):

            extr_low = (s_prov[i,la-1] > s_prov[i,la]) & (s_prov[i,la] <= s_prov[i,la+1])
            extr_high = (s_prov[i,la-1] < s_prov[i,la]) & (s_prov[i,la] >= s_prov[i,la+1])

            if extr_low or extr_high:
                e = int(last_extr[i])
                last_extr[i] = la

                if (s_prov[i,la] > self.OP_params_dict['max_soc'] + epsilon) or (s_prov[i,la] < self.OP_params_dict['min_soc'] - epsilon):

                    if type == 'rescale':
                        d[i, e:la], c[i, e:la] = rescale_decisions(d[i, e:la], c[i, e:la],s_prov[i, e:la+1])
                    elif type == 'priority':
                        d[i, e:la], c[i, e:la] = prioritize_decisions(d[i, e:la], c[i, e:la],s_prov[i, e:la+1],price[i,e:la])

                    s_prov[i, e:la+1] = recalculate_soc(d[i, e :la], c[i,e:la],s_prov[i,e:la + 1])


        return d.to(self.dev),c.to(self.dev),s_prov.to(self.dev),last_extr

    def rescale_decisions_cyclicBC(self,d,c):

        epsilon = self.OP_params_dict['max_charge']/10000

        c_sum = torch.sum(c,dim=-1).unsqueeze(1) + epsilon
        d_sum = torch.sum(d,axis=-1).unsqueeze(1) + epsilon

        mask_overfull = c_sum > d_sum
        mask_underempty = d_sum >= c_sum

        c_rescaled = torch.where(mask_overfull,c*d_sum/c_sum,c)
        d_rescaled = torch.where(mask_underempty,d*c_sum/d_sum,d)

        return d_rescaled,c_rescaled

    def restore_decisions_cyclicBC_greedy(self,d,c):

        epsilon = self.OP_params_dict['max_charge']/10000

        diff = torch.sum(c,dim=-1).unsqueeze(1) - torch.sum(d,axis=-1).unsqueeze(1)
        len = diff.shape[0]

        for i in range(len):
            j=1
            if diff[i] > epsilon:
                while diff[i] > epsilon:
                    correction = min(c[i,-j],diff[i]).item()
                    diff[i] -= correction
                    c[i,-j] -= correction
                    j+=1
            elif diff[i] < -epsilon:
                while diff[i] < -epsilon:
                    correction = min(d[i,-j],-diff[i]).item()
                    diff[i] += correction
                    d[i,-j] -= correction
                    j+=1

        return d,c

    def restore_decisions_vectorized(self, d, c):
        epsilon = self.OP_params_dict['max_charge'] / 1000
        diff = torch.sum(c, dim=-1) - torch.sum(d, dim=-1)

        # Ensure diff is at least 2D (N x 1) for broadcasting
        diff = diff.unsqueeze(-1)

        # Reverse tensors for backward cumulative operation
        c_rev = c.flip(dims=[-1])
        d_rev = d.flip(dims=[-1])

        # Compute reverse cumulative sums to represent the total correction possible at each step
        c_cum_rev = torch.cumsum(c_rev, dim=-1)
        d_cum_rev = torch.cumsum(d_rev, dim=-1)

        # Calculate masks based on diff
        mask_c = (diff > epsilon) & (c_cum_rev - c_rev < diff)
        mask_d = (diff < -epsilon) & (d_cum_rev - d_rev < -diff)

        # Apply corrections where masks are True
        correction_c_rev = torch.where(mask_c, c_rev, torch.tensor(0.0, device=c.device)).unsqueeze(-1)
        correction_d_rev = torch.where(mask_d, d_rev, torch.tensor(0.0, device=d.device)).unsqueeze(-1)

        # Avoid division by zero by adding a small epsilon to the denominator
        safe_denom_c = torch.sum(correction_c_rev, dim=1) + 1e-8
        safe_denom_d = torch.sum(correction_d_rev, dim=1) + 1e-8

        correction_factor_c = (torch.sum(correction_c_rev,dim=1)-diff)/safe_denom_c
        correction_factor_d = (torch.sum(correction_d_rev, dim=1) + diff) / safe_denom_d

        c_rev = torch.where(mask_c,c_rev*correction_factor_c,c_rev)
        d_rev = torch.where(mask_d,d_rev*correction_factor_d,d_rev)

        c_corrected = c_rev.flip(dims=[-1])
        d_corrected = d_rev.flip(dims=[-1])

        return d_corrected, c_corrected

    def constrain_decisions(self,s_prov_ts,d_ts,c_ts,soc_f=None):

        if soc_f is None:
            lim_high = self.OP_params_dict['max_soc']
            lim_low = self.OP_params_dict['min_soc']
        else:
            lim_high = soc_f
            lim_low = soc_f

        mask_overfull = s_prov_ts > lim_high
        mask_underempty = s_prov_ts < lim_low

        c = torch.where(mask_overfull,c_ts-(s_prov_ts-lim_high),c_ts)
        d = torch.where(mask_underempty,d_ts-(lim_low-s_prov_ts),d_ts)

        return d,c

    def get_opti_outcome(self,y_hat,fix_soc=True,smooth=True,to_torch=True):

        if smooth:
            solver = cp.ECOS
        else:
            solver = cp.ECOS

        if isinstance(y_hat,list):
            lambda_hat = y_hat[0]
        else:
            lambda_hat = y_hat

        d = np.zeros_like(lambda_hat.cpu().detach().numpy())
        c = np.zeros_like(lambda_hat.cpu().detach().numpy())
        mu = np.zeros_like(lambda_hat.cpu().detach().numpy())

        lambda_hat_np = lambda_hat.cpu().detach().numpy()

        if smooth:
            prob = self.op_sm
            env={}
        else:
            prob = self.op_RN
            env = {
                "output_flag": 0
            }

        try:
            _ = lambda_hat.size()[1]

            for i in range(lambda_hat.size()[0]):

                if fix_soc:
                    prob.params[0].value = self.OP_params_dict['soc_0']
                prob.params[1].value = lambda_hat_np[i, :]

                try:
                    prob.prob.solve(solver=solver,env=env)
                except:
                    prob.prob.solve(solver=cp.SCS)

                d[i, :] = prob.vars[1].value
                c[i, :] = prob.vars[2].value

                for j in range(self.OP_params_dict['lookahead']):
                    if j == 0:
                        mu[i, j] = prob.constraints[6].dual_value[j]
                    else:
                        mu[i, j] = prob.constraints[7].dual_value[j]

        except Exception as e:

            print("An error occured in optimizing: ", e)

            if isinstance(prob.params,list):
                prob.params[0].value = self.OP_params_dict['soc_0']
                prob.params[1].value = lambda_hat_np[:]
            else:
                prob.params.value = lambda_hat_np[:]

            prob.op.solve(solver=solver)

            list_keys = list(self.op.var_dict.keys())

            d[:] = torch.from_numpy(self.op.var_dict[list_keys[0]].value)
            c[:] = torch.from_numpy(self.op.var_dict[list_keys[1]].value)
            mu[0] = self.op.constraints[6].dual_value[0]
            for j in range(self.OP_params_dict['lookahead']):
                if j == 0:
                    mu[j] = self.op.constraints[6].dual_value[j]
                else:
                    mu[j] = self.op.constraints[7].dual_value[j]
        #

        if to_torch:
            d = torch.tensor(d).to(self.dev)
            c = torch.tensor(c).to(self.dev)
            mu = torch.tensor(mu).to(self.dev)


        return [d, c, mu]

class NeuralNet(torch.nn.Module):
    def __init__(self, nn_params):
        super(NeuralNet, self).__init__()

        self.input_feat = nn_params['input_size_d'] * nn_params['decoder_seq_length']
        self.output_dim = nn_params['output_dim']*nn_params['decoder_seq_length']
        self.list_units, self.list_act = limit_size_units_act(nn_params['list_units'], nn_params['list_act'])

        self.dev = nn_params['dev']

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
                self.hidden_layers.append(torch.nn.Linear(self.input_feat,units).to(self.dev))
            else:
                self.hidden_layers.append(torch.nn.Linear(self.list_units[i-1],units).to(self.dev))

            self.act_fcts.append(dict_act_fcts[self.list_act[i]])

        if len(self.list_units)>0:
            self.final_layer = torch.nn.Linear(self.list_units[-1], self.output_dim).to(self.dev)
        else:
            self.final_layer = torch.nn.Linear(self.input_feat, self.output_dim).to(self.dev)

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
        self.layers_e = nn_params['layers']
        self.layers_d = nn_params['layers']
        self.hidden_size_lstm = nn_params['hidden_size_lstm']  # hidden state
        self.output_dim = nn_params['output_dim']
        self.dev = nn_params['dev']

        self.lstm_e = torch.nn.LSTM(input_size=self.input_size_e, hidden_size=self.hidden_size_lstm, num_layers=self.layers_e,
                                    batch_first=True,bidirectional=False).to(self.dev)  # Encoder
        self.lstm_d = torch.nn.LSTM(input_size=self.input_size_d, hidden_size=self.hidden_size_lstm, num_layers=self.layers_d,
                                    batch_first=True,bidirectional=False).to(self.dev)  # Decoder
        self.fc = torch.nn.Linear(self.hidden_size_lstm, self.output_dim).to(self.dev) # fully connected 1

        self.list_units,self.list_act = limit_size_units_act(nn_params['list_units'],nn_params['list_act'])



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

        for i,act in enumerate(self.act_fcts):

            output_d = act(self.hidden_layers[i](output_d))

        out = torch.squeeze(self.final_layer(output_d))
        return out




class Attention(nn.Module):
    def __init__(self, hidden_size,dev):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.v.data.normal_(mean=0, std=1. / hidden_size**0.5)

        self.to(dev)

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
        self.nn_params = nn_params
        self.input_size_e = nn_params['input_size_e']  # input size
        self.input_size_d = nn_params['input_size_d']  # input size
        self.layers_e = nn_params['layers']
        self.layers_d = nn_params['layers']
        self.hidden_size_lstm = nn_params['hidden_size_lstm']  # hidden state
        self.output_dim = nn_params['output_dim']
        self.dev = nn_params['dev']
        self.do = nn_params['dropout']

        self.dropout = torch.nn.Dropout(self.do)

        self.lstm_e = nn.LSTM(input_size=self.input_size_e, hidden_size=self.hidden_size_lstm, num_layers=self.layers_e,
                              batch_first=True, bidirectional=False,dropout=self.do).to(self.dev)  # Encoder
        self.lstm_d = nn.LSTM(input_size=self.input_size_d+self.hidden_size_lstm, hidden_size=self.hidden_size_lstm, num_layers=self.layers_d,
                              batch_first=True, bidirectional=False,dropout=self.do).to(self.dev)  # Decoder
        self.attention = Attention(self.hidden_size_lstm,self.dev)


        self.list_units, self.list_act = limit_size_units_act(nn_params['list_units'], nn_params['list_act'])

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

        encoder_outputs = self.dropout(encoder_outputs)

        # Attention mechanism
        attn_weights = self.attention(h_e, encoder_outputs) #dim attn_weights: [bs,la]
        context = attn_weights.bmm(encoder_outputs)  # dim context: [bs,1,hidden]
        context = context.repeat(1, x_d.size(1), 1)

        context = self.dropout(context)

        # Concatenate the context vector with the decoder input
        x_d = torch.cat([x_d, context], dim=2)

        output_d, (h_d, c_d) = self.lstm_d(x_d, (h_e, c_e))

        output_d = self.dropout(output_d)

        for i,act in enumerate(self.act_fcts):

            output_d = act(self.hidden_layers[i](output_d))
            output_d = self.dropout(output_d)

        out = torch.squeeze(self.final_layer(output_d))

        return out

    def set_device(self,dev):
        self.dev = dev
        self.lstm_e.to(dev)
        self.lstm_d.to(dev)
        for l in self.hidden_layers:
            l.to(dev)

        self.attention.to(dev)
        self.attention.v = nn.Parameter(self.attention.v.data.to(dev))
        self.final_layer.to(dev)


class ED_Transformer(nn.Module):
    def __init__(self,nn_params):
        super(ED_Transformer, self).__init__()

        self.encoder_seq_length = nn_params['encoder_seq_length']
        self.decoder_seq_length = nn_params['decoder_seq_length']
        self.decoder_size = nn_params['encoder_size']
        self.encoder_size = nn_params['encoder_size']
        self.num_heads = nn_params['num_heads']
        self.num_layers = nn_params['layers']
        self.ff_dim = nn_params['ff_dim']
        self.dropout = nn_params['dropout']
        self.dev = nn_params['dev']
        self.output_size = nn_params['output_dim']
        self.input_size_e = nn_params['input_size_e']
        self.input_size_d = nn_params['input_size_d']

        #Convert input to right dimension
        self.fc_input_e = nn.Linear(self.input_size_e, self.encoder_size).to(self.dev)
        self.fc_input_d = nn.Linear(self.input_size_d, self.decoder_size).to(self.dev)

        # Transformer Encoder
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=self.encoder_size, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=self.num_layers).to(self.dev)

        # Transformer Decoder
        self.decoder_layers = nn.TransformerDecoderLayer(d_model=self.decoder_size, nhead=self.num_heads, dim_feedforward=self.ff_dim, dropout=self.dropout,batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=self.num_layers).to(self.dev)

        # Fully connected layer for output
        self.fc = nn.Linear(self.decoder_size, self.output_size).to(self.dev)

    def forward(self, x):
        encoder_input = self.fc_input_e(x[0])
        decoder_input = self.fc_input_d(x[1])
        # Ensure the input sequence lengths match the specified lengths
        assert encoder_input.size(1) == self.encoder_seq_length, "Encoder input sequence length mismatch"
        assert decoder_input.size(1) == self.decoder_seq_length, "Decoder input sequence length mismatch"

        # Forward pass through the transformer encoder
        encoder_output = self.transformer_encoder(encoder_input)

        # Forward pass through the transformer decoder
        decoder_output = self.transformer_decoder(decoder_input, encoder_output)

        # Apply fully connected layer for final prediction
        output = torch.squeeze(self.fc(decoder_output))

        return output




#Overarching functions

def limit_size_units_act(units,act_fcts):

    cut_len = min(len(units),len(act_fcts))

    return units[0:cut_len],act_fcts[0:cut_len]

