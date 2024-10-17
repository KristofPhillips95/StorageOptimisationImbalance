import os
import sys
import random
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

experiment_dir = os.path.join(current_dir, '..', '..', 'experiment')
sys.path.insert(0,experiment_dir)

import torch
from torch.autograd import gradcheck
import copy
import h5py
import numpy as np
import time
import loss_fcts_DFL as lf
import math
import gzip
import shutil
import torch_classes as ct
#import forecaster_models as fm

# import nn_classes as nnc
# from nn_classes import RNN
import pickle
import loss_fcts_DFL
import nn_with_opti as nnwo
#import functions_support as sf


class Train_model():
    def __init__(self, nn_params=None, training_params=None, OP_params=None, data_params=None, data=None, loc=None, config=None):

        super(Train_model,self).__init__()


        #Initialize model based on provided parameters or location
        if (nn_params is not None)&(training_params is not None)&(OP_params is not None) & (data_params is not None):
            self.set_initialization(nn_params,training_params,OP_params,data_params,data)
        elif (loc is not None) & (config is not None):
            self.load_model(loc,config)
        else:
            raise ValueError("Unsupported initialization. Provide arguments nn_params,training_params,OP_params and data_dict, or loc and config")

    def set_initialization(self,nn_params,training_params,OP_params,data_dict,data):
        self.nn_params = nn_params
        self.OP_params = OP_params
        self.training_params = training_params

        #price_gen = self.init_price_gen(nn_params)

        self.forecaster = nnwo.Forecaster(nn_params=nn_params,
                                            OP_params=OP_params,
                                            training_params=training_params,
                                            reg_type=training_params['reg_type'],
                                            reg_val=training_params['reg'])

        if OP_params['overwrite_from_proxy']:
            self.preprocess_OP_params()

        self.data_dict = data_dict
        if data is not None:
            self.load_data(data)

        if self.forecaster.include_opti:
            self.set_sched_calculator()

    def load_data(self,data=None):
        if data == None:
            # Load data and split features/labels in numpy arrays
            features_train, features_val, features_test, price_train, price_val, price_test, price_fc_list, price_fc_scaled_list = sf.preprocess_data(
                self.data_dict)

            # Get optimal schedules if required
            labels_train, labels_val, labels_test = sf.preprocess_labels(self.training_params, price_train, price_val, price_test,
                                                                         self.OP_params, self.data_dict)

            # Convert to tensors
            list_tensors = sf.get_train_validation_tensors(features_train, labels_train, features_val, labels_val,
                                                           features_test, labels_test)

            data = {
                'train': (list_tensors[0],list_tensors[1]),
                'val': (list_tensors[2],list_tensors[3]),
                'test': (list_tensors[4], list_tensors[5]),
            }

        self.data = data
        self.set_training_loader()

    def set_training_loader(self,set_seed=True):
        if set_seed:
            if self.data_dict['sensitivity']:
                torch.manual_seed(73+self.OP_params['add_seed'])
                #torch.manual_seed(73)
                print(f'seed {self.OP_params["add_seed"]} added')
            else:
                torch.manual_seed(73)



        train_Dataset = loss_fcts_DFL.Dataset_Lists(self.data['train'][0],self.data['train'][1])
        self.training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=self.training_params['batch_size'], shuffle=True)

    def set_sched_calculator(self):

        self.sched_calculator_mu = nnwo.Schedule_Calculator(OP_params_dict=self.OP_params,fw=self.training_params['framework'],dev=self.training_params['device'])
        self.sched_calculator_mu_ID = nnwo.Schedule_Calculator(OP_params_dict=self.OP_params,fw='ID',dev=self.training_params['device'])

        params_RN = copy.deepcopy(self.OP_params)
        params_RN['gamma'] = 0
        self.optiLayer_RN = nnwo.OptiLayer(params_RN)

    def load_OP_dict(self,name):
        try:
            loc = 'OP/'+name
            with open(loc, 'rb') as fp:
                dict_OP = pickle.load(fp)
        except:
            loc = '../ML_proxy/OP'+name
            with open(loc, 'rb') as fp:
                dict_OP = pickle.load(fp)

        return dict_OP

    def set_loss_fct(self):
        l = self.training_params['loss_fct_str']
        self.loss_fct_str = l
        # if l in ['profit','mse','mae','mse_weighted','pinball','generalized']:
        #     #self.loss_fct = ct.Loss(l,self.training_params['loss_params'])
        #     self.training_params['loss_params']['type'] = l
        #     self.loss_fct = ct.LossNew(self.training_params['loss_params'])
        # else:
        #     raise ValueError(f"Loss function {l} not supported")

        self.loss_fct = define_loss(l,self.training_params)

        #self.feas_loss = ct.LossFeasibilitySoc(self.OP_params,self.training_params)

    def set_best_net(self):
        #Function ensuring that when new optimum is found, that specific forecaster is not overwritten by subsequent gradient updates

        self.best_net = self.forecaster.make_clone()

    def set_config(self,c):
        self.config = c

    def get_config(self):
        return self.config

    def get_best_loss(self):
        return self.loss_best

    def init_loss(self):
        self.set_loss_fct()
        self.loss_fcts_eval_str = self.training_params['loss_fcts_eval_str']
        self.required_fc_output, self.required_labels = get_required_labels(self.loss_fcts_eval_str)

        # Initialize loss functions for evaluating performance
        self.loss_fcts_eval = {}
        self.loss_evolution = {}
        if self.training_params['include_loss_evol_smooth']:
            self.loss_evolution_smooth = {}
        self.loss_best = {}

        for loss_str in self.loss_fcts_eval_str:
            self.loss_evolution[loss_str] = [[], [], []]
            if self.training_params['include_loss_evol_smooth']:
                self.loss_evolution_smooth[loss_str] = [[], [], []]

            # params_loss = copy.deepcopy(self.training_params['loss_params'])
            # params_loss['type'] = loss_str
            # self.loss_fcts_eval[loss_str] = ct.LossNew(params_loss)
            self.loss_fcts_eval[loss_str] = define_loss(loss_str,self.training_params)

        self.train_loss_insample = []

        if self.training_params['keep_prices_train']:
            self.train_prices_evol = [self.data['train'][1][0].cpu().detach().numpy()]
        if self.training_params['keep_sched_train']:
            self.train_sched_evol = [self.sched_calculator_mu(self.data['train'][1][0],smooth=False)[0].cpu().detach().numpy()]

    def update_best_loss(self,tr,va,te):
        self.loss_best = [tr,va,te]

    def update_losses_set(self,set):

        if self.training_params['all_validation'] or set == 'val':
            #This if-else construct allows to only validate the validation set if chosen (self.training_params['all_validation'] == False)

            yhat_set,yhat_check = self.calc_fc(set)
            labels_set = self.calc_labels(set)


            for loss_str in self.loss_fcts_eval_str:
                if (loss_str == 'mse_mu') & (set == 'train'):
                    x=1
                l_set = self.loss_fcts_eval[loss_str](yhat_set, labels_set).item()
                self.loss_evolution[loss_str][set_str_to_int(set)].append(l_set)

        else:

            for loss_str in self.loss_fcts_eval_str:
                self.loss_evolution[loss_str][set_str_to_int(set)].append(0)

    def calc_fc(self,set):

        yhat = []
        yhat_check = []
        price = self.calc_prediction_batched(self.data[set][0])[0]

        if (set == 'train') & (self.training_params['keep_prices_train']):
            self.train_prices_evol.append(price.cpu().detach().numpy())

        if 'price' in self.required_fc_output:
            yhat.append(price)
            yhat_check.append(price)

        if 'sched' in self.required_fc_output:
            #sched,_,_ = self.sched_calculator_mu(price,smooth=False)
            if self.training_params['MPC']:
                opti_input = [price,self.data[set][0][2]]
                #TODO: check if this is correct??????
            else:
                opti_input = [price]
                #soc_0 = torch.tensor([self.OP_params['soc_0']], dtype=torch.float32).to(self.training_params['device'])
            sched, _ = self.optiLayer_RN(opti_input)
            yhat.append(sched)
            yhat_check.append(sched)

        if ('sched_sm' in self.required_fc_output) or ('mu' in self.required_fc_output):
            sched_sm,_,mu = self.sched_calculator_mu(price,smooth=True)
            #sched_sm_ID,_,mu_ID = self.sched_calculator_mu_ID(price,smooth=True)

            if "sched_sm" in self.required_fc_output:
                yhat.append(sched_sm)
                #yhat_check.append([sched_sm,sched_sm_ID])
                if (set == 'train') & (self.training_params['keep_sched_train']):
                    self.train_sched_evol.append(sched_sm.cpu().detach().numpy())
            if "mu" in self.required_fc_output:
                yhat.append(mu)
                #yhat_check.append([mu,mu_ID])

        return yhat,yhat_check

    def calc_labels(self,set):

        labels =  copy.deepcopy(self.data[set][1]) #Gives [price]

        if ('sched' in self.required_labels) or ('mu' in self.required_labels):
            sched,_,mu = self.sched_calculator_mu(labels[0],smooth=False)
            if "sched" in self.required_labels:
                labels.append(sched)
            if "mu_act" in self.required_labels:
                labels.append(mu)

        if 'mu_sm' in self.required_labels:

            price_fc = self.calc_prediction_batched(self.data[set][0])[0]
            _,_,mu_sm = self.sched_calculator_mu(price_fc,smooth=False)
            labels.append(mu_sm)

        return labels
    #
    # def calc_loss_all(self,loss_str,smooth=False):
    #
    #     train_loss = self.calc_loss_single_set(loss_str,'train',smooth)
    #     val_loss = self.calc_loss_single_set(loss_str,'val',smooth)
    #     test_loss = self.calc_loss_single_set(loss_str,'test',smooth)
    #
    #     return [train_loss,val_loss,test_loss]
    #
    # def calc_loss_single_set(self,loss_str,set,smooth=False):
    #
    #     #price = self.forecaster.get_price(self.data[set][0])
    #     price = self.calc_prediction_batched(self.data[set][0])[0]
    #     if (set == 'train')&(self.training_params['keep_prices_train']):
    #         self.train_prices_evol.append(price.cpu().detach().numpy())
    #
    #     labels = self.data[set][1]
    #
    #     if self.loss_fcts_eval[loss_str].requires_mu:
    #         if set == 'train':
    #             x=1
    #         yhat = [price, *self.sched_calculator_mu(price)]
    #         #yhat = [price, *self.sched_calculator_mu([price,self.OP_params['soc_0']])]
    #         _,_,mu_labels = self.sched_calculator_mu(price,smooth=False)
    #         labels += [mu_labels]
    #     else:
    #         if smooth:
    #             yhat = self.forecaster(self.data[set][0])
    #         else:
    #             yhat = [price,self.calculate_schedule_from_price(price,set)]
    #
    #     loss = self.loss_fcts_eval[loss_str](yhat, labels).item()
    #
    #     print(f"Loss {loss_str} for set {set}: {loss}")
    #
    #     return loss

    def update_loss_evol(self):

        #TODO: clean up

        if self.forecaster.include_opti:

            for set in ['train', 'val', 'test']:
                self.update_losses_set(set)

        else:

            yhat_train = self.calc_prediction_batched(self.data['train'][0])
            yhat_val = self.calc_prediction_batched(self.data['val'][0])
            yhat_test = self.calc_prediction_batched(self.data['test'][0])

            for loss_str in self.loss_fcts_eval_str:
                self.loss_evolution[loss_str][0].append(
                    self.loss_fcts_eval[loss_str](yhat_train, self.data['train'][1]).item())
                self.loss_evolution[loss_str][1].append(
                    self.loss_fcts_eval[loss_str](yhat_val, self.data['val'][1]).item())
                self.loss_evolution[loss_str][2].append(
                    self.loss_fcts_eval[loss_str](yhat_test, self.data['test'][1]).item())

        train_loss = self.loss_evolution[self.loss_fct_str][0][-1]
        val_loss = self.loss_evolution[self.loss_fct_str][1][-1]
        test_loss = self.loss_evolution[self.loss_fct_str][2][-1]


        return train_loss,val_loss,test_loss

    def calculate_schedule_from_price(self,price,set):

        #check_optilayer = self.optilayer_RN(price)
        #print('at least one calculation succeeded')

        # if set == 'train':
        #     tic = time.time()
        #     check = self.optiLayer_RN(price)
        #     print(f"Time for optimization with cvxpylayers: {time.time()-tic}")
        #     tic = time.time()
        #     sched = torch.tensor(self.sched_calculator.calc_linear([price,self.OP_params['soc_0']])[0]).to(self.training_params['device'])
        #     print(f"Time for optimization with Gurobi batched: {time.time()-tic}")
        #
        #
        # elif set == 'val':
        #     sched = torch.tensor(self.sched_calculator_val.calc_linear_batched([price, self.OP_params['soc_0']])[0]).to(self.training_params['device'])
        # elif set == 'test':
        #     sched = torch.tensor(self.sched_calculator_val.calc_linear_batched([price, self.OP_params['soc_0']])[0]).to(self.training_params['device'])
        #
        # return sched
        return self.optiLayer_RN(price)[0]

    def train(self):

        optimizer = torch.optim.AdamW(self.forecaster.get_trainable_params(), lr=self.training_params['lr'])
        self.best_net = self.forecaster

        #
        # fc_test = [self.forecaster.nn(self.data['test'][0])]
        # lf = self.training_params['loss_check']
        # loss = lf(fc_test,self.data['test'][1])



        self.init_loss()
        tr_l, va_l, te_l = self.update_loss_evol()
        self.loss_best = [tr_l,  va_l,  te_l]

        print(
            f'Epoch {0} \t\t Training Loss: {tr_l} \t\t Validation Loss: {va_l} \t\t Test Loss: {te_l}')

        epochs_since_improvement = 0

        #Keep track of different types of train times
        train_time_forward = 0
        train_time_loss = 0
        train_time_backward = 0
        train_time_val = 0

        time_start = time.time()

        if 'add_seed' in self.OP_params:
            sd = self.OP_params['add_seed']
        else:
            sd = 0


        torch.manual_seed(73+sd)

        for e in range(self.training_params['epochs']):

            if e == 3:
                x=1

            if epochs_since_improvement >= self.training_params['patience']:
                break

            ### TRAINING PHASE ###

            #Keep track of parts of loss function
            train_loss = 0.0
            loss_pure = 0.0
            loss_reg = 0.0
            loss_feas = 0.0
            train_start = time.time()
            for (i, data) in enumerate(self.training_loader):
                tic_start = time.time()
                x, y = data

                # clear gradients
                optimizer.zero_grad()
                # Forward pass
                y_hat = self.forecaster(x)
                tic_start_loss = time.time()
                l1 = self.loss_fct(y_hat,y)
                l2 = self.forecaster.calc_reg()
                loss = l1+l2

                loss_pure_batch = l1.item()
                loss_reg_batch = l2.item()
                loss_pure += loss_pure_batch
                loss_reg += loss_reg_batch
                train_loss += loss_pure_batch+loss_reg_batch

                # if self.training_params['feasibility_loss']:
                #     l3 = self.feas_loss(y_hat[2])
                #     loss += l3
                #     loss_feas_batch = l3.item()
                #     loss_feas += loss_feas_batch
                #     train_loss += loss_feas_batch

                tic_start_backward = time.time()

                loss.backward()
                # Update weights

                #torch.nn.utils.clip_grad_norm_(self.forecaster.get_trainable_params(), max_norm=1)

                optimizer.step()

                tic_end = time.time()

                train_time_forward += tic_start_loss-tic_start
                train_time_loss += tic_start_backward - tic_start_loss
                train_time_backward += tic_end-tic_start_backward

            train_time = time.time() - train_start

            self.train_loss_insample.append(train_loss)

            print(f"Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Train time: {train_time} \t\t Loss fct: {loss_pure}; reg loss: {loss_reg}; feas loss: {loss_feas}")

            ### VALIDATION PHASE ###

            train_loss,val_loss,test_loss = self.update_loss_evol()

            #price_fc = self.forecaster(self.data['val'][0])[0].cpu().detach().numpy()
            #actual_price = self.data['val'][1][0].cpu().detach().numpy()

            print(f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss} \t\t Test Loss: {test_loss}')

            if self.loss_best[1] > val_loss:
                print(f'Validation Loss Decreased({self.loss_best[1]:.6f}--->{val_loss:.6f}) \t Saving The Model')
                print('')
                self.update_best_loss(train_loss,val_loss,test_loss)
                self.set_best_net()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            train_time_val += time.time() - tic_end

        self.tot_train_time = time.time() - time_start
        if not self.training_params['epochs'] > 0:
            e = 0
        self.train_time_per_epoch = self.tot_train_time/(e+1)

        self.time_forward_per_epoch = train_time_forward/(e+1)
        self.time_loss_per_epoch = train_time_loss/(e+1)
        self.time_backward_per_epoch = train_time_backward/(e+1)
        self.time_val_per_epoch = train_time_val/(e+1)

    def save(self,loc,save_data=False,save_evols=True):

        if self.config is None:
            self.set_config(1)

        #loc_torch_ext = f"{loc}/config_{self.config}_best_model_ext.pth"
        loc_torch = f"{loc}/config_{self.config}_best_model.pth"
        #loc_pickle = f"{loc}/config_{self.config}_model_state.pkl"

        torch.save(self.best_net.nn.state_dict(), loc_torch)

        # # Set the model attribute to None or a placeholder before saving
        # model_best_backup = self.best_net.make_clone()
        # model_backup = self.forecaster.make_clone()
        # data_backup = copy.deepcopy(self.data)
        # tl_backup = copy.deepcopy(self.training_loader)
        #
        # self.best_net = None
        # self.forecaster = None
        #
        # if not save_data:
        #     self.data = None
        #     self.training_loader = None
        #
        # if not save_evols:
        #     self.loss_evol = None
        #
        # # Save the rest of the class instance
        # with open(loc_pickle, 'wb') as file:
        #     self.sched_calculator=None #check why you can't save the sched_calculator
        #     self.sched_calculator_mu = None #check why you can't save the sched_calculator
        #     pickle.dump(self, file)

        new_dict = {
            'training_params': self.training_params,
            'data_dict': self.data_dict,
            'nn_params': self.nn_params,
            'OP_params': self.OP_params,
            'loss_evolution': self.loss_evolution,
            'loss_best': self.loss_best,
            'train_loss_insample': self.train_loss_insample,
            'train_times': {'time_back': self.time_backward_per_epoch, 'time_fw': self.time_forward_per_epoch, 'time_loss': self.time_loss_per_epoch, 'time_val': self.time_val_per_epoch, 'tot_train_time': self.tot_train_time},
            'config': self.config
        }

        if self.training_params['include_loss_evol_smooth']:
            new_dict['loss_evolution_smooth'] = self.loss_evolution_smooth

        loc_dict = f"{loc}/config_{self.config}_dict_model.pkl"
        with open(loc_dict, 'wb') as file:
            pickle.dump(new_dict, file)

        # # Restore the model attributes
        # self.best_net = model_best_backup
        # self.forecaster = model_backup
        # self.data = data_backup
        # self.training_loader = tl_backup

    def load_model(self,loc,config,cpu_load=True):
        loc_torch = f"{loc}config_{config}_best_model.pth"
        loc_pickle = f"{loc}config_{config}_model_state.pkl"

        with open(loc_pickle, 'rb') as file:
            saved_instance = pickle.load(file)

        for attr,value in vars(saved_instance).items():
            setattr(self,attr,value)

        # Load the PyTorch model
        if cpu_load:
            self.best_net = torch.load(loc_torch,map_location='cpu')
        else:
            self.best_net = torch.load(loc_torch)
        self.forecaster = self.best_net

    def calc_prediction_batched(self,features,type='price',bs=128):

        all_pred = []

        for start in range(0,features[0].shape[0],bs):
            end = min(start+bs,features[0].shape[0])
            feats_chunk = [f[start:end,...] for f in features]

            with torch.no_grad():
                if type == 'price':
                    fc = [self.forecaster.get_price(feats_chunk)]
                elif type == 'all':
                    fc = self.forecaster(feats_chunk)
                else:
                    ValueError(f"{type} invalid type of calculating outcome in batched fashion")
                all_pred.append(fc)

        output_list = []

        for i in range(len(all_pred[0])):
            out = torch.cat([f[i] for f in all_pred],dim=0)
            output_list.append(out)

        return output_list

    def preprocess_OP_params(self):

        with open(self.OP_params['loc_proxy_params'], 'rb') as file:
            # Load the object from the pickle file
            params_proxy = pickle.load(file)

        for key in params_proxy:
            if params_proxy[key] != self.OP_params[key]:
                print(f"Replacing optimization parameter {key} from value {self.OP_params[key]} --> {params_proxy[key]}")
                self.OP_params[key] = params_proxy[key]





def define_loss(loss_fct_str,training_dict):

    if loss_fct_str == 'profit':

        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'profit',
            'la': training_dict['la'],
            'loc_preds': 1,
            'loc_labels': 0
        }

    elif loss_fct_str == 'mse_mu':

        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'mse',
            'la': training_dict['la'],
            'loc_preds': 3,
            'loc_labels': 2
        }

    elif loss_fct_str == 'mse_price':

        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'mse',
            'la': training_dict['la'],
            'loc_preds': 0,
            'loc_labels': 0
        }

    elif loss_fct_str == 'mae_price':

        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'mae',
            'la': training_dict['la'],
            'loc_preds': 0,
            'loc_labels': 0
        }

    elif loss_fct_str == 'mse_sched':
        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'mse',
            'la': training_dict['la'],
            'loc_preds': 1,
            'loc_labels': 1
        }


    elif loss_fct_str == 'mse_sched_weighted_profit':
        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'mse_weighted_profit',
            'la': training_dict['la'],
            'loc_preds': 1,
            'loc_labels': 1
        }

    elif loss_fct_str == 'mse_sched_first':
        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'mse_first',
            'la': training_dict['la'],
            'loc_preds': 1,
            'loc_labels': 1
        }

    elif loss_fct_str == 'mse_sched_first_weighted_profit':
        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'mse_first_weighted_profit',
            'la': training_dict['la'],
            'loc_preds': 1,
            'loc_labels': 1
        }

    elif loss_fct_str == 'mse_sched_sm':
        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'mse',
            'la': training_dict['la'],
            'loc_preds': 2,
            'loc_labels': 1
        }

    elif loss_fct_str == 'pinball':
        params_loss = {
            'loss_str': loss_fct_str,
            'type': 'pinball',
            'la': training_dict['la'],
            'loc_preds': 0,
            'loc_labels': 0,
            'quantile_tensor': training_dict['loss_params']['quantile_tensor']
        }

    elif loss_fct_str == 'generalized':
        params_loss = training_dict['loss_params']
        params_loss['type'] = 'generalized'



    return ct.LossNew(params_loss)

def get_required_labels(list_loss_fct_str):
    forecast_output = []
    labels = []

    for loss_str in list_loss_fct_str:
        if loss_str == 'profit':
            forecast_output.append('sched')
        elif loss_str == 'mse_mu':
            forecast_output.append('mu')
            labels.append('mu_sm')
        elif loss_str == 'mse_price':
            forecast_output.append('price')
            labels.append('price')
        elif (loss_str == 'mse_sched') or (loss_str == 'mse_sched_weighted_profit'):
            forecast_output.append('sched')
            labels.append('sched')
        elif (loss_str == 'mse_sched_first') or (loss_str == 'mse_sched_first_weighted_profit'):
            forecast_output.append('sched')
            labels.append('sched')
        elif loss_str == 'mse_sched_sm':
            forecast_output.append('sched_sm')
            labels.append('sched')
        else:
            ValueError(f"{loss_str} unsupported")

    return forecast_output,labels

def set_str_to_int(str):
    if str == 'train':
        return 0
    elif str == 'val':
        return 1
    elif str == 'test':
        return 2
    else:
        ValueError(f"{str} unsupported set string")


