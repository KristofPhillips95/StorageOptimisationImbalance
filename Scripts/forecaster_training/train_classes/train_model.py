import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

experiment_dir = os.path.join(current_dir, '..', '..', 'experiment')
sys.path.insert(0,experiment_dir)

import torch
import copy
import h5py
import numpy as np
import time
import loss_fcts as lf
import math
import torch_classes as ct
#import forecaster_models as fm

# import nn_classes as nnc
# from nn_classes import RNN
import pickle
import loss_fcts
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

        if self.forecaster.include_opti:
            self.set_optiLayer_RN()

        self.data_dict = data_dict
        self.load_data(data)

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
            torch.manual_seed(73)

        train_Dataset = loss_fcts.Dataset_Lists(self.data['train'][0],self.data['train'][1])
        self.training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=self.training_params['batch_size'], shuffle=True)

    # def init_price_gen(self,nn_params):
    #     # Function initializing the re-forecaster with a pretrained network if warm start applied
    #     torch.manual_seed(71)
    #
    #     init_net = nnwo.NeuralNet(nn_params=nn_params)
    #
    #     if nn_params['warm_start']:
    #         if nn_params['list_units'] == []:
    #             idx_fc = self.OP_params['feat_cols'].index('y_hat')
    #             with torch.no_grad():
    #                 for i in range(init_net.final_layer.weight.shape[0]):
    #                     init_net.final_layer.bias[i] = 0
    #                     for j in range(init_net.final_layer.weight.shape[1]):
    #                         if j == self.OP_params['n_diff_features'] * i + idx_fc:
    #                             init_net.final_layer.weight[i, j] = 1
    #                         else:
    #                             init_net.final_layer.weight[i, j] = 0
    #
    #         else:
    #
    #             loc = "../../data/pretrained_fc/model_softplus_wd_nlfr_genfc_yhat_scaled_pos/"
    #
    #             weights_layer_1 = torch.tensor(np.load(loc + 'weights_layer_1.npz'), dtype=torch.float32)
    #             biases_layer_1 = torch.tensor(np.load(loc + 'biases_layer_1.npz'), dtype=torch.float32)
    #             weights_layer_2 = torch.tensor(np.load(loc + 'weights_layer_2.npz'), dtype=torch.float32)
    #             biases_layer_2 = torch.tensor(np.load(loc + 'biases_layer_2.npz'), dtype=torch.float32)
    #
    #             with torch.no_grad():
    #                 init_net.hidden_layers[0].weight.copy_(weights_layer_1)
    #                 init_net.hidden_layers[0].bias.copy_(biases_layer_1)
    #                 init_net.final_layer.weight.copy_(weights_layer_2)
    #                 init_net.final_layer.bias.copy_(biases_layer_2)
    #
    #     return init_net

    def set_optiLayer_RN(self):
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
        if l in ['profit', 'mse_sched', 'mse_sched_weighted','mse_price','mae_price','pinball']:
            self.loss_fct = ct.Loss(l,self.training_params['loss_params'])
        else:
            raise ValueError(f"Loss function {l} not supported")

    def set_best_net(self):
        #Function ensuring that when new optimum is found, that specific forecaster is not overwritten by subsequent gradient updates

        # clone = type(self.forecaster)(self.nn_params)
        # clone.load_state_dict(self.nn.state_dict())
        # self.best_net = clone

        # clone_price_gen = type(self.forecaster.price_generator)(self.nn_params)
        # clone_price_gen.load_state_dict(self.forecaster.price_generator.state_dict())
        # clone_nn_with_opti = nnwo.NeuralNetWithOpti(clone_price_gen, self.OP_params, self.nn_params['framework'], self.training_params['reg_type'], self.training_params['reg'])
        # self.best_net = clone_nn_with_opti

        self.bet_net = self.forecaster.make_clone()

        # price_gen = model.price_generator
        # clone_price_gen = type(price_gen)(price_gen.input_feat,price_gen.output_dim,price_gen.list_units,price_gen.list_act)
        # clone_price_gen.load_state_dict(price_gen.state_dict())

    def set_config(self,c):
        self.config = c

    def get_config(self):
        return self.config

    def get_best_loss(self):
        return self.loss_best

    def init_loss(self):
        self.set_loss_fct()
        self.loss_fcts_eval_str = self.training_params['loss_fcts_eval_str']

        # Initialize loss functions for evaluating performance
        self.loss_fcts_eval = {}
        self.loss_evolution = {}
        self.loss_evolution_smooth = {}
        self.loss_best = {}

        for loss_str in self.loss_fcts_eval_str:
            self.loss_best = [np.inf,np.inf,np.inf]
            self.loss_evolution[loss_str] = [[], [], []]
            self.loss_evolution_smooth[loss_str] = [[], [], []]
            self.loss_fcts_eval[loss_str] = ct.Loss(loss_str,self.training_params['loss_params'])

    def update_best_loss(self,tr,va,te):
        self.loss_best = [tr,va,te]

    def update_loss_evol(self):

        #TODO: clean up

        if self.forecaster.include_opti:

            yhat_train_sm = self.forecaster(self.data['train'][0])
            yhat_val_sm = self.forecaster(self.data['val'][0])
            yhat_test_sm = self.forecaster(self.data['test'][0])

            yhat_train_RN = [self.optiLayer_RN(yhat_train_sm[1]),yhat_train_sm[1]]
            yhat_val_RN = [self.optiLayer_RN(yhat_val_sm[1]),yhat_val_sm[1]]
            yhat_test_RN = [self.optiLayer_RN(yhat_test_sm[1]),yhat_test_sm[1]]

            for loss_str in self.loss_fcts_eval_str:
                self.loss_evolution[loss_str][0].append(self.loss_fcts_eval[loss_str](yhat_train_RN, self.data['train'][1]).item())
                self.loss_evolution[loss_str][1].append(self.loss_fcts_eval[loss_str](yhat_val_RN, self.data['val'][1]).item())
                self.loss_evolution[loss_str][2].append(self.loss_fcts_eval[loss_str](yhat_test_RN, self.data['test'][1]).item())

                self.loss_evolution_smooth[loss_str][0].append(self.loss_fcts_eval[loss_str](yhat_train_sm, self.data['train'][1]).item())
                self.loss_evolution_smooth[loss_str][1].append(self.loss_fcts_eval[loss_str](yhat_val_sm, self.data['val'][1]).item())
                self.loss_evolution_smooth[loss_str][2].append(self.loss_fcts_eval[loss_str](yhat_test_sm, self.data['test'][1]).item())

            val_loss =  self.loss_evolution[self.loss_fct_str][1][-1]
            test_loss = self.loss_evolution[self.loss_fct_str][2][-1]

        else:

            yhat_train = self.forecaster(self.data['train'][0])
            yhat_val = self.forecaster(self.data['val'][0])
            yhat_test = self.forecaster(self.data['test'][0])

            for loss_str in self.loss_fcts_eval_str:
                self.loss_evolution[loss_str][0].append(
                    self.loss_fcts_eval[loss_str](yhat_train, self.data['train'][1]).item())
                self.loss_evolution[loss_str][1].append(
                    self.loss_fcts_eval[loss_str](yhat_val, self.data['val'][1]).item())
                self.loss_evolution[loss_str][2].append(
                    self.loss_fcts_eval[loss_str](yhat_test, self.data['test'][1]).item())

            val_loss = self.loss_evolution[self.loss_fct_str][1][-1]
            test_loss = self.loss_evolution[self.loss_fct_str][2][-1]


        return val_loss,test_loss

    def train(self):

        self.init_loss()

        optimizer = torch.optim.AdamW(self.forecaster.get_trainable_params(), lr=self.training_params['lr'])
        self.best_net = self.forecaster

        epochs_since_improvement = 0

        for e in range(self.training_params['epochs']):

            if e%10 == 0:
                print("Epoch 10 reached")

            if epochs_since_improvement >= self.training_params['patience']:
                break

            ### TRAINING PHASE ###
            train_loss = 0.0
            loss_pure = 0.0
            loss_reg = 0.0
            train_start = time.time()
            for (i, data) in enumerate(self.training_loader):
                x, y = data
                # clear gradients
                optimizer.zero_grad()
                # Forward pass
                y_hat = self.forecaster(x)
                loss = self.loss_fct(y_hat, y) + self.forecaster.calc_reg()
                loss.backward()

                # Update weights
                optimizer.step()
                train_loss += loss.item()
                loss_pure += self.loss_fct(y_hat, y).item()
                loss_reg += self.forecaster.calc_reg().item()

            train_time = time.time() - train_start

            print(f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Train time: {train_time}')
            print(f"Loss fct: {loss_pure}; reg loss: {loss_reg}")

            ### VALIDATION PHASE ###

            val_loss,test_loss = self.update_loss_evol()


            print(f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss} \t\t Test Loss: {test_loss}')

            if self.loss_best[1] > val_loss:
                print(f'Validation Loss Decreased({self.loss_best[1]:.6f}--->{val_loss:.6f}) \t Saving The Model')
                self.update_best_loss(train_loss,val_loss,test_loss)
                self.set_best_net()
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

        x=1

    def save(self,loc,save_data=False,save_evols=True):

        if self.config is None:
            self.set_config(1)

        loc_torch = f"{loc}/config_{self.config}_best_model.pth"
        loc_pickle = f"{loc}/config_{self.config}_model_state.pkl"

        torch.save(self.best_net, loc_torch)

        # Set the model attribute to None or a placeholder before saving
        model_best_backup = self.best_net
        model_backup = self.forecaster

        self.best_net = None
        self.nn = None

        if not save_data:
            self.train_data = None
            self.val_data = None
            self.test_data = None
            self.training_loader = None

        if not save_evols:
            self.loss_evol = None

        # Save the rest of the class instance
        with open(loc_pickle, 'wb') as file:
            pickle.dump(self, file)

        # Restore the model attribute
        self.best_net = model_best_backup
        self.forecaster = model_backup

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







