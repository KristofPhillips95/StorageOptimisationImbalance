import os
import copy
import itertools
import train_model as m
import pprint
import pickle
import torch
import pandas as pd
import math
import numpy as np
import csv
import time
from concurrent.futures import ProcessPoolExecutor
#import dill
from joblib import parallel_backend, Parallel, delayed
import concurrent.futures as cf

#TODO: clean up how data is handled

class HPTuner():
    def __init__(self, hp_dict, hp_trans, nn_params, OP_params, training_params, data_params, save_loc, sensitivity=False, delete_models=False, app=None,  data=None, n_configs=None):
        super(HPTuner,self).__init__()

        self.hp_dict = hp_dict
        self.hp_trans = hp_trans
        self.init_strategy()
        self.nn_params = nn_params
        self.OP_params = OP_params
        self.training_params = training_params
        self.data_params = data_params
        self.app = app
        self.data = data
        self.save_loc = save_loc
        self.n_configs = n_configs
        self.delete_models = delete_models
        self.sensitivity = sensitivity
        self.data_params['sensitivity'] = sensitivity

        #check if any changes in used data will occur for the hp tuning. If not, later on in the code the data will be loaded only once (to avoid repeatedly optimizing schedules)
        self.one_dataset = self.check_single_dataset()

        print('HP tuner initialized')

    def __call__(self):
        #Make new dir to save results
        if self.training_params['makedir']:
            os.makedirs(self.save_loc)
            print(f'Dir was created at location {self.save_loc}')
        else:
            print("No dir was created")

        #Create a list of hp dicts
        self.iterable = self.make_iterable_hps()

        #Create list of train_models initialized with the HPs
        self.list_models = self.initialize_models()
        print('Models initialized')

        #Execute the training of all the training models
        self.execute_training(self.training_params['exec'])

        if self.training_params['exec'] == 'par':
            self.save_overview(self.list_models)

        if self.delete_models:
            self.unsave_models()

    def unsave_models(self):

        if self.training_params['exec'] == 'seq':
            if self.app is not None:
                keep_config = np.argmax(self.d['val_val'])
            else:
                keep_config = np.argmin(self.d['val_loss'])
        elif self.training_params['exec'] == 'par':

            df = pd.read_csv(f"{self.save_loc}/output_par.csv")

            if self.app is not None:
                index = df['val_val'].idxmax()
            else:
                index = df['val_loss'].idxmin()

            keep_config = df.loc[index,'config'] - 1 #-1 because has to be index


        for i in range(len(self.iterable)):
            if i != keep_config:
                loc = self.save_loc + f'config_{i+1}_best_model.pth'
                os.remove(loc)

    def check_single_dataset(self):
        check = True
        for (key,list_vals) in self.hp_dict.items():
            if (len(list_vals) > 1) & (self.hp_trans[key] == 'data'):
                check=False
        return check

    def execute_training(self, exec_type):

        def train_single_model_parallel(model, print_new_config, save_loc):
            return train_single_model(model, print_new_config, save_loc)

        print('Training execution started')
        if exec_type == 'seq':
            self.build_outcome_skeleton()
            tic = time.time()
            for i, model in enumerate(self.list_models):
                self.list_models[i] = train_single_model(model, self.print_new_config, self.save_loc) #TODO: Why isn't this a function of the class?
                self.write_outcome_row(i)
            print(f"Total train time: {time.time()-tic}")
        elif exec_type == 'par':
            torch.set_num_threads(1)
            print("*****Starting parallel execution*****")
            # with ProcessPoolExecutor(max_workers=self.training_params['num_cpus']) as executor:
            #     trained_models = list(
            #         executor.map(train_single_model, self.list_models, [self.print_new_config] * len(self.list_models),
            #                      [self.save_loc] * len(self.list_models)))
            # self.list_models = trained_models
            tic=time.time()
            with parallel_backend("multiprocessing", n_jobs=len(self.iterable)):
                trained_models = Parallel()(
                    delayed(train_single_model)(model, self.print_new_config, self.save_loc) for model in
                    self.list_models)
            self.list_models = trained_models
            print(f"Total train time: {time.time()-tic}")


        else:
            raise ValueError(f"{exec_type} is not a valid type of execution.")

    def build_outcome_skeleton(self):

        self.d = {
            'config': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_time': [],
            'time_per_epoch': [],
            'time_forward_per_epoch': [],
            'time_loss_per_epoch': [],
            'time_backward_per_epoch': [],
            'time_val_per_epoch': [],

        }

        if self.app is not None:
            self.d['train_val'] = []
            self.d['val_val'] = []
            self.d['test_val'] = []
            self.d['train_val_opt'] = []
            self.d['val_val_opt'] = []
            self.d['test_val_opt'] = []

        for hp_key in self.hp_dict:
            self.d[f"hp_{hp_key}"] = []

        with open(self.save_loc + 'output.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.d.keys())

            # Write the headers
            writer.writeheader()

    def write_outcome_row(self,config):

        model = self.list_models[config]

        c = model.get_config()
        self.d['config'].append(c)

        best_losses = model.get_best_loss()

        self.d['train_loss'].append(best_losses[0])
        self.d['val_loss'].append(best_losses[1])
        self.d['test_loss'].append(best_losses[2])
        self.d['train_time'].append(model.tot_train_time)
        self.d['time_per_epoch'].append(model.train_time_per_epoch)
        self.d['time_forward_per_epoch'].append(model.time_forward_per_epoch)
        self.d['time_loss_per_epoch'].append(model.time_loss_per_epoch)
        self.d['time_backward_per_epoch'].append(model.time_backward_per_epoch)
        self.d['time_val_per_epoch'].append(model.time_val_per_epoch)

        for hp_key in self.hp_dict:
            self.d[f"hp_{hp_key}"].append(self.iterable[c - 1][hp_key])

        if (self.app is not None):
            if config==0:
                #Assuming all models have the same data
                global opt_values
                opt_values = self.calc_values(self.list_models[config].data)

            fc_values = self.calc_values(model.data, model.best_net)
            self.d['train_val'].append(fc_values[0])
            self.d['val_val'].append(fc_values[1])
            self.d['test_val'].append(fc_values[2])
            self.d['train_val_opt'].append(opt_values[0])
            self.d['val_val_opt'].append(opt_values[1])
            self.d['test_val_opt'].append(opt_values[2])

        with open(self.save_loc + 'output.csv', 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.d.keys())
            row = {key: self.d[key][config] for key in self.d}
            writer.writerow(row)

    def init_strategy(self):
        self.strat = self.hp_dict['strategy']
        del self.hp_dict['strategy']
        assert(set(self.hp_dict.keys()).issubset(self.hp_trans.keys())), "The keys of hp_dict should be a subset of the keys of hp_trans"

    def initialize_models(self):
        list_models = []
        for (i,hp_dict) in enumerate(self.iterable):
            list_dicts = self.get_dictionaries(hp_dict)
            list_models.append(self.init_single_model(list_dicts,i))

        return list_models

    def init_single_model(self,list_dicts,i):
        if (self.one_dataset):
            if i == 0:
                model = m.Train_model(nn_params=list_dicts[0],
                                      training_params=list_dicts[1],
                                      OP_params=list_dicts[2],
                                      data_params=list_dicts[3],
                                      data=self.data)
                self.single_dataset = model.data

            elif i > 0:
               model = m.Train_model(nn_params=list_dicts[0],
                                    training_params=list_dicts[1],
                                    OP_params=list_dicts[2],
                                    data_params=list_dicts[3],
                                    data=self.single_dataset)
        else:
            model = m.Train_model(nn_params=list_dicts[0],
                                  training_params=list_dicts[1],
                                  OP_params=list_dicts[2],
                                  data_params=list_dicts[3],
                                  data=self.data)

        model.set_config(i+1)

        return model

    def print_new_config(self,model):
        c = model.get_config()
        print(f"\n ##### STARTING CONFIG {c} ##### \n")
        pprint.pprint(self.iterable[c-1])
        print()

    def get_dictionaries(self,hp_dict):

        def process_repair(dict_params):

            repair = dict_params['repair']

            if repair == 'gg':
                dict_params['repair_proxy_feasibility'] = True
                dict_params['constrain_decisions'] = 'greedy'
                dict_params['restore_cyclic_bc'] = 'greedy'
            elif repair == 'gs':
                dict_params['repair_proxy_feasibility'] = True
                dict_params['constrain_decisions'] = 'greedy'
                dict_params['restore_cyclic_bc'] = 'rescale'
            elif repair == 'ng':
                dict_params['repair_proxy_feasibility'] = True
                dict_params['constrain_decisions'] = 'none'
                dict_params['restore_cyclic_bc'] = 'greedy'
            elif repair == 'ns':
                dict_params['repair_proxy_feasibility'] = True
                dict_params['constrain_decisions'] = 'none'
                dict_params['restore_cyclic_bc'] = 'rescale'
            elif repair == 'nn':
                dict_params['repair_proxy_feasibility'] = False
                dict_params['constrain_decisions'] = 'none'
                dict_params['restore_cyclic_bc'] = 'rescale'
            else:
                ValueError(f"{repair} not a valid repair")

            return dict_params

        def retrieve_direct_proxy_loc(OP_params):
            # translate_dict = {
            #     1/4: 10,
            #     1/2: 20,
            #     2/3: 66,
            #     3/4: 22,
            #     1: 5,
            #     4/3: 96,
            #     3/2: 115,
            #     2: 26,
            #     4: 27
            # }

            gammas = [1/4,1/2,2/3,3/4,1,4/3,3/2,2,4]

            loc = OP_params['loc_proxy_model'] + 'output.csv'
            df_output = pd.read_csv(loc)

            translate_dict = {}

            for gamma in gammas:
                filtered_df = df_output[df_output['hp_proxy_steepness'] == gamma]
                min_row = filtered_df.loc[filtered_df['val_loss'].idxmin()]
                config_min = min_row['config']
                translate_dict[gamma] = int(config_min)

            config = translate_dict[OP_params['gamma']]

            OP_params_out = copy.deepcopy(OP_params)
            OP_params_out['loc_proxy_model'] = f"{OP_params['loc_proxy_model']}config_{config}"

            return OP_params_out

        nn_params = copy.deepcopy(self.nn_params)
        training_params = copy.deepcopy(self.training_params)
        OP_params = copy.deepcopy(self.OP_params)
        data_params = copy.deepcopy(self.data_params)

        # for key in hp_dict:
        #     if self.hp_trans[key] == 'nn':
        #         nn_params[key] = hp_dict[key]
        #     elif self.hp_trans[key] == 'train':
        #         training_params[key] = hp_dict[key]
        #     elif self.hp_trans[key] == 'OP_params':
        #         OP_params[key] = hp_dict[key]
        #     elif self.hp_trans[key] == 'data':
        #         data_params[key] = hp_dict[key]
        #     elif self.hp_trans[key] == 'loss':
        #         training_params['loss_params'][key] = hp_dict[key]
        #     else:
        #         raise ValueError(f"Key {key} unsupported hyperparameter")

        for key in hp_dict:
            for dest_dict in self.hp_trans[key]:
                if dest_dict == 'nn':
                    nn_params[key] = hp_dict[key]
                elif dest_dict == 'train':
                    training_params[key] = hp_dict[key]
                elif dest_dict == 'OP_params':
                    OP_params[key] = hp_dict[key]
                elif dest_dict == 'data':
                    data_params[key] = hp_dict[key]
                elif dest_dict == 'loss':
                    training_params['loss_params'][key] = hp_dict[key]
                else:
                    raise ValueError(f"Key {key} unsupported hyperparameter")

        OP_params = process_repair(OP_params)

        if isinstance(training_params['warm_start'], str):
            loc_state = training_params['warm_start'] + "_dict_model.pkl"
            if os.path.exists(loc_state):
                with open(loc_state, 'rb') as file:
                    model_state = pickle.load(file)  # Gives a dict
                nn_params = model_state['nn_params']
            else:
                loc = training_params['warm_start'] + '_best_model.pth'
                model = torch.load(loc, map_location=training_params['device'])
                nn_params = model.nn_params

            nn_params['warm_start'] = training_params['warm_start']
            nn_params['dev'] = training_params['device']

        if training_params['framework'] in ['proxy_direct','proxy_direct_linear']:
            OP_params = retrieve_direct_proxy_loc(OP_params)

        return [nn_params,training_params,OP_params,data_params]

    def make_iterable_hps(self):

        def get_best_hps(fw,df,smoothing,repair=None):

            filter_smoothing = smoothing

            if fw == 'IDF':
                filter_fw = 'ID'
                filter_soc_smoothing = True
                filter_repair = 'ns'
            elif fw == 'IDR':
                filter_fw = 'ID'
                filter_soc_smoothing = False
                if self.training_params['MPC']:
                    filter_repair = 'gg'
                else:
                    filter_repair = 'ns'
            elif fw == 'MLPRc':
                filter_fw = 'GS_proxy'
                filter_soc_smoothing = False
                filter_repair = 'ns'
            elif fw == 'MLPRb':
                filter_fw = 'GS_proxy'
                filter_soc_smoothing = False
                filter_repair = 'gg'

            #Do it differently for direct proxy, to be adjusted for other frameworks
            elif fw in ['proxy_direct', 'proxy_direct_linear']:
                filter_fw = fw
                filter_soc_smoothing = False
                filter_repair = repair

            filtered_df = df[(df['hp_smoothing']==filter_smoothing) & (df['hp_framework']==filter_fw) & (df['hp_include_soc_smoothing']==filter_soc_smoothing) & (df['hp_repair']==filter_repair)]

            gamma = filtered_df.iloc[0]['hp_gamma']
            lr = filtered_df.iloc[0]['hp_lr']

            return lr,gamma

        def get_iterable_sensitivity():

            n_repeats = 12

            if self.training_params['MPC']:
                with open('dict_best_models_MPC.pkl', 'rb') as f:
                    dict_best_models = pickle.load(f)

            else:
                with open('dict_best_models_2.pkl', 'rb') as f:
                    dict_best_models = pickle.load(f)

            fw = self.hp_dict['framework'][0]
            EP = self.OP_params['max_soc']
            smoothing = self.OP_params['smoothing']

            if fw == 'GS_proxy':

                if self.training_params['MPC']:

                    # full repair
                    lr_gg, gamma_gg = get_best_hps('MLPRb', dict_best_models[EP], smoothing)

                    dict_gg = {}
                    for key in self.hp_dict:
                        dict_gg[key] = self.hp_dict[key][0]
                    dict_gg['gamma'] = gamma_gg
                    dict_gg['lr'] = lr_gg
                    dict_gg['repair'] = 'gg'

                    list_gg = []
                    for i in range(n_repeats):
                        d = copy.deepcopy(dict_gg)
                        d['add_seed'] = i
                        list_gg.append(d)

                    iterable = list_gg

                else:

                    #no repair
                    lr_nn,gamma_nn = get_best_hps('MLPRc',dict_best_models[EP],smoothing)

                    dict_nn = {}
                    for key in self.hp_dict:
                        dict_nn[key] = self.hp_dict[key][0]
                    dict_nn['gamma'] = gamma_nn
                    dict_nn['lr'] = lr_nn
                    dict_nn['repair'] = 'nn'

                    #cyclic BC repair
                    lr_nc,gamma_nc = get_best_hps('MLPRc',dict_best_models[EP],smoothing)

                    dict_nc = {}
                    for key in self.hp_dict:
                        dict_nc[key] = self.hp_dict[key][0]
                    dict_nc['gamma'] = gamma_nc
                    dict_nc['lr'] = lr_nc
                    dict_nc['repair'] = 'ns'

                    #full repair
                    lr_gg,gamma_gg = get_best_hps('MLPRb',dict_best_models[EP],smoothing)

                    dict_gg = {}
                    for key in self.hp_dict:
                        dict_gg[key] = self.hp_dict[key][0]
                    dict_gg['gamma'] = gamma_gg
                    dict_gg['lr'] = lr_gg
                    dict_gg['repair'] = 'gg'

                    #iterable = [dict_nn for _ in range(n_repeats)] + [dict_nc for _ in range(n_repeats)] + [dict_gg for _ in range(n_repeats)]

                    list_nn = []
                    for i in range(n_repeats):
                        d = copy.deepcopy(dict_nn)
                        d['add_seed'] = i
                        list_nn.append(d)

                    list_nc = []
                    for i in range(n_repeats):
                        d = copy.deepcopy(dict_nc)
                        d['add_seed'] = i
                        list_nc.append(d)

                    list_gg = []
                    for i in range(n_repeats):
                        d = copy.deepcopy(dict_gg)
                        d['add_seed'] = i
                        list_gg.append(d)

                    iterable = list_nn + list_nc + list_gg

            elif fw in ['proxy_direct','proxy_direct_linear']:

                #no repair
                lr_nn,gamma_nn = get_best_hps(fw,dict_best_models[EP],smoothing,'nn')

                dict_nn = {}
                for key in self.hp_dict:
                    dict_nn[key] = self.hp_dict[key][0]
                dict_nn['gamma'] = gamma_nn
                dict_nn['lr'] = lr_nn
                dict_nn['repair'] = 'nn'

                #cyclic BC repair
                lr_nc,gamma_nc = get_best_hps(fw,dict_best_models[EP],smoothing,'ns')

                dict_nc = {}
                for key in self.hp_dict:
                    dict_nc[key] = self.hp_dict[key][0]
                dict_nc['gamma'] = gamma_nc
                dict_nc['lr'] = lr_nc
                dict_nc['repair'] = 'ns'

                #full repair
                lr_gg,gamma_gg = get_best_hps(fw,dict_best_models[EP],smoothing,'gg')

                dict_gg = {}
                for key in self.hp_dict:
                    dict_gg[key] = self.hp_dict[key][0]
                dict_gg['gamma'] = gamma_gg
                dict_gg['lr'] = lr_gg
                dict_gg['repair'] = 'gg'

                #iterable = [dict_nn for _ in range(n_repeats)] + [dict_nc for _ in range(n_repeats)] + [dict_gg for _ in range(n_repeats)]

                list_nn = []
                for i in range(n_repeats):
                    d = copy.deepcopy(dict_nn)
                    d['add_seed'] = i
                    list_nn.append(d)

                list_nc = []
                for i in range(n_repeats):
                    d = copy.deepcopy(dict_nc)
                    d['add_seed'] = i
                    list_nc.append(d)

                list_gg = []
                for i in range(n_repeats):
                    d = copy.deepcopy(dict_gg)
                    d['add_seed'] = i
                    list_gg.append(d)

                iterable = list_nn + list_nc + list_gg

            elif fw == 'ID':

                if self.training_params['MPC']:
                    # ID with reduced smoothing
                    lr_r, gamma_r = get_best_hps('IDR', dict_best_models[EP], smoothing)

                    dict_r = {}
                    for key in self.hp_dict:
                        dict_r[key] = self.hp_dict[key][0]
                    dict_r['gamma'] = gamma_r
                    dict_r['lr'] = lr_r
                    dict_r['include_soc_smoothing'] = False

                    # iterable = [dict_r for _ in range(n_repeats)] + [dict_f for _ in range(n_repeats)]

                    list_r = []
                    for i in range(n_repeats):
                        d = copy.deepcopy(dict_r)
                        d['add_seed'] = i
                        list_r.append(d)


                    iterable = list_r

                else:


                    # ID with full smoothing
                    lr_f, gamma_f = get_best_hps('IDF', dict_best_models[EP], smoothing)

                    dict_f = {}
                    for key in self.hp_dict:
                        dict_f[key] = self.hp_dict[key][0]
                    dict_f['gamma'] = gamma_f
                    dict_f['lr'] = lr_f
                    dict_f['include_soc_smoothing'] = True

                    #ID with reduced smoothing
                    lr_r, gamma_r = get_best_hps('IDR', dict_best_models[EP], smoothing)

                    dict_r = {}
                    for key in self.hp_dict:
                        dict_r[key] = self.hp_dict[key][0]
                    dict_r['gamma'] = gamma_r
                    dict_r['lr'] = lr_r
                    dict_r['include_soc_smoothing'] = False

                    #iterable = [dict_r for _ in range(n_repeats)] + [dict_f for _ in range(n_repeats)]

                    list_r = []
                    for i in range(n_repeats):
                        d = copy.deepcopy(dict_r)
                        d['add_seed'] = i
                        list_r.append(d)

                    list_f = []
                    for i in range(n_repeats):
                        d = copy.deepcopy(dict_f)
                        d['add_seed'] = i
                        list_f.append(d)

                    iterable = list_r + list_f



            return iterable

        """
        Takes as input a dictionary of lists of hyperparameter values, and returns a list of dictionaries with single
        values of those hyperparameters
        :return:
        """
        if self.sensitivity:
            iterable = get_iterable_sensitivity()

        elif self.strat == 'grid_search':
            keys,values = zip(*self.hp_dict.items())

            combinations = itertools.product(*values)

            iterable = [dict(zip(keys,combination)) for combination in combinations]

        else:
            raise ValueError(f"{self.strat} is an unsupported HP tuning strategy")

        return iterable

    def save_overview(self,list_trained_models):

        outcome_dict = self.get_outcome_dict(list_trained_models)

        with open(self.save_loc+'output_par.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=outcome_dict.keys())

            # Write the headers
            writer.writeheader()

            # Write the data
            for i in range(len(list_trained_models)):
                row = {key: outcome_dict[key][i] for key in outcome_dict}
                writer.writerow(row)

    def get_outcome_dict(self,list_trained_models):

        d = {
            'config': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'train_time': [],
            'time_per_epoch': [],
            'time_forward_per_epoch': [],
            'time_loss_per_epoch': [],
            'time_backward_per_epoch': [],
            'time_val_per_epoch': [],
        }


        if self.app is not None:
            d['train_val'] = []
            d['val_val'] = []
            d['test_val'] = []
            d['train_val_opt'] = []
            d['val_val_opt'] = []
            d['test_val_opt'] = []

            #Assuming all models have the same data
            opt_values = self.calc_values(list_trained_models[0].data)


        for hp_key in self.hp_dict:
            d[f"hp_{hp_key}"] = []

        for model in list_trained_models:
            c = model.get_config()
            d['config'].append(c)

            best_losses = model.get_best_loss()

            d['train_loss'].append(best_losses[0])
            d['val_loss'].append(best_losses[1])
            d['test_loss'].append(best_losses[2])
            d['train_time'].append(model.tot_train_time)
            d['time_per_epoch'].append(model.train_time_per_epoch)
            d['time_forward_per_epoch'].append(model.time_forward_per_epoch)
            d['time_loss_per_epoch'].append(model.time_loss_per_epoch)
            d['time_backward_per_epoch'].append(model.time_backward_per_epoch)
            d['time_val_per_epoch'].append(model.time_val_per_epoch)

            if self.app is not None:
                fc_values = self.calc_values(model.data,model.best_net)
                d['train_val'].append(fc_values[0])
                d['val_val'].append(fc_values[1])
                d['test_val'].append(fc_values[2])
                d['train_val_opt'].append(opt_values[0])
                d['val_val_opt'].append(opt_values[1])
                d['test_val_opt'].append(opt_values[2])


            for hp_key in self.hp_dict:
                d[f"hp_{hp_key}"].append(self.iterable[c-1][hp_key])

        return d

    def calc_values(self,data,forecaster=None):
        values = []

        for set in ['train', 'val', 'test']:

            if forecaster is not None:
                _,_,val = self.app(forecaster.get_price(data[set][0]).cpu().detach().numpy(),data[set][1][0].cpu().detach().numpy())
            else:
                _,_,val = self.app(data[set][1][0].cpu().detach().numpy(),data[set][1][0].cpu().detach().numpy())

            values.append(val)

        return values


def train_single_model(model, print_config_func, save_location):
    """
    Function that trains and saves a single model of type Train_model. This function is outside of the HPTuner class for the parallel execution to work
    :param model: initialized model
    :param print_config_func: function that prints some information before training
    :param save_location: location where the trained model should be stored

    :return model: trained model
    """
    print_config_func(model)
    model.train()
    model.save(save_location)
    return model
