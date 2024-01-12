import os
import copy
import itertools
import train_model as m
import pprint
import math
import csv
import time
from concurrent.futures import ProcessPoolExecutor

#TODO: clean up how data is handled

class HPTuner():
    def __init__(self, strat, hp_dict, hp_trans, nn_params, OP_params, training_params, data_params, save_loc,  data=None, n_configs=None):
        super(HPTuner,self).__init__()

        self.strat = strat
        self.hp_dict = hp_dict
        assert(set(hp_dict.keys()).issubset(hp_trans.keys())), "The keys of hp_dict should be a subset of the keys of hp_trans"
        self.hp_trans = hp_trans
        self.nn_params = nn_params
        self.OP_params = OP_params
        self.training_params = training_params
        self.data_params = data_params
        self.data = data
        self.save_loc = save_loc
        self.n_configs = n_configs

        #check if any changes in used data will occur for the hp tuning. If not, later on in the code the data will be loaded only once (to avoid repeatedly optimizing schedules)
        self.one_dataset = self.check_single_dataset()

    def __call__(self):
        #Make new dir to save results
        if self.training_params['makedir']:
            os.makedirs(self.save_loc)
        else:
            print("No dir was created")

        #Create a list of hp dicts
        self.iterable = self.make_iterable_hps()

        #Create list of train_models initialized with the HPs
        tic = time.time()
        self.list_models = self.initialize_models()
        init_time = time.time()-tic

        #Execute the training of all the training models
        self.execute_training(self.training_params['exec'])


        self.save_overview(self.list_models)

    def check_single_dataset(self):
        check = True
        for (key,list_vals) in self.hp_dict.items():
            if (len(list_vals) > 1) & (self.hp_trans[key] == 'data'):
                check=False
        return check

    def execute_training(self, exec_type):
        if exec_type == 'seq':
            for i, model in enumerate(self.list_models):
                self.list_models[i] = train_single_model(model, self.print_new_config, self.save_loc)
        elif exec_type == 'par':
            with ProcessPoolExecutor(max_workers=self.training_params['num_cpus']) as executor:
                trained_models = list(
                    executor.map(train_single_model, self.list_models, [self.print_new_config] * len(self.list_models),
                                 [self.save_loc] * len(self.list_models)))
            self.list_models = trained_models
        else:
            raise ValueError(f"{exec_type} is not a valid type of execution.")

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

        nn_params = copy.deepcopy(self.nn_params)
        training_params = copy.deepcopy(self.training_params)
        OP_params = copy.deepcopy(self.OP_params)
        data_params = copy.deepcopy(self.data_params)


        for key in hp_dict:
            if self.hp_trans[key] == 'nn':
                nn_params[key] = hp_dict[key]
            elif self.hp_trans[key] == 'train':
                training_params[key] = hp_dict[key]
            elif self.hp_trans[key] == 'OP_params':
                OP_params[key] = hp_dict[key]
            elif self.hp_trans[key] == 'data':
                data_params[key] = hp_dict[key]
            else:
                raise ValueError(f"Key {key} unsupported hyperparameter")

        return [nn_params,training_params,OP_params,data_params]

    def make_iterable_hps(self):
        """
        Takes as input a dictionary of lists of hyperparameter values, and returns a list of dictionaries with single
        values of those hyperparameters
        :return:
        """

        if self.strat == 'grid_search':
            keys,values = zip(*self.hp_dict.items())

            combinations = itertools.product(*values)

            iterable = [dict(zip(keys,combination)) for combination in combinations]

        else:
            raise ValueError(f"{self.strat} is an unsupported HP tuning strategy")

        return iterable

    def save_overview(self,list_trained_models):

        outcome_dict = self.get_outcome_dict(list_trained_models)

        with open(self.save_loc+'output.csv', 'w', newline='') as file:
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
            'test_loss': []
        }

        for hp_key in self.hp_dict:
            d[f"hp_{hp_key}"] = []

        for model in list_trained_models:
            c = model.get_config()
            best_losses = model.get_best_loss()
            d['config'].append(c)
            d['train_loss'].append(best_losses[0])
            d['val_loss'].append(best_losses[1])
            d['test_loss'].append(best_losses[2])
            for hp_key in self.hp_dict:
                d[f"hp_{hp_key}"].append(self.iterable[c-1][hp_key])

        return d

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

if __name__ == "__main__":

    #Data split
    train_share = 1
    days_train = math.floor(16/train_share) #64
    last_ex_test = 19 #59
    repitition = 1

    factor_size_ESS = 100
    la=24

    training_dict = {
        'device': 'cpu',
        'model_type': "LR",
        'epochs': 5,
        'patience': 25,
        'type_train_labels': 'price_schedule',  # 'price' or 'price_schedule'
        'sched_type': 'net_discharge', # What type of schedule to use as labels: "net_discharge" or "extended" (latter being stack of d,c,soc)
        'reg_type': 'quad',  # 'quad' or 'abs',
        'reg': 1,
        'batch_size': 64,
        'lr': 0.001,
        'loss_fct_str': 'profit', #loss function that will be used to calculate gradients
        'loss_fcts_eval_str': ['profit', 'mse_sched', 'mse_sched_weighted','mse_price'], #loss functions to be tracked during training procedure
        'config': 1,
        'exec': 'seq', #'seq' or 'par'
        'num_cpus': 2
    }

    data_dict = {
        # Dict containing all info required to retrieve and handle data
        'loc_data': '../../data/processed_data/SPO_DA/',
        'feat_cols': ["weekday", "NL+FR", "GEN_FC", "y_hat"],
        'col_label_price': 'y',
        'col_label_fc_price': 'y_hat',
        'lookahead': la,
        'days_train': days_train,
        'last_ex_test': last_ex_test,
        'train_share': train_share,
        'val_split_mode': 'alt_test',
        # 'separate' for the validation set right before test set, 'alernating' for train/val examples alternating or 'alt_test' for val/test examples alternating
        'scale_mode': 'stand',  # 'norm','stand' or 'none'
        'scale_base': 'y_hat',  # False or name of column; best to use in combination with scale_mode = 'stand'
        'cols_no_centering': ['y_hat'],
        'scale_price': True,
    }

    nn_dict = {
        'list_units': [],
        'list_act': [],
        'warm_start': False,
        'input_feat': len(data_dict['feat_cols']*la),
        'output_dim': la,
    }

    OP_params_dict = {
    #Dict containing info of optimization program
        'max_charge': 0.01 * factor_size_ESS,
        'max_discharge': 0.01 * factor_size_ESS,
        'eff_d': 0.95,
        'eff_c': 0.95,
        'max_soc': 0.04 * factor_size_ESS,
        'min_soc': 0,
        'soc_0': 0.02 * factor_size_ESS,
        'ts_len': 1,
        'opti_type': 'exo',
        'opti_package': 'scipy',
        'lookahead': la,
        'soc_update': False,
        'cyclic_bc': False,
        'combined_c_d_max': False, #If True, c+d<= P_max; if False: c<=P_max; d<= P_max
        'degradation': False,
        'inv_cost': 0,
        'lookback': 0,
        'quantiles_FC_SI': [0.01, 0.05],
        'col_SI': 1,
        'perturbation': 0.2,
        'feat_cols': data_dict['feat_cols'],
        'restrict_forecaster_ts': True,
        'n_diff_features': len(data_dict['feat_cols']),
        'gamma': 1,
        'smoothing': 'quadratic'  # 'quadratic' or 'logBar'
    }

    dict_hps = {
        #Dictionary of hyperparameters (which are the keys) where the list are the allowed values the HP can take
        'reg': [0.0],  # [0,0.01],
        'batches': [64],  # [8,64],
        'gamma': [0,10],  # [0.1,0.3,1,3,10],
        'lr': [0.001],  # [0.000005,0.00005],
        'loss_fct_str': ['mse_sched_weighted'],  # 'profit', 'mse_sched' or 'mse_sched_weighted'
        'smoothing': ['logBar'],  # 'logBar' or 'quadratic'
        'framework': ["ID"], # "ID" (Implicit Differentiation) or "GS" (Gradient Smoothing) or "GS_proxy" (Gradient smoothing with proxy for mu calculation)
        'list_units': [[]],
        'list_act': [[]] #[['softplus']]
    }

    hp_trans = {
        #Dictionary accompanying dict_hps and assigning the HPs to a specific state dictionary in a Train_model object
        'reg': 'nn',
        'batches': 'train',
        'gamma': 'OP_params',
        'lr': 'train',
        'loss_fct_str': 'train',
        'smoothing': 'OP_params',
        'framework': 'nn',
        'list_units': 'nn',
        'list_act': 'nn'
    }


    strat = "grid_search"
    save_loc = "../train_output/20231213_test7/"

    hp_tuner = HPTuner(strat=strat,
                       hp_dict=dict_hps,
                       hp_trans=hp_trans,
                       nn_params=nn_dict,
                       training_params=training_dict,
                       OP_params=OP_params_dict,
                       data_params=data_dict,
                       save_loc=save_loc)
    hp_tuner()

