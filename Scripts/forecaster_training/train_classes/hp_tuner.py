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
    def __init__(self, hp_dict, hp_trans, nn_params, OP_params, training_params, data_params, save_loc,  data=None, n_configs=None):
        super(HPTuner,self).__init__()

        self.hp_dict = hp_dict
        self.hp_trans = hp_trans
        self.init_strategy()
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
                self.list_models[i] = train_single_model(model, self.print_new_config, self.save_loc) #TODO: Why isn't this a function of the class?
        elif exec_type == 'par':
            with ProcessPoolExecutor(max_workers=self.training_params['num_cpus']) as executor:
                trained_models = list(
                    executor.map(train_single_model, self.list_models, [self.print_new_config] * len(self.list_models),
                                 [self.save_loc] * len(self.list_models)))
            self.list_models = trained_models
        else:
            raise ValueError(f"{exec_type} is not a valid type of execution.")

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
