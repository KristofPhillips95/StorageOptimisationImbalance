import torch
import time
import numpy as np
import sys
import os
import torch_classes as tc
import csv
import copy
import pandas as pd
import itertools
import pickle


def get_split_indices(dict):
    split = dict['split_val_test']
    split_indices_val = []
    split_indices_test = []
    X_val = dict['val_test_feat'][0]
    X_test = dict['val_test_feat'][1]

    n_val = X_val[0].shape[0]
    n_test = X_test[0].shape[0]

    for i in range(split):
        split_indices_val.append(int(i * n_val / (split)))
        split_indices_test.append(int(i * n_test / (split)))

    return split_indices_val, split_indices_test

def filter_data_list_index(data_list, filter, filter_pos):
    return_list = []

    for data in data_list:
        if filter_pos == len(filter) - 1:
            filtered_data = data[filter[filter_pos]:]
        else:
            filtered_data = data[filter[filter_pos]:filter[filter_pos] + 1]
        return_list.append(filtered_data)

    return return_list

def train_forecaster(input_dict):

    net = input_dict['net']
    training_loader = input_dict['training_loader']

    if 'loss_fct' in input_dict:
        loss_fct = input_dict['loss_fct']
    else:
        loss_fct = torch.nn.MSELoss()

    if 'val_test_feat' in input_dict:
        val_test_feat = input_dict['val_test_feat']
        X_val = val_test_feat[0]
        X_test = val_test_feat[1]
        split_indices_val,split_indices_test = get_split_indices(input_dict)
    else:
        val_test_feat = ['NA','NA']


    if 'val_test_lab' in input_dict:
        val_test_lab = input_dict['val_test_lab']
        Y_val = val_test_lab[0]
        Y_test = val_test_lab[1]
    else:
        val_test_lab = ['NA', 'NA']

    if 'epochs' in input_dict:
        epochs = input_dict['epochs']
    else:
        epochs = 100

    if 'lr' in input_dict:
        lr = input_dict['lr']
    else:
        lr = 0.001

    if 'patience' in input_dict:
        patience = input_dict['patience']
    else:
        patience = 25

    if 'rd_seed' in input_dict:
        rd_seed = input_dict['rd_seed']
    else:
        rd_seed = 42


    ### INITIALIZATION ###

    print(f"lr = {lr}")

    optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
    valid_loss_best = np.inf
    test_loss_best = np.inf
    train_loss_best = 0
    #best_net = copy.deepcopy(net)
    best_net = net





    loss_evolution = np.zeros(epochs)

    epochs_since_improvement = 0

    torch.manual_seed(rd_seed)

    for e in range(epochs):

        if epochs_since_improvement >= patience:
            break

        ### TRAINING PHASE ###
        train_loss = 0.0
        train_start = time.time()
        for i, data in enumerate(training_loader):

            features,labels = retrieve_features_from_loader(data,input_dict)
            # clear gradients
            optimizer.zero_grad()
            # Forward pass
            fc_SI_quant = net(features)

            loss = loss_fct(labels,fc_SI_quant)
            loss.backward()

            # Update weights
            optimizer.step()
            train_loss += loss.item()

        train_time = time.time() - train_start

        loss_evolution[e] = train_loss
        print(f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Train time: {train_time}'  )


        ### VALIDATION PHASE ###

        if X_val == 'NA':
            pass
        else:

            test_loss = 0
            val_loss = 0

            for i in range(input_dict['split_val_test']):

                feat_val = filter_data_list_index(data_list=X_val,filter=split_indices_val,filter_pos = i)
                feat_test = filter_data_list_index(data_list=X_test,filter=split_indices_test,filter_pos = i)
                lab_val = filter_data_list_index(data_list=Y_val,filter=split_indices_val,filter_pos = i)
                lab_test = filter_data_list_index(data_list=Y_test,filter=split_indices_test,filter_pos = i)


                fc_SI_quant_val = net(feat_val)
                fc_SI_quant_test = net(feat_test)

                loss_val = loss_fct(lab_val,fc_SI_quant_val)
                val_loss += loss_val.item()
                loss_test = loss_fct(lab_test,fc_SI_quant_test)
                test_loss += loss_test.item()


            print(
                f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss} \t\t Test Loss: {test_loss}')
            if valid_loss_best > val_loss:
                print(f'Validation Loss Decreased({valid_loss_best:.6f}--->{val_loss:.6f}) \t Saving The Model')
                valid_loss_best = val_loss
                train_loss_best = train_loss
                test_loss_best = test_loss
                epochs_since_improvement = 0
                #best_net = copy.deepcopy(net)
                best_net = net
            else:
                epochs_since_improvement += 1


    return best_net, train_loss_best, valid_loss_best, test_loss_best

def retrieve_features_from_loader(data,dict):
    n_ft = dict['n_components_feat']
    n_lb = dict['n_components_lab']

    if (n_ft == 2) & (n_lb == 1):
        feat_1,feat_2,lab = data
    else:
        sys.exit('You have not modeled this combination of feature/label components')

    return [feat_1,feat_2],[lab]

def set_arrays_to_tensors_device(list_arrays,dev):
    # Set list of tensors to specified device

    global_list = []

    for item in list_arrays:
        if type(item) is list:
            new_entry = [torch.from_numpy(i).float().to(dev) for i in item]
        else:
            new_entry = torch.from_numpy(item).float().to(dev)
        global_list.append(new_entry)

    return global_list

def initialize_net(dict,dev,type):

    input_size_e = dict['input_size_e']
    hidden_size_lstm = dict['hidden_size_lstm']
    input_size_d = dict['input_size_d']
    output_dim = dict['output_dim']
    layers_e = dict['layers_lstm']
    layers_d = dict['layers_lstm']


    if type == 'ED_RNN':
        net = tc.LSTM_ED(input_size_e=input_size_e,
                         layers_e = layers_e,
                         hidden_size_lstm=hidden_size_lstm,
                         input_size_d=input_size_d,
                         layers_d=layers_d,
                         output_dim=output_dim,
                         dev=dev
                         )
    elif type == 'ED_RNN_att':
        net = tc.LSTM_ED_Attention(input_size_e=input_size_e,
                         layers_e = layers_e,
                         hidden_size_lstm=hidden_size_lstm,
                         input_size_d=input_size_d,
                         layers_d=layers_d,
                         output_dim=output_dim,
                         dev=dev
                         )
    else:
        raise ValueError(f"{type} unsupported type of forecaster")
    return net

def hp_tuning(dict,dict_HPs, list_arrays):


    """
        Initialization of data, directory to store outcome, gaussian process

        dev: device to train neural network, cuda or cpu

        train_Dataset: tensordataset from train features and labels. Distinction is made for edRNN (with 2 distinct inputs for encoder and decoder) and other models (with 1 set of input features)

    """
    la = dict['lookahead']
    store_code = dict['store_code']
    dev = dict['dev']
    [feat_train_pt, lab_train_pt, feat_val_pt, lab_val_pt, feat_test_pt, lab_test_pt] = set_arrays_to_tensors_device(list_arrays, dev)

    train_Dataset = torch.utils.data.TensorDataset(feat_train_pt[0], feat_train_pt[1], lab_train_pt[0])
    training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=dict['batch_size'], shuffle=True)

    dict['val_test_feat']= [feat_val_pt,feat_test_pt]
    dict['val_test_lab']= [lab_val_pt,lab_test_pt]
    dict['training_loader'] = training_loader



    outcome_dict = {
        'config': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
    }






    dir = f'output/trained_models/LA_{la}/{store_code}/'
    os.mkdir(dir)


    #### Train sequentially ####
    tic = time.time()
    for hp_config in range(dict['n_configs']):


        #For every hp configuration, initalize a new net
        net = initialize_net(dict_HPs,hp_config,dev)
        train_loss_fct = tc.Loss_pinball(list_quantiles = dict['list_quantiles'],dev=dev)

        #train_loss_fct = sf.Loss_cross_profit()

        input_dict_config = copy.deepcopy(dict)
        input_dict_config['net'] = net
        input_dict_config['loss_fct'] = train_loss_fct
        input_dict_config['config'] = hp_config


        print('')
        print('***** STARTING CONFIG ' + str(hp_config + 1) + ' TRAINING *****\n')


        training_outcome = train_forecaster(input_dict_config)
        net, train_loss, validation_loss, test_loss = training_outcome


        #Store trained model
        store_path = f"{dir}config_{hp_config+1}.pt"
        torch.save(net,store_path)

        outcome_dict['train_loss'].append(train_loss)
        outcome_dict['val_loss'].append(validation_loss)
        outcome_dict['test_loss'].append(test_loss)
        outcome_dict['config'].append(hp_config + 1)


    train_time_seq = time.time()-tic
    print(f"Train time: {train_time_seq}")

    dict_HPs['config']=outcome_dict['config']
    save_outcome(dict_outcome=outcome_dict,dict_HP=dict_HPs,dir=dir)




##### New function training #####
def run_training(dict_params,dict_HPs,list_arrays):

    def get_list_hp_configs(input_dict):
        keys = input_dict.keys()
        values = input_dict.values()

        combos = []
        for combo_values in itertools.product(*values):
            combo = dict(zip(keys, combo_values))
            combos.append(combo)

        return combos


    la = dict_params['lookahead']
    store_code = dict_params['store_code']
    dev = dict_params['dev']
    [feat_train_pt, lab_train_pt, feat_val_pt, lab_val_pt, feat_test_pt, lab_test_pt] = set_arrays_to_tensors_device(list_arrays, dev)

    #train_Dataset = torch.utils.data.TensorDataset(feat_train_pt[0], feat_train_pt[1], lab_train_pt[0])
    train_Dataset = torch.utils.data.TensorDataset(*feat_train_pt,lab_train_pt[0]) #Dynamically define how many feature inputs to train dataset
    training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=dict_params['batch_size'], shuffle=True)

    train_loss_fct = tc.Loss_pinball(list_quantiles=dict_params['list_quantiles'], dev=dev)

    dict_params['val_test_feat']= [feat_val_pt,feat_test_pt]
    dict_params['val_test_lab']= [lab_val_pt,lab_test_pt]
    dict_params['training_loader'] = training_loader
    dict_params['loss_fct'] = train_loss_fct


    list_input_dict = []
    list_output_dict = []
    list_hp_configs = get_list_hp_configs(dict_HPs)
    c = 1

    for dict_hps_config in list_hp_configs:
        input_dict = copy.deepcopy(dict_params)
        input_dict['val_test_feat'] = [feat_val_pt, feat_test_pt]
        input_dict['val_test_lab'] = [lab_val_pt, lab_test_pt]
        input_dict['training_loader'] = training_loader
        input_dict['net'] = initialize_net(dict=dict_hps_config,dev=dev,type=dict_params['forecaster_type'])
        input_dict['config'] = c
        input_dict['lr'] = dict_hps_config['lr']
        c+=1
        list_input_dict.append(input_dict)

    tic = time.time()
    for (i,input_dict) in enumerate(list_input_dict):
        best_net, train_loss_best, valid_loss_best, test_loss_best = train_forecaster(input_dict)
        outcome_c = {
            'config': input_dict['config'],
            'best_net': best_net,
            'train_loss': train_loss_best,
            'val_loss': valid_loss_best,
            'test_loss': test_loss_best,
        }

        for hp_key,hp_val in list_hp_configs[i].items():
            outcome_c[hp_key] = hp_val

        list_output_dict.append(outcome_c)

    train_time = time.time()-tic
    print(f"Train time: {train_time}")

    return list_output_dict


def save_outcome(list_dict_outcome, dict_data, store_code):
    # Function saving all information in a list of dicts coming out of the NN training procedure in specified location

    for dict_out in list_dict_outcome:
        path = f"{store_code}config_{dict_out['config']}.pt"
        torch.save(dict_out['best_net'], path)
        del dict_out['best_net']

    dict_outcome = {}

    for d in list_dict_outcome:
        for key, value in d.items():
            if key not in dict_outcome:
                dict_outcome[key] = []
            dict_outcome[key].append(value)

    path_outcome = f"{store_code}outcome.csv"
    path_data = f"{store_code}data_dict.pkl"

    with open(path_outcome, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dict_outcome.keys())
        writer.writerows(zip(*dict_outcome.values()))

    with open(path_data, 'wb') as file:
        pickle.dump(dict_data, file)

def save_outcome_old(dict_outcome, dir, dict_HP = 'NA', weights = []):
    path = dir + 'outcome.csv'

    with open(path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dict_outcome.keys())
        writer.writerows(zip(*dict_outcome.values()))

    if dict_HP != 'NA':
        path = dir + 'fixed_HPs.csv'
        with open(path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(dict_HP.keys())
            writer.writerows(zip(*dict_HP.values()))

    if weights != []:
        path = dir + 'weights.xlsx'

        df = pd.DataFrame(weights)
        df.to_excel(path,index=False)













