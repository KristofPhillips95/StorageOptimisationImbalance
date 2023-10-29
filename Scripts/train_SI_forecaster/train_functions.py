import torch
import time
import numpy as np
import sys
import os
import torch_classes as tc
import csv
import copy
import pandas as pd


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

def train_forecaster(dict):

    net = dict['net']
    training_loader = dict['training_loader']

    if 'loss_fct' in dict:
        loss_fct = dict['loss_fct']
    else:
        loss_fct = torch.nn.MSELoss()

    if 'val_test_feat' in dict:
        val_test_feat = dict['val_test_feat']
        X_val = val_test_feat[0]
        X_test = val_test_feat[1]
        split_indices_val,split_indices_test = get_split_indices(dict)
    else:
        val_test_feat = ['NA','NA']


    if 'val_test_lab' in dict:
        val_test_lab = dict['val_test_lab']
        Y_val = val_test_lab[0]
        Y_test = val_test_lab[1]
    else:
        val_test_lab = ['NA', 'NA']

    if 'epochs' in dict:
        epochs = dict['epochs']
    else:
        epochs = 100

    if 'lr' in dict:
        lr = dict['lr']
    else:
        lr = 0.001

    if 'patience' in dict:
        patience = dict['patience']
    else:
        patience = 25

    if 'rd_seed' in dict:
        rd_seed = dict['rd_seed']
    else:
        rd_seed = 42


    ### INITIALIZATION ###

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

            features,labels = retrieve_features_from_loader(data,dict)
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

            for i in range(dict['split_val_test']):

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


def initialize_net(dict,config,dev):

    input_size_e = dict['input_size_e']
    hidden_size_lstm = dict['hidden_size_lstm']
    input_size_d = dict['input_size_d']
    input_size_past_t = dict['input_size_past_t']
    input_size_fut_t = dict['input_size_fut_t']
    output_dim = dict['output_dim']


    net = tc.LSTM_ED(input_size_e=input_size_e[config],
                     hidden_size_lstm=hidden_size_lstm[config],
                     input_size_d=input_size_d[config],
                     input_size_past_t=input_size_past_t[config],
                     input_size_fut_t=input_size_fut_t[config],
                     output_dim=output_dim[config],
                     dev=dev
                     )

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



def save_outcome(dict_outcome, dir, dict_HP = 'NA', weights = []):
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
