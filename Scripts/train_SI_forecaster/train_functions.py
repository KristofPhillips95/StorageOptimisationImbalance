import torch
import time
import numpy as np
import sys

def train_forecaster(dict):

    net = dict['net']
    training_loader = dict['training_loader']

    if 'loss_fct' in dict:
        loss_fct = dict['loss_fct']
    else:
        loss_fct = torch.nn.MSELoss()

    if 'val_test_feat' in dict:
        val_test_feat = dict['val_test_feat']
    else:
        val_test_feat = ['NA','NA']


    if 'val_test_lab' in dict:
        val_test_lab = dict['val_test_lab']
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
        patience = 10

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


    X_val = val_test_feat[0]
    X_test = val_test_feat[1]
    Y_val = val_test_lab[0]
    Y_test = val_test_lab[1]


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



            torch.cuda.empty_cache()
            fc_SI_quant_val = net(X_val)
            torch.cuda.empty_cache()
            fc_SI_quant_test = net(X_test)

            loss_val = loss_fct(Y_val,fc_SI_quant_val)
            val_loss = loss_val.item()
            loss_test = loss_fct(Y_test,fc_SI_quant_test)
            #test_loss = loss_test.item()


            ### Code if you want to use cvxpylayers ###
            # _,sched_fc_val = net(X_val)
            # _,sched_fc_test = net(X_test)
            #
            # loss = sf.Loss_profit()
            #
            # profit_val = loss(sched_fc_val,Y_val)
            # profit_test = loss(sched_fc_test,Y_test)
            #
            # valid_loss = -profit_val.item()
            # test_loss = -profit_test.item()

            print(
                f'Epoch {e + 1} \t\t Training Loss: {train_loss} \t\t Validation Loss: {val_loss} \t\t Test Loss: {loss_test}')
            if valid_loss_best > val_loss:
                print(f'Validation Loss Decreased({valid_loss_best:.6f}--->{val_loss:.6f}) \t Saving The Model')
                valid_loss_best = val_loss
                train_loss_best = train_loss
                test_loss_best = 0
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

def hp_tuning_bOpt(dict_hps, params_dict, model_input_dict, loss_hp_dict, model_type, type_train_labels, list_tensors, store_code,ab_testing=False):


    """
        Initialization of data, directory to store outcome, gaussian process

        dev: device to train neural network, cuda or cpu

        train_Dataset: tensordataset from train features and labels. Distinction is made for edRNN (with 2 distinct inputs for encoder and decoder) and other models (with 1 set of input features)

    """

    dev = data_dict['device']
    decomp = dict_hps['decomp']
    batch_size = model_input_dict['batch_size']

    [train_feat,train_lab,val_feat,val_lab,test_feat,test_lab] = set_tensors_to_device(list_tensors,dev)

    train_lab_training,val_lab_training,test_lab_training = get_training_labels(train_lab,val_lab,test_lab,decomp)


    if model_type == 'edRNN':
        if type_train_labels == 'price_schedule':
            train_Dataset = torch.utils.data.TensorDataset(train_feat[0], train_feat[1], train_lab_training[0],train_lab_training[1])
        else:
            train_Dataset = torch.utils.data.TensorDataset(train_feat[0], train_feat[1], train_lab_training)
    else:
        if type_train_labels =='price_schedule':
            train_Dataset = torch.utils.data.TensorDataset(train_feat, train_lab_training[0],train_lab_training[1])
        else:
            train_Dataset = torch.utils.data.TensorDataset(train_feat, train_lab_training)

    training_loader = torch.utils.data.DataLoader(train_Dataset, batch_size=batch_size, shuffle=True)

    hp_fixed_dict = {
        'lr': [dict_hps['lr']],
        'act_1': [dict_hps['act_1']],
        'act_2': [dict_hps['act_2']],
        'hidden_units': [dict_hps['hidden_units']],
        'loss_mode': [loss_hp_dict['mode_n_k']]
    }

    outcome_dict = {
        'config': [],
        'train_loss': [],
        'val_loss': [],
        'test_loss': [],
        'profit_val': [],
        'profit_test': [],
        'rd_seed': []
    }
    la = str(OP_params_dict['lookahead'])
    dir = f'output/trained_models/{model_type}_LA{la}_{store_code}/'
    #dir = 'C://Users//u0137781//OneDrive - KU Leuven//imbalance_price_forecast//Python scripts//trained_models//' + model_type + '//LA' + str(OP_params_dict['lookahead']) + '//pytorch//' + store_code + '//'

    input_dict_fix = {
        'training_loader': training_loader,
        'model_type': model_type,
        'val_test_feat': [val_feat,test_feat],
        'val_test_lab_training': [val_lab_training,test_lab_training],
        'epochs': model_input_dict['epochs'],
        'lr': lr,
        'patience': model_input_dict['patience'],
        'dir': dir+'seq/',
        'params_dict': params_dict,
        'val_feat': val_feat,
        'val_lab_training': val_lab_training,
        'test_feat': test_feat,
        'test_lab_training': test_lab_training,
    }

    if dict_hps['n_configs'] > 0:

        os.mkdir(dir)
        os.mkdir(dir+'xplr/')
        os.mkdir(dir+'sst/')

        array_HPs = []
        array_obj = []

        array_input_dict = []



        #### Train sequentially ####
        tic = time.time()
        for hp_config in range(dict_hps['n_configs']):

            #If p varies, random sampling
            if loss_hp_dict['var_p'] != {}:
                if loss_hp_dict['trad']:
                    p_array = loss_hp_dict['trad_p']
                    for i in range(len(p_array)):
                        if hp_config <= dict_hps['n_configs']/len(p_array)*(i+1):
                            p = p_array[i]
                            break

                else:
                    p = np.random.uniform(low=loss_hp_dict['var_p']['low'],high=loss_hp_dict['var_p']['high'])

                loss_hp_dict['range_p'][0] = p


            #For every hp configuration, initalize a new net
            net = initialize_net(model_type=model_type,model_input_dict=model_input_dict, model_hp_dict=dict_hps, opti_params_dict=params_dict,decomp=decomp,dev=dev)

            if (ab_testing and hp_config % 2 == 1):
                flat_weights[-1] = 1

            else:


                weights_model,flat_weights = random_sampling(loss_hp_dict=loss_hp_dict,max_pos=model_input_dict['lookahead']-1)

                flat_weights.append(loss_hp_dict['range_p'][0])  # Add p to sampled weights
                flat_weights = np.array(flat_weights)
                """
                if hp_config == 0:
                    flat_weights=[1,0,0,1]
                elif hp_config ==2:
                    flat_weights = [0,1,0,1]
                elif hp_config == 4:
                    flat_weights = [0.5,0.5,0,1]
                """


            array_HPs.append(flat_weights)


            weights = fold_weights(flat_weights[:-1], weights_model, loss_hp_dict=loss_hp_dict,max_pos=model_input_dict['lookahead']-1)
            weights_n = weights[:,:,0]
            weights_k = weights[:,:,1:]

            fgv = flat_weights[-1]

            print(flat_weights)
            print(fgv)

            rd_seed = int(round(random.uniform(0,1000)))

            loss_hp_dict['range_p_filled'] = [flat_weights[0]]

            train_loss_fct = initialize_loss_fct(loss_hp_dict,weights,decomp,batch_size)
            #train_loss_fct = sf.Loss_cross_profit()

            input_dict_config = copy.deepcopy(input_dict_fix)
            input_dict_config['rd_seed'] = rd_seed
            input_dict_config['net'] = net
            input_dict_config['loss_fct'] = train_loss_fct
            input_dict_config['config'] = hp_config
            input_dict_config['flat_weights'] = flat_weights

            array_input_dict.append(input_dict_config)

            print('')
            print('***** STARTING CONFIG ' + str(hp_config + 1) + ' TRAINING *****\n')


            training_outcome = train_nn_simple(input_dict_config)


            net, train_loss, validation_loss, test_loss = training_outcome

            #Store trained model
            store_code = dir+'xplr/' + 'config_' + str(hp_config + 1)
            path = store_code + '.pt'
            torch.save(net,path)

            tic =time.time()

            validation_profit = get_profit(net.to('cpu'),params_dict,model_type,val_feat,val_lab,decomp)
            test_profit = get_profit(net.to('cpu'),params_dict,model_type,test_feat,test_lab,decomp)

            opti_time = time.time()-tic

            print(f'Optimization time: {opti_time}s')

            print('')
            print(f'Validation profit: {validation_profit} \t\t Test profit: {test_profit}')



            new_obj = validation_profit
            array_obj.append(new_obj)


            outcome_dict['train_loss'].append(train_loss)
            outcome_dict['val_loss'].append(validation_loss)
            outcome_dict['test_loss'].append(test_loss)
            outcome_dict['config'].append(hp_config + 1)
            outcome_dict['profit_val'].append(validation_profit)
            outcome_dict['profit_test'].append(test_profit)
            outcome_dict['rd_seed'].append(rd_seed)

        train_time_seq = time.time()-tic
        print(f"Train time seq: {train_time_seq}")

        flat_weights_array = np.stack(array_HPs)

        #save_outcome(dict_outcome=outcome_pool,dict_HP_fixed=hp_fixed_dict,weights=flat_weights_array,dir=dir+'pool//')
        save_outcome(dict_outcome=outcome_dict,dict_HP_fixed=hp_fixed_dict,weights=flat_weights_array,dir=dir+'xplr/')



    if loss_hp_dict['sst']['path'] != '':

        path = loss_hp_dict['sst']['path']

        outcome_dict = pd.read_csv(path+'outcome.csv').reset_index().to_dict(orient='list')

        weights_read = pd.read_excel(path+'weights.xlsx').to_numpy()





    ##### Retrain everything with different random seeds #####
    n = loss_hp_dict['sst']['number']
    if n > 0:

        perfo_list = copy.deepcopy(outcome_dict['profit_val'])
        sensitivities = loss_hp_dict['sst']['repeats'] #Number of sensitivities
        list_highest_indices = [perfo_list.index(i) for i in sorted(perfo_list, reverse=True)][:n]
        list_lowest_indices = [perfo_list.index(i) for i in sorted(perfo_list)][:n]

        list_indices = list_highest_indices + list_lowest_indices

        hp_array_sst = []
        outcome_dict_sst = {
            'config': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': [],
            'profit_val': [],
            'profit_test': [],
            'rd_seed': []
        }

        for index in list_indices:

            flat_weights = weights_read[index]
            weights_model, _ = random_sampling(loss_hp_dict=loss_hp_dict,max_pos=model_input_dict['lookahead'] - 1)
            weights = fold_weights(flat_weights[:-1], weights_model, loss_hp_dict=loss_hp_dict,max_pos=model_input_dict['lookahead']-1)
            train_loss_fct = initialize_loss_fct(loss_hp_dict,weights,decomp,batch_size)

            if loss_hp_dict['var_p'] != {}:
                if loss_hp_dict['trad']:
                    p_array = loss_hp_dict['trad_p']
                    for i in range(len(p_array)):
                        if hp_config <= dict_hps['n_configs']/len(p_array)*(i+1):
                            p = p_array[i]
                            break

                else:
                    p = flat_weights[-1]

                loss_hp_dict['range_p'][0] = p





            hp_config = outcome_dict['config'][index]

            print(flat_weights)

            for i in range(sensitivities):

                hp_array_sst.append(flat_weights)

                rd_seed = (outcome_dict['rd_seed'][index]+i+1)%1000

                net = initialize_net(model_type=model_type, model_input_dict=model_input_dict, model_hp_dict=dict_hps,
                                     opti_params_dict=params_dict, decomp=decomp, dev=dev)

                print('')
                print(f'***** STARTING MODEL {index+1} CONFIG {i+1} TRAINING *****\n')

                input_dict_config = copy.deepcopy(input_dict_fix)
                input_dict_config['rd_seed'] = rd_seed
                input_dict_config['net'] = net
                input_dict_config['loss_fct'] = train_loss_fct
                input_dict_config['config'] = hp_config
                input_dict_config['flat_weights'] = flat_weights

                training_outcome = train_nn_simple(input_dict_config)

                net, train_loss, validation_loss, test_loss = training_outcome

                # Store trained model
                store_code = dir +'sst/config_' + str(hp_config+1) + 'sst_' + str(i+1)
                path = store_code + '.pt'
                torch.save(net, path)

                validation_profit = get_profit(net.to('cpu'), params_dict, model_type, val_feat, val_lab,decomp)
                test_profit = get_profit(net.to('cpu'), params_dict, model_type, test_feat, test_lab,decomp)

                print('')
                print(f'Validation profit: {validation_profit} \t\t Test profit: {test_profit}')


                outcome_dict_sst['train_loss'].append(train_loss)
                outcome_dict_sst['val_loss'].append(validation_loss)
                outcome_dict_sst['test_loss'].append(test_loss)
                outcome_dict_sst['config'].append(index + 1)
                outcome_dict_sst['profit_val'].append(validation_profit)
                outcome_dict_sst['profit_test'].append(test_profit)
                outcome_dict_sst['rd_seed'].append(rd_seed)

        flat_weights_array_sst = np.stack(hp_array_sst)

        save_outcome(dict_outcome=outcome_dict_sst,dict_HP_fixed=hp_fixed_dict,weights=flat_weights_array_sst,dir=dir+'sst/')
