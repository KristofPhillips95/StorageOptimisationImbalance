import sys
import math
import torch
import numpy as np
from datetime import datetime
import os
import pickle

test = 1

torch.set_num_threads(1)

current_dir = os.path.dirname(os.path.abspath(__file__))
train_classes_dir = os.path.join(current_dir,'..','train_classes')
data_processing_dir = os.path.join(current_dir,'..','data','scripts_preprocessing')
data_load_dir = os.path.join(current_dir,'..','data','scripts_data_loading')
old_dir = os.path.join(current_dir,'..','..','train_SI_forecaster')

sys.path.insert(0,old_dir)
sys.path.insert(0,data_processing_dir)
sys.path.insert(0,train_classes_dir)
sys.path.insert(0,data_load_dir)

from hp_tuner import HPTuner
import preprocess_data_DA as pre
import preprocess_data_IMB as pre_IMB
import load_data as ld

import torch_classes as tc
import opti_problem
import nn_with_opti as nnwo
import functions_data_preprocessing as fdp


def add_soc0_features(data, OP_params, dev):
    for key in data:
        n_datapoints = data[key][0][0].shape[0]
        np.random.seed(73)
        soc_0 = np.random.uniform(low=OP_params['min_soc'], high=OP_params['max_soc'], size=(n_datapoints, 1))
        data[key][0].append(torch.from_numpy(soc_0).to(torch.float32).to(dev))

    return data

def requires_optimal_schedule(list_loss_fcts):
    loss_req_sched = ['mse_sched', 'mse_sched_first','mse_sched_weighted_profit','mse_sched_first_weighted_profit']
    return any(elem in list_loss_fcts for elem in loss_req_sched)

def add_schedule_labels(data,params,dev):
    sched_calculator_mu = nnwo.Schedule_Calculator(OP_params_dict=params,
                                                        fw='ID',
                                                        dev=dev)

    for key in data:
        prices = data[key][1][0]
        sched,_,_ = sched_calculator_mu(prices,smooth=False)
        print(f"{key} set optimized")
        data[key][1].append(sched)

    return data

def add_schedule_labels_2(data,params,dev):
    optiLayer = nnwo.OptiLayer(params)

    for key in data:
        prices = data[key][1][0]
        sched,_ = optiLayer([data[key][1][0],data[key][0][2]])
        print(f"{key} set optimized")
        data[key][1].append(sched)

    return data

def get_dict_choices(fw,smoothing,loss_fct,MPC,indepthLoss,warmStart,EP,manyGamma,repair_list,test_WS):

    dict_choices = {
        'fw': fw,
        'smoothing': smoothing,
        #'config': config,
        'indepthLoss': indepthLoss
    }

    if fw == 'ID':
        if manyGamma:
            dict_choices['include_soc_smoothing'] = [False]
        else:
            dict_choices['include_soc_smoothing'] = [True,False]
        dict_choices['repair_feas'] = [True]
    elif fw in ['GS_proxy', 'proxy_direct', 'proxy_direct_linear']:
        dict_choices['include_soc_smoothing'] = [False]

        if manyGamma:
            dict_choices['repair_feas'] = [True]
        else:
            dict_choices['repair_feas'] = [True,False]
    else:
        ValueError(f'{fw} unsupported framework')

    if fw == 'proxy_direct':
        dict_choices['gamma'] = [1/4,1/2,2/3,3/4,1,4/3,3/2,2,4]
        dict_choices['smoothing_fct'] = [smoothing]
    elif fw == 'proxy_direct_linear':
        dict_choices['gamma'] = [1/4]
        dict_choices['smoothing_fct'] = [smoothing]

    else:

        if smoothing in ['quadratic','quadratic_symm', 'piecewise']:
            if manyGamma:
                dict_choices['gamma'] = [0.001,0.003, 0.01, 0.03, 0.1, 0.3,1,3,10,30,100,300]
            else:
                dict_choices['gamma'] = [0.1,0.3,1,3,10,30]
            dict_choices['smoothing_fct'] = [smoothing]

        elif smoothing == 'logBar':
            if manyGamma:
                dict_choices['gamma'] = [0.001,0.003, 0.01, 0.03, 0.1, 0.3,1,3,10,30,100,300]
            else:
                dict_choices['gamma'] = [0.003,0.01,0.03,0.1,0.3,1]
            dict_choices['smoothing_fct'] = ['logBar']

        else:
            ValueError(f'{smoothing} unsupported smoothing')

    if MPC:
        if warmStart == 'MSE':
            if EP == 1:
                dict_choices['warmStart'] = [f'../pre_training/train_output_imbPrice/20240508_EP1_CBC/config_21']
            elif EP == 4:
                dict_choices['warmStart'] = [f'../pre_training/train_output_imbPrice/20240508_EP4_CBC/config_3']
        elif warmStart == 'LT':
                #dict_choices['warmStart'] = [f'../pre_training/train_output_imbPrice/20240325_CBC_EP1/config_34']
                if EP == 1:
                    dict_choices['warmStart'] = [f'../pre_training/train_output_imbPrice/20240508_EP1_CBC/config_20']
                elif EP == 4:
                    dict_choices['warmStart'] = [f'../pre_training/train_output_imbPrice/20240508_EP4_CBC/config_28']

        elif warmStart == 'cold':
            dict_choices['warmStart'] = [False]
        else:
            ValueError(f"{warmStart} unsupported warm start")
    else:
        if warmStart == 'MSE':
                dict_choices['warmStart'] = ['../pre_training/train_output/20240415_MSE_2/config_11']
        elif warmStart == 'LT':
                dict_choices['warmStart'] = ['../pre_training/train_output/20240411_EP4_generalizedLoss_1000_128/config_44']
        elif warmStart == 'cold':
            dict_choices['warmStart'] = [False]
        else:
            ValueError(f"{warmStart} unsupported warm start")

    if manyGamma:
        dict_choices['gammastr'] = 'manyGamma'
    else:
        dict_choices['gammastr'] = 'redGamma'

    if indepthLoss:
        dict_choices['loss_fcts'] = ['profit','mse_price','mse_sched','mse_sched_sm','mse_mu']
        dict_choices['indepthstr'] = 'inDepth'
    else:
        dict_choices['loss_fcts'] = ['profit','mse_price']
        dict_choices['indepthstr'] = 'fast'

    if MPC:
        #dict_choices['loss_fcts']+= ['mse_sched_first','mse_sched', 'mse_sched_weighted_profit', 'mse_sched_first_weighted_profit']


        if loss_fct == 'm':
            dict_choices['loss_fct'] = ['mse_sched']
        elif loss_fct == 'p':
            dict_choices['loss_fct'] = ['profit']
        elif loss_fct == 'm1':
            dict_choices['loss_fct'] = ['mse_sched_first']
        elif loss_fct == 'mw':
            dict_choices['loss_fct'] = ['mse_sched_weighted_profit']
        elif loss_fct == 'm1w':
            dict_choices['loss_fct'] = ['mse_sched_first_weighted_profit']

    else:
        dict_choices['loss_fct'] = ['profit']


    if MPC:
        if EP == 1:
            dict_choices['loc_proxy_model'] = '../ML_proxy/trained_models_MPC/20240506_eff90_MPC_EP1_realData_100k_avgInput_MSEvar50/config_63'
        elif EP == 4:
            dict_choices['loc_proxy_model'] = '../ML_proxy/trained_models_MPC/20240506_eff90_MPC_EP4_realData_100k_avgInput_MSEvar50/config_44'
        else:
            ValueError(f'EP {EP} not valid')

    else:
        if EP == 1:
            dict_choices['loc_proxy_model'] = '../ML_proxy/trained_models/20240408_eff90_DA_EP1_realData_10k_avgInput_MSEvar25/config_12'
        elif EP == 4:
            dict_choices['loc_proxy_model'] = '../ML_proxy/trained_models/20240408_eff90_DA_EP4_realData_10k_avgInput_MSEvar50/config_12'
        elif EP == 8:
            dict_choices['loc_proxy_model'] = '../ML_proxy/trained_models/20240408_eff90_DA_EP8_realData_10k_avgInput_relAbsError10/config_23'
        elif EP == 12:
            dict_choices['loc_proxy_model'] = '../ML_proxy/trained_models/20240408_eff90_DA_EP12_realData_10k_avgInput_MSEvar25/config_12'
        elif EP == 24:
            dict_choices['loc_proxy_model'] = '../ML_proxy/trained_models/20240408_eff90_DA_EP24_realData_10k_avgInput_MSEvar50/config_11'
        else:
            ValueError(f'EP {EP} not valid')


        if fw == 'proxy_direct':
            dict_choices['loc_proxy_model'] = f'../ML_proxy/trained_models/20240524_DA_cyclic_EP{EP}_extended_3dist_cov_high_eff90_realData_10k_direct/'
        elif fw == 'proxy_direct_linear':
            dict_choices['loc_proxy_model'] = f'../ML_proxy/trained_models/20240524_DA_cyclic_EP{EP}_extended_3dist_cov_high_eff90_realData_10k_direct_linear/'




    if fw == 'ID':
        repair_list = [repair_list[0]]
    dict_choices['repair_list'] = repair_list

    dict_choices['repair_str'] = ""
    for r in repair_list:
        dict_choices['repair_str']+= f"_{r}"


    if test_WS:
        dict_choices['gammastr'] = 'test_WS'
        dict_choices['gamma'] = [0.1, 0.3, 1, 3, 10, 30]

    return dict_choices





if __name__ == '__main__':

    sensitivity = False
    test_WS = False

    fw = 'NA' #'NA', 'ID', 'GS_proxy', 'proxy_direct' or 'proxy_direct_linear'
    smoothing = 'piecewise' #'quadratic' or 'logBar', 'quadratic_symm' , 'piecewise'
    loss_fct = 'mse' #'p' for profit, 'm' for MSE of schedule, 'm1' for MSE on first instance of schedule, 'mw' for MSE of schedule weighted with PF profit, 'm1w' for MSE on first instance of schedule weighted with PF profit
    #this only has impact when MPC = True, otherwise loss_fct = profit
    MPC = True
    EP = 4
    manyGamma = False
    indepthLoss = False
    warmStart = 'cold' #'MSE', 'LT' or 'cold'
    #config = 44 #4->44; 8->55; 12->62; 24->62
    del_models = True
    repair_list = ["nn","ns","gg"]

    dict_choices = get_dict_choices(fw,smoothing,loss_fct,MPC,indepthLoss,warmStart,EP,manyGamma,repair_list,test_WS)

    repitition = 1
    la=10
    lb=8
    dev = 'cuda'
    loc = "../data/processed_data/SPO_DA/X_df_ds.csv"
    #save_loc = f"train_output/20240528_DA_EP{EP}_eff90_{fw}_{smoothing}_{dict_choices['indepthstr']}_{warmStart}_{dict_choices['gammastr']}_repair{dict_choices['repair_str']}/"
    save_loc = f"train_output_MPC/20240619_MSE_4/"
    makedir = False
    overwrite_OP_params_proxy = False #CHECK WHAT THE HELL HAPPENS IF YOU TAKE TRUE


    # dict_hps = {
    #     # Dictionary of hyperparameters (which are the keys) where the list are the allowed values the HP can take
    #     'strategy': 'grid_search',
    #     'type': ['LSTM_ED_Attention'], #'vanilla', 'vanilla_separate', 'RNN_decoder', 'LSTM_ED', 'LSTM_ED_Attention' or 'ED_Transformer'
    #     'reg': [0],  # [0,0.01],
    #     'batch_size': [64],  # [8,64],
    #     'lr': [0.0003,0.0001,0.00005],
    #     'loss_fct_str': dict_choices['loss_fct'],
    #     'list_units': [[64]], #[[64,32]]
    #     'list_act': [['relu']],  #[['elu','relu']]  [['softplus']]
    #     'hidden_size_lstm': [128], #ED
    #     'layers': [4], #ED and transformer; For MPC: 2, for DA: 4
    #     'dropout': [0],
    #     'pen_feasibility': [0],
    #     'gamma': dict_choices['gamma'],
    #     'smoothing': dict_choices['smoothing_fct'], #'logBar' or 'quadratic' or 'quadratic_symm' or 'logistic'
    #     'framework': [dict_choices['fw']], #'ID' or 'GS_proxy'
    #     'include_soc_smoothing': dict_choices['include_soc_smoothing'], #Only has effect on ID
    #     #'repair_proxy_feasibility': dict_choices['repair_feas'],
    #     'repair': dict_choices['repair_list'],
    #     'warm_start': dict_choices['warmStart']
    # }

    dict_hps = {
        # Dictionary of hyperparameters (which are the keys) where the list are the allowed values the HP can take
        'strategy': 'grid_search',
        'type': ['LSTM_ED_Attention'], #'vanilla', 'vanilla_separate', 'RNN_decoder', 'LSTM_ED', 'LSTM_ED_Attention' or 'ED_Transformer'
        'reg': [0],  # [0,0.01],
        'batch_size': [8],  # [8,64],
        'lr': [0.0001],  # [0.000005,0.00005],
        'loss_fct_str': dict_choices['loss_fct'],
        'list_units': [[64]], #[[64,32]]
        'list_act': [['relu']],  #[['elu','relu']]  [['softplus']]
        'hidden_size_lstm': [128], #ED
        'layers': [4], #ED and transformer; For MPC: 2, for DA: 4
        'dropout': [0],
        'pen_feasibility': [0],
        'gamma': [1],
        'smoothing': [smoothing], #'logBar' or 'quadratic' or 'quadratic_symm' or 'logistic'
        'framework': [fw], #'ID' or 'GS_proxy',
        'include_soc_smoothing': [False], #Only has effect on ID
        'repair': ['gg'],
        'warm_start': dict_choices['warmStart']
    }

    hp_trans = {
        # Dictionary accompanying dict_hps and assigning the HPs to a specific state dictionary in a Train_model object
        'type': ['nn'],
        'reg': ['train'],
        'batch_size': ['train'],
        'add_seed': ['OP_params'],
        'warm_start': ['train','nn'],
        'gamma': ['OP_params'],
        'include_soc_smoothing': ['OP_params'],
        #'repair_proxy_feasibility': ['OP_params'],
        'repair': ['OP_params'],
        'lr': ['train'],
        'loss_fct_str': ['train'],
        'smoothing': ['OP_params'],
        'framework': ['train'],
        'list_units': ['nn'],
        'list_act': ['nn'],
        'hidden_size_lstm': ['nn'],
        'layers': ['nn'],
        'dropout': ['nn'],
        'ff_dim': ['nn'],
        'num_heads': ['nn'],
        'encoder_size': ['nn'],
        'decay_n': ['loss'],
        'decay_k': ['loss'],
        'p': ['loss'],
        'pen_feasibility': ['train']
    }

    training_dict = {
        'MPC': MPC,
        'device': dev,
        'epochs': 0,
        'patience': 100,
        'all_validation': True, #If true: losses will be calculated on train, val and test set. If False: losses calculated only on validation set
        'sched_type': 'net_discharge',
        'la': la,
        # What type of schedule to use as labels: "net_discharge" or "extended" (latter being stack of d,c,soc)
        'reg_type': 'quad',  # 'quad' or 'abs',
        #'loss_params': {'la': la, 'loc_preds': 1, 'loc_labels': 0},
        'loss_fcts_eval_str': dict_choices['loss_fcts'], # loss functions to be tracked during training procedure, ['profit','mse_price','mse_sched','mse_sched_sm','mse_mu']
        #'require_mu': True,
        'include_loss_evol_smooth': False,
        'exec': 'seq',  # 'seq' or 'par'
        #'num_cpus': 2,
        'makedir': makedir,
        'feasibility_loss': False,
        'keep_prices_train': True,
        'keep_sched_train': False,
    }



    nn_dict = {
        #'warm_start': training_dict['warm_start'],
        'type': 'LSTM_ED', #'vanilla', 'vanilla_separate', 'RNN_decoder' or 'LSTM_ED'
        'seq_length': la,
        'output_dim': 1, #Vanilla & RNN_decoder
        #'input_size': len(data_dict['feat_cols']),
        'output_size': 1,
        'output_length': la,
        'out_dim_per_neuron': 1,
        'dev': dev,
        'act_last': 'relu',
        # Stuff for transformer
        'encoder_seq_length': lb,
        'decoder_seq_length': la,
    }

    #New way of loading data:

    # if training_dict['MPC']:
    #     data_np, data = ld.load_data_MPC(la=la,lb=lb,dev=dev,limit_train_set=5000)
    # else:
    #     data_np, data = ld.load_data_DA(days_train=1000, last_ex_test=365, dev=dev)

    # Old way of loading data:

    data_dict = {
        'data_file_loc': "../../data_preprocessing/data_scaled.h5",
        'read_cols_past_ctxt': ['SI','PV_act','PV_fc','wind_act', 'wind_fc','load_act', 'load_fc'],
        'read_cols_fut_ctxt': ['PV_fc','wind_fc','Gas_fc', 'Nuclear_fc','load_fc'],
        'cols_temp': ['working_day','month_cos','month_sin', 'hour_cos', 'hour_sin', 'qh_cos', 'qh_sin'],
        'target_col': 'SI', #Before: "Frame_SI_norm"
        'datetime_from': datetime(2018,1,1,0,0,0),
        'datetime_to': datetime(2022,1,1,0,0,0),
        'list_quantiles': [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
        'tvt_split': [2/4,1/4,1/4],
        'lookahead': la,
        'lookback': lb,
        'split_val_test': 20, #split up forward pass on validation & test set to avoid memory issues
        'loc_scaler': "../../scaling/Scaling_values.xlsx",
        "unscale_labels":True,
    }

    df_past_ctxt = fdp.read_data_h5(input_dict=data_dict, mode='past')  # .drop(["FROM_DATE"],axis=1)
    df_fut_ctxt = fdp.read_data_h5(input_dict=data_dict, mode='fut')  # .drop(["FROM_DATE"],axis=1)
    df_temporal = fdp.get_temporal_information(data_dict)

    array_past_ctxt = df_past_ctxt.to_numpy()
    array_fut_ctxt = df_fut_ctxt.to_numpy()
    array_temp = df_temporal.to_numpy()

    # Extend arrays (for RNN input)
    array_ext_past_ctxt, array_ext_fut_ctxt, array_ext_past_temp, array_ext_fut_temp = fdp.get_3d_arrays(
        past_ctxt=array_past_ctxt, fut_ctxt=array_fut_ctxt, temp=array_temp, input_dict=data_dict)
    labels_ext = fdp.get_3d_arrays_labels(labels=df_past_ctxt, input_dict=data_dict)

    array_ext_past = np.concatenate((array_ext_past_ctxt, array_ext_past_temp), axis=2)
    array_ext_fut = np.concatenate((array_ext_fut_ctxt, array_ext_fut_temp), axis=2)

    feat_train, feat_val, feat_test = fdp.get_train_val_test_arrays([array_ext_past, array_ext_fut], data_dict)
    lab_train, lab_val, lab_test = fdp.get_train_val_test_arrays([labels_ext], data_dict)
    # list_arrays = [feat_train,lab_train,feat_val,lab_val,feat_test,lab_test]

    data_np = {
        'train': (feat_train,lab_train),
        'val': (feat_val,lab_val),
        'test': (feat_test, lab_test)
    }

    data = {
        'train': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in feat_train],
                  [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in lab_train]),
        'val': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in feat_val],
                [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in lab_val]),
        'test': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in feat_test],
                 [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in lab_test]),
    }


    data_dict = {}


    nn_dict['input_size_e'] = data_np['train'][0][0].shape[2]
    nn_dict['input_size_d'] = data_np['train'][0][1].shape[2]


    OP_params_dict = {
    #Dict containing info of optimization program
        'MPC': training_dict['MPC'],
        'inv_cost': 0,
        'lookback': 0,
        'lookahead': la,
        'quantiles_FC_SI': [0.01, 0.05],
        'col_SI': 1,
        'restrict_forecaster_ts': True,
        'max_charge': 0.25,
        'max_discharge': 0.25,
        'eff_d': 0.9,
        'eff_c': 0.9,
        'max_soc': EP,
        'min_soc': 0,
        'soc_0': EP/2,
        'ts_len': 1,
        'gamma': 0,
        'smoothing': smoothing,  # 'quadratic' or 'log-barrier',
        'soc_update': False,
        'cyclic_bc': True,
        'combined_c_d_max': False,  # If True, c+d<= P_max; if False: c<=P_max; d<= P_max
        'degradation': False,
        'loc_proxy_params': f'../ML_proxy/OP/20240430_MPC_cyclic_EP{EP}_extended_3dist_cov_high_eff90_realData_100k.pkl',
        'overwrite_from_proxy': overwrite_OP_params_proxy,
        'loc_proxy_model': dict_choices['loc_proxy_model'],
        'repair_proxy_feasibility': False #For calculation of optimal schedules, will be overwritten in hp_tuner()
    }


    if training_dict['MPC']:
        data = add_soc0_features(data,OP_params_dict,dev)
    if requires_optimal_schedule(dict_hps['loss_fct_str']):
        data = add_schedule_labels_2(data,OP_params_dict,dev)




    #Init loss fct application
    loss_dict = {
        'type': 'profit_first',
        'loc_preds': 0,
        'loc_labels': 0,
    }

    loss = tc.LossNew(loss_dict)

    da_app = opti_problem.Application('MPC', OP_params_dict,loss)


    hp_tuner = HPTuner(hp_dict=dict_hps,
                       hp_trans=hp_trans,
                       nn_params=nn_dict,
                       training_params=training_dict,
                       OP_params=OP_params_dict,
                       data_params=data_dict,
                       save_loc=save_loc,
                       data=data,
                       sensitivity=sensitivity,
                       app=da_app,
                       delete_models=del_models)
    hp_tuner()


