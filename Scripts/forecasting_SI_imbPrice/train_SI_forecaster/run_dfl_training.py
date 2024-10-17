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

    repitition = 1
    la=10
    lb=8
    dev = 'cuda'
    loc = "../data/processed_data/SPO_DA/X_df_ds.csv"
    #save_loc = f"train_output/20240528_DA_EP{EP}_eff90_{fw}_{smoothing}_{dict_choices['indepthstr']}_{warmStart}_{dict_choices['gammastr']}_repair{dict_choices['repair_str']}/"
    save_loc = f"train_output_MPC/20240619_MSE_4/"
    makedir = False
    overwrite_OP_params_proxy = False #CHECK WHAT THE HELL HAPPENS IF YOU TAKE TRUE


    dict_hps = {
        # Dictionary of hyperparameters (which are the keys) where the list are the allowed values the HP can take
        'strategy': 'grid_search',
        'type': ['LSTM_ED_Attention'], #'vanilla', 'vanilla_separate', 'RNN_decoder', 'LSTM_ED', 'LSTM_ED_Attention' or 'ED_Transformer'
        'reg': [0],  # [0,0.01],
        'batch_size': [8],  # [8,64],
        'lr': [0.0001],  # [0.000005,0.00005],
        'loss_fct_str': ['mse'],
        'list_units': [[64]], #[[64,32]]
        'list_act': [['relu']],  #[['elu','relu']]  [['softplus']]
        'hidden_size_lstm': [128], #ED
        'layers': [4], #ED and transformer; For MPC: 2, for DA: 4
        'dropout': [0],
        'warm_start': [False],
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
        'loss_fcts_eval_str': ['mse'], # loss functions to be tracked during training procedure, ['profit','mse_price','mse_sched','mse_sched_sm','mse_mu']
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
        'data_file_loc': os.path.join(current_dir,'..','..','..','data_preprocessing','data_qh_SI_imbPrice_scaled_alpha.h5'),
        'read_cols_past_ctxt': ['Imb_price', 'SI', 'PV_act', 'PV_fc', 'wind_act', 'wind_fc', 'load_act', 'load_fc'] + [f"-{int((i + 1) * 100)}MW" for i in range(3)] + [f"{int((i + 1) * 100)}MW" for i in range(3)],
        'read_cols_fut_ctxt': ['PV_fc', 'wind_fc', 'Gas_fc', 'Nuclear_fc', 'load_fc'] + [f"-{int((i + 1) * 100)}MW" for i in range(10)] + [f"{int((i + 1) * 100)}MW" for i in range(10)],
        'cols_temp': ['working_day', 'month_cos', 'month_sin', 'hour_cos', 'hour_sin', 'qh_cos', 'qh_sin'],
        'target_col': 'Imb_price',  # Before: "Frame_SI_norm"
        'datetime_from': datetime(2019, 1, 1, 0, 0, 0),
        'datetime_to': datetime(2020, 12, 30, 0, 0, 0),
        'list_quantiles': [0.5],
        'list_quantiles_SI': [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
        'tvt_split': [5/6, 1/12, 1/12],
        'lookahead': la,
        'lookback': lb,
        'dev': dev,
        'adjust_alpha': False,
        'loc_scaler': os.path.join(current_dir, '..', '..', '..', 'scaling', 'Scaling_values.xlsx'),
        "unscale_labels": True,
    }

    data_np, data = ld.load_data_MPC(la=la, lb=lb, dev=dev, data_dict=data_dict, limit_train_set=20)



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


