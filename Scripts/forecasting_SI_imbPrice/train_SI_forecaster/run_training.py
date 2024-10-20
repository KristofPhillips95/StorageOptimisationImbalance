import sys
import os
import numpy as np
from datetime import datetime
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
old_dir = os.path.join(current_dir,'..','..','train_SI_forecaster')
dir_scaling = os.path.join(old_dir,'..','scaling')
dir_classes = os.path.join(current_dir,'..','train_classes')
sys.path.insert(0,old_dir)
sys.path.insert(0,dir_scaling)
sys.path.insert(0,dir_classes)

import scaling
import functions_train as ft
import functions_data_preprocessing as fdp
from hp_tuner import HPTuner







if __name__ == '__main__':


    la = 12
    lb = 8
    dev = 'cuda'

    data_dict = {
        'data_file_loc': "../../data_preprocessing/data_scaled.h5",
        'read_cols_past_ctxt': ['SI','PV_act','PV_fc','wind_act', 'wind_fc','load_act', 'load_fc'],
        'read_cols_fut_ctxt': ['PV_fc','wind_fc','Gas_fc', 'Nuclear_fc','load_fc'],
        'cols_temp': ['working_day','month_cos','month_sin', 'hour_cos', 'hour_sin', 'qh_cos', 'qh_sin'],
        'target_col': 'SI', #Before: "Frame_SI_norm"
        'datetime_from': datetime(2018,1,1,0,0,0),
        'datetime_to': datetime(2018,2,1,0,0,0),
        'list_quantiles': [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
        'tvt_split': [2/4,1/4,1/4],
        'lookahead': la,
        'lookback': lb,
        'split_val_test': 20, #split up forward pass on validation & test set to avoid memory issues
        'loc_scaler': "../../scaling/Scaling_values.xlsx",
        "unscale_labels":True,
    }

    #scaler = scaling.Scaler(idd['loc_scaler'])

    df_past_ctxt = fdp.read_data_h5(input_dict=data_dict, mode='past')#.drop(["FROM_DATE"],axis=1)
    df_fut_ctxt = fdp.read_data_h5(input_dict=data_dict, mode='fut')#.drop(["FROM_DATE"],axis=1)
    df_temporal = fdp.get_temporal_information(data_dict)

    array_past_ctxt = df_past_ctxt.to_numpy()
    array_fut_ctxt = df_fut_ctxt.to_numpy()
    array_temp = df_temporal.to_numpy()

    #Extend arrays (for RNN input)
    array_ext_past_ctxt, array_ext_fut_ctxt,array_ext_past_temp,array_ext_fut_temp = fdp.get_3d_arrays(past_ctxt=array_past_ctxt,fut_ctxt=array_fut_ctxt,temp=array_temp,input_dict=data_dict)
    labels_ext = fdp.get_3d_arrays_labels(labels = df_past_ctxt,input_dict=data_dict)

    array_ext_past = np.concatenate((array_ext_past_ctxt,array_ext_past_temp),axis=2)
    array_ext_fut = np.concatenate((array_ext_fut_ctxt,array_ext_fut_temp),axis=2)

    feat_train,feat_val,feat_test = fdp.get_train_val_test_arrays([array_ext_past,array_ext_fut],data_dict)
    lab_train,lab_val,lab_test = fdp.get_train_val_test_arrays([labels_ext],data_dict)
    #list_arrays = [feat_train,lab_train,feat_val,lab_val,feat_test,lab_test]

    OP_params_dict = {
        'overwrite_from_proxy': False,
    }

    dict_hps = {
        'hidden_size_lstm': [32],
        'layers_lstm': [2],
        'lr': [0.003],
        'batch_size': [64],
        'strategy': 'grid_search',
        'reg': [0],
        'list_units': [[16]],
        'list_act': [['relu']],
        'encoder_size': [64,128],
        'num_heads': [4],
        'layers': [2,4],
        'ff_dim': [128,512],
        'dropout': [0.1],
        'framework': ['NA'],
        'repair': ['nn'],
        'warm_start': [None],
        'quantile_tensor': [torch.tensor(data_dict['list_quantiles']).to(dev)]
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
        'encoder_size': ['nn'],
        'decay_n': ['loss'],
        'decay_k': ['loss'],
        'p': ['loss'],
        'pen_feasibility': ['train'],
        'hidden_size_lstm': ['nn'],
        'layers_lstm': ['nn'],
        'encoder_size': ['nn'],
        'num_heads': ['nn'],
        'ff_dim': ['nn'],
        'quantile_tensor': ['loss']
    }

    training_dict = {
        'device': dev,
        #'num_cpus': 2,
        'epochs': 100,
        'patience': 10,
        'reg_type': 'quad',  # 'quad' or 'abs',
        'loss_fct_str': 'pinball',  # loss function that will be used to calculate gradients
        'loss_fcts_eval_str': ['pinball'],  # loss functions to be tracked during training procedure
        'loss_params': {},
        'exec': 'seq',  # 'seq' or 'par'
        'makedir': False,
        'framework': 'NA', #Smoothing framework,
        'la': la,
        'include_loss_evol_smooth': False,
        'keep_prices_train': False,
        'keep_sched_train': False
    }

    nn_dict = {
        'type': 'ED_Transformer',  # 'vanilla', 'vanilla_separate', 'RNN_decoder' or 'LSTM_ED'
        'seq_length_d': la,
        'seq_length_e': lb,
        #'input_feat': len(data_dict['feat_cols']) * la,  # Vanilla
        'warm_start': False,
        'output_dim': len(data_dict['list_quantiles']),  # Vanilla & RNN_decoder
        'input_size_e': len(data_dict['read_cols_past_ctxt']) + len(data_dict['cols_temp']), #number of features per timestep encoder
        'input_size_d': len(data_dict['read_cols_fut_ctxt']) + len(data_dict['cols_temp']), #number of features per timestep decoder
        'layers_d': 2,
        'layers_e': 2,
        'hidden_size_lstm': 32,
        'out_dim_per_neuron': len(data_dict['list_quantiles']),
        'dev': dev,
        #Stuff for transformer
        'encoder_seq_length': lb,
        'decoder_seq_length': la,
    }

    save_loc = f"output/trained_models/LA_{la}/20240117_transformer/"

    data = {
        'train': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in feat_train],
                  [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in lab_train]),
        'val': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in feat_val],
                [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in lab_val]),
        'test': ([torch.from_numpy(f).to(torch.float32).to(dev) for f in feat_test],
                 [torch.squeeze(torch.from_numpy(l).to(torch.float32)).to(dev) for l in lab_test]),
    }

    hp_tuner = HPTuner(hp_dict=dict_hps,
                       hp_trans=hp_trans,
                       nn_params=nn_dict,
                       training_params=training_dict,
                       OP_params=OP_params_dict,
                       data_params=data_dict,
                       save_loc=save_loc,
                       data=data)
    hp_tuner()







    ##### OLD TRAINING EXECUTION #####

    # list_output_dict = ft.run_training(dict_params=idd,dict_HPs=hp_dict,list_arrays=list_arrays)
    #
    # dict_data = {key:data_dict[key] for key in ['read_cols_past_ctxt','read_cols_fut_ctxt','cols_temp','list_quantiles']}
    #
    # ft.save_outcome(list_output_dict,dict_data,dir)