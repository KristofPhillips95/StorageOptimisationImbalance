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

    data_dict = {
        'data_file_loc': "../../data_preprocessing/data_scaled.h5",
        'read_cols_past_ctxt': ['SI','PV_act','PV_fc','wind_act', 'wind_fc','load_act', 'load_fc'],
        'read_cols_fut_ctxt': ['PV_fc','wind_fc','Gas_fc', 'Nuclear_fc','load_fc'],
        'cols_temp': ['working_day','month_cos','month_sin', 'hour_cos', 'hour_sin', 'qh_cos', 'qh_sin'],
        'target_col': 'SI', #Before: "Frame_SI_norm"
        'datetime_from': datetime(2017,1,1,0,0,0),
        'datetime_to': datetime(2022,1,1,0,0,0),
        #'batch_size': 63,
        'list_quantiles': [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
        'tvt_split': [3/5,1/5,1/5],
        'lookahead': la,
        'lookback': lb,
        #'n_components_feat':2, #number of input tensors to neural network for forward pass
        #'n_components_lab': 1, #number of input tensors for loss function calc
        'split_val_test': 20, #split up forward pass on validation & test set to avoid memory issues
        #'store_code': '20231211_attention_bidirectional',
        #'epochs': 100,
        #'patience': 10,
        'loc_scaler': "../../scaling/Scaling_values.xlsx",
        "unscale_labels":True,
        #'forecaster_type': 'ED_RNN_att' # 'ED_RNN' or 'ED_RNN_att'
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

    data = {
        'train': ([torch.from_numpy(f).to(torch.float32) for f in feat_train],
                  [torch.squeeze(torch.from_numpy(l).to(torch.float32)) for l in lab_train]),
        'val': ([torch.from_numpy(f).to(torch.float32) for f in feat_val],
                [torch.squeeze(torch.from_numpy(l).to(torch.float32)) for l in lab_val]),
        'test': ([torch.from_numpy(f).to(torch.float32) for f in feat_test],
                 [torch.squeeze(torch.from_numpy(l).to(torch.float32)) for l in lab_test]),
    }

    OP_params_dict = {}

    dict_hps = {
        'hidden_size_lstm': [64,32],
        'layers_lstm': [1],
        'lr': [0.001,0.005,0.0005],
        #'batch_size': [32,64,128], Not included here, defined in the larger stuff
        #'recurrent_dropout': xyz #Is included in paper Jérémie (?)
        #'gradient_norm': xyz #Also used in paper Jéremie (?)
        'strategy': 'grid_search',
        'reg': [0],
    }

    hp_trans = {
        # Dictionary accompanying dict_hps and assigning the HPs to a specific state dictionary in a Train_model object
        'reg': 'train',
        'batches': 'train',
        'lr': 'train',
        'loss_fct_str': 'train',
        'layers_lstm': 'nn',
        'hidden_size_lstm': 'nn'
    }

    training_dict = {
        'device': 'cuda',
        'num_cpus': 2,
        'epochs': 1,
        'patience': 1,
        'reg_type': 'quad',  # 'quad' or 'abs',
        'batch_size': 64,
        'loss_fct_str': 'pinball',  # loss function that will be used to calculate gradients
        'loss_fcts_eval_str': ['pinball'],  # loss functions to be tracked during training procedure
        'exec': 'seq',  # 'seq' or 'par'
        'makedir': True,
    }

    nn_dict = {
        'type': 'LSTM_ED',  # 'vanilla', 'vanilla_separate', 'RNN_decoder' or 'LSTM_ED'
        'seq_length_d': la,
        'seq_length_e': lb,
        'list_units': [100],  # Vanilla
        'list_act': ['relu'],  # Vanilla
        #'input_feat': len(data_dict['feat_cols']) * la,  # Vanilla
        'warm_start': False,
        'output_dim': 1,  # Vanilla & RNN_decoder
        'input_size_e': len(data_dict['read_cols_past_ctxt']) + len(data_dict['cols_temp']) + 1,
        'input_size_d': len(data_dict['read_cols_past_ctxt']) + len(data_dict['cols_temp']),
        'layers_d': 1,
        'layers_e': 1,
        'hidden_size_lstm': 128,
        'out_dim_per_neuron': len(data_dict['list_quantiles']),
        'dev': training_dict['device'],
    }

    save_loc = "../../train_SI_forecaster/output/trained_models/20240108_test/"

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