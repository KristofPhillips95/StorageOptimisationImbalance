import sys
sys.path.insert(0,"../data_preprocessing")
sys.path.insert(0,"../scaling")
import scaling
import functions_train as ft
import functions_data_preprocessing as fdp
import numpy as np
from datetime import datetime
import os



if __name__ == '__main__':

    idd = {
        'data_file_loc': "../data_preprocessing/data_scaled.h5",
        'read_cols_past_ctxt': ['SI','PV_act','PV_fc','wind_act', 'wind_fc','load_act', 'load_fc'],
        'read_cols_fut_ctxt': ['PV_fc','wind_fc','Gas_fc', 'Nuclear_fc','load_fc'],
        'cols_temp': ['working_day','month_cos','month_sin', 'hour_cos', 'hour_sin', 'qh_cos', 'qh_sin'],
        'target_col': 'SI', #Before: "Frame_SI_norm"
        'datetime_from': datetime(2019,9,1,0,0,0),
        'datetime_to': datetime(2023,9,1,0,0,0),
        'batch_size': 64,
        'list_quantiles': [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
        'tvt_split': [5/7,1/7,1/7],
        'lookahead': 12,
        'lookback': 8,
        'dev': 'cuda',
        'n_components_feat':2, #number of input tensors to neural network for forward pass
        'n_components_lab': 1, #number of input tensors for loss function calc
        'split_val_test': 20, #split up forward pass on validation & test set to avoid memory issues
        #'n_configs': 3, #Number of HP configurations
        'store_code': '20231206',
        'epochs': 100,
        'patience': 20,
        'loc_scaler': "../scaling/Scaling_values.xlsx",
        "unscale_labels":True
    }

    #scaler = scaling.Scaler(idd['loc_scaler'])

    df_past_ctxt = fdp.read_data_h5(input_dict=idd, mode='past')#.drop(["FROM_DATE"],axis=1)
    df_fut_ctxt = fdp.read_data_h5(input_dict=idd, mode='fut')#.drop(["FROM_DATE"],axis=1)
    df_temporal = fdp.get_temporal_information(idd)

    array_past_ctxt = df_past_ctxt.to_numpy()
    array_fut_ctxt = df_fut_ctxt.to_numpy()
    array_temp = df_temporal.to_numpy()

    #Extend arrays (for RNN input)
    array_ext_past_ctxt, array_ext_fut_ctxt,array_ext_past_temp,array_ext_fut_temp = fdp.get_3d_arrays(past_ctxt=array_past_ctxt,fut_ctxt=array_fut_ctxt,temp=array_temp,input_dict=idd)
    labels_ext = fdp.get_3d_arrays_labels(labels = df_past_ctxt,input_dict=idd)

    array_ext_past = np.concatenate((array_ext_past_ctxt,array_ext_past_temp),axis=2)
    array_ext_fut = np.concatenate((array_ext_fut_ctxt,array_ext_fut_temp),axis=2)

    feat_train,feat_val,feat_test = fdp.get_train_val_test_arrays([array_ext_past,array_ext_fut],idd)
    lab_train,lab_val,lab_test = fdp.get_train_val_test_arrays([labels_ext],idd)
    list_arrays = [feat_train,lab_train,feat_val,lab_val,feat_test,lab_test]


    ##### TO DO: write function to create hp list dict #####
    ise = array_ext_past.shape[2]
    isd = array_ext_fut.shape[2]
    # hp_dict = {
    #
    #     'input_size_e': [ise for i in range(idd['n_configs'])],
    #     'hidden_size_lstm': [32,64],
    #     'input_size_d': [isd for i in range(idd['n_configs'])],
    #     'input_size_past_t':[1 for i in range(idd['n_configs'])], #TODO: not doing anything right now? Check
    #     'input_size_fut_t': [1 for i in range(idd['n_configs'])], #TODO: not doing anything right now? Check
    #     'output_dim': [len(idd['list_quantiles']) for i in range(idd['n_configs'])],
    # }

    hp_dict = {
        'input_size_e': [ise], #not a hyperparameter?
        'hidden_size_lstm': [32,64],
        'layers_lstm': [1,2],
        'lr': [0.001,0.005,0.0005,0.0001],
        #'batch_size': [32,64,128], Not included here, defined in the larger stuff
        'input_size_d': [isd], #not a hyperparameter?
        #'input_size_past_t': [1 for i in range(idd['n_configs'])],  # TODO: not doing anything right now? Check
        #'input_size_fut_t': [1 for i in range(idd['n_configs'])],  # TODO: not doing anything right now? Check
        'output_dim': [len(idd['list_quantiles'])], #not a hyperparameter?
        #'recurrent_dropout': xyz #Is included in paper Jérémie (?)
        #'gradient_norm': xyz #Also used in paper Jéremie (?)
    }

    la = idd['lookahead']
    store_code = idd['store_code']
    dir = f'output/trained_models/LA_{la}/{store_code}/'
    os.mkdir(dir)

    #ft.hp_tuning(dict=idd,dict_HPs=hp_dict,list_arrays=list_arrays)
    list_output_dict = ft.run_training(dict_params=idd,dict_HPs=hp_dict,list_arrays=list_arrays)



    dict_data = {key:idd[key] for key in ['read_cols_past_ctxt','read_cols_fut_ctxt','cols_temp','list_quantiles']}

    ft.save_outcome(list_output_dict,dict_data,dir)


    x=1
