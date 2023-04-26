import numpy as np
import time
import pandas as pd
import datetime as dt
from datetime import datetime,timedelta
import os
import torch
from workalendar.europe import Belgium
import torch_classes as tc
import train_functions as tf
import os

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

def read_data_h5(input_dict,mode):
    loc = input_dict['data_file_loc']
    if mode == 'past':
        cols = input_dict['read_cols_past_ctxt']
    elif mode == 'future':
        cols = input_dict['read_cols_fut_ctxt']

    for col in cols:
        if col == cols[0]:
            df = pd.read_hdf(loc,col)
        else:
            df = df.join(pd.read_hdf(loc,col))

    return df.reset_index()

def get_temporal_information(input_dict):
    loc = input_dict['data_file_loc']
    target_col = input_dict['target_col']
    df = pd.read_hdf(loc,target_col)
    cal_BE = Belgium()

    df['Holiday'] = [int(not cal_BE.is_working_day(x.date())) for x in df.index]
    df['month'] = df.index.month - 1  # 0 to 11 max
    df['dayofweek'] = df.index.dayofweek  # 0 to 6 max
    df['hour'] = df.index.hour  ## 0 to 23max
    df['qh'] = df.index.minute / 15  ## 0 to 3 max Multiple of 15 min

    return df.reset_index()


def convert_dict_array(data_dict):
    list_data_arrays = []
    for col in data_dict:
        array = data_dict[col].to_numpy()
        list_data_arrays.append(array)

    concatenated_data = np.concatenate(list_data_arrays,axis=1)

    return concatenated_data

def get_training_indices(list_dataframes, lookahead,lookback):

    list_of_indices = []
    df_0 = list_dataframes[0]



    for index,row in df_0.iterrows():
        if (index <= df_0.shape[0] - lookahead) & (index >= lookback):
            print(f"index: {index}")
            include_index = True
            for df in list_dataframes:
                if row['FROM_DATE'] + dt.timedelta(minutes=(lookahead-1) * 15) != df['FROM_DATE'][index+lookahead-1]:
                    include_index = False
                if row['FROM_DATE'] - dt.timedelta(minutes=(lookback) * 15) != df['FROM_DATE'][index -lookback]:
                    include_index = False
            if include_index:
                list_of_indices.append(index)

    return list_of_indices

def get_3d_arrays(past_ctxt,fut_ctxt,temp,lookahead,lookback):

    n_ex = past_ctxt.shape[0]-lookahead-lookback
    n_cols_past_ctxt = past_ctxt.shape[1]
    n_cols_fut_ctxt = fut_ctxt.shape[1]
    n_cols_temp = temp.shape[1]

    past_ctxt_ext = np.zeros((n_ex,lookback,n_cols_past_ctxt))
    fut_ctxt_ext = np.zeros((n_ex,lookahead,n_cols_fut_ctxt))
    past_temp_ext = np.zeros((n_ex,lookback,n_cols_temp))
    fut_temp_ext = np.zeros((n_ex, lookahead, n_cols_temp))

    for ex in range(n_ex):
        index = ex + lookback

        for lb in range(lookback):
            past_ctxt_ext[ex,lb,:] = past_ctxt[index-lb-1,:]
            past_temp_ext[ex,lb,:] = temp[index-lb-1,:]

        for la in range(lookahead):
            fut_ctxt_ext[ex,la,:] = fut_ctxt[index+la,:]
            fut_temp_ext[ex,la,:] = temp[index+la,:]

    return past_ctxt_ext,fut_ctxt_ext,past_temp_ext,fut_temp_ext

def get_3d_arrays_labels(labels,lookahead,lookback,n_quantiles):
    n_ex = labels.shape[0]-lookahead-lookback
    labels_ext = np.zeros((n_ex,lookahead,n_quantiles))

    for ex in range(n_ex):
        for quant in range(n_quantiles):
            index = ex + lookback
            labels_ext[ex,:,quant] = labels[index:index+lookahead]

    return labels_ext


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

def get_train_val_test_arrays(list_data,idd):

    def get_indices_tvt(data,idd):

        stop_train = int(idd['tvt_split'][0] * data.shape[0])
        start_val = stop_train + max(idd['lookahead'], idd['lookback'])
        stop_val = start_val + int(idd['tvt_split'][1] * data.shape[0])
        start_test = stop_val + max(idd['lookahead'], idd['lookback'])

        return stop_train,start_val,stop_val,start_test

    stop_train,start_val,stop_val,start_test = get_indices_tvt(list_data[0],idd)

    list_train = []
    list_val = []
    list_test = []

    for d in list_data:
        list_train.append(d[0:stop_train])
        list_val.append(d[start_val:stop_val])
        list_test.append(d[start_test:])

    return list_train,list_val,list_test


if __name__ == '__main__':


    idd = {
        'data_file_loc': r'input_data_scaled.h5',
        'read_cols_past_ctxt': ['Frame_RT_wind_total_norm','Frame_RT_pv_norm','Frame_RT_load_norm','Frame_produced_nuclear_norm', 'Frame_produced_gas_norm', 'Frame_produced_norm', 'Frame_BE_Netpos_norm', 'Frame_ACE_norm', 'Frame_SI_norm'],
        'read_cols_fut_ctxt': ['Frame_DA_F_wind_total_norm','Frame_DA_Prices_norm','Frame_DA_F_pv_norm','Frame_DA_F_load_norm','Frame_DA_scheduling_nuclear_norm','Frame_DA_scheduling_gas_norm','Frame_DA_scheduling_water_norm'],
        'target_col': 'Frame_SI_norm',
        'batch_size': 64,
        'list_quantiles': [0.1, 0.5, 0.9],
        'tvt_split': [5/7,1/7,1/7],
        'lookahead': 10,
        'lookback': 4,
        'dev': 'cuda',
        'n_components_feat':2,
        'n_components_lab': 1,
    }

    df_past_ctxt = read_data_h5(input_dict=idd, mode='past').drop(['FROM_DATE'], axis=1)
    df_fut_ctxt = read_data_h5(input_dict=idd, mode='future').drop(['FROM_DATE'], axis=1)
    df_temporal = get_temporal_information(idd).drop(['FROM_DATE'], axis=1)

    array_past_ctxt = df_past_ctxt.to_numpy()
    array_fut_ctxt = df_fut_ctxt.to_numpy()
    array_temp = df_temporal.to_numpy()

    #Check first time: do all indices have the correct dependencies?
    #indices = get_training_indices([df_temporal],lookahead=10,lookback=4)


    #Extend arrays (for RNN input)
    array_ext_past_ctxt, array_ext_fut_ctxt,array_ext_past_temp,array_ext_fut_temp = get_3d_arrays(past_ctxt=array_past_ctxt,fut_ctxt = array_fut_ctxt,temp = array_temp, lookahead = 10, lookback = 4)
    labels_ext = get_3d_arrays_labels(labels = df_past_ctxt['SI'].to_numpy(),lookahead=10,lookback=4,n_quantiles = 3)


    feat_train,feat_val,feat_test = get_train_val_test_arrays([array_ext_past_ctxt,array_ext_fut_ctxt],idd)
    lab_train,lab_val,lab_test = get_train_val_test_arrays([labels_ext],idd)


    tensor_labels_ext = torch.from_numpy(labels_ext)[0:100].to('cuda')

    net = tc.LSTM_ED(input_size_e=9, hidden_size_lstm=64, input_size_d=7, input_size_past_t=1, input_size_fut_t=1, output_dim=len(idd['list_quantiles']), dev='cuda')
    list_quantiles = [0.1,0.5,0.9]







    #Run training


    list_arrays = [feat_train,lab_train,feat_val,lab_val,feat_test,lab_test]

    [feat_train_pt,lab_train_pt,feat_val_pt,lab_val_pt,feat_test_pt,lab_test_pt] = set_arrays_to_tensors_device(list_arrays,idd['dev'])


    train_Dataset = torch.utils.data.TensorDataset(feat_train_pt[0], feat_train_pt[1], lab_train_pt[0])
    training_loader = torch.utils.data.DataLoader(train_Dataset,batch_size=idd['batch_size'],shuffle=True)

    idd['training_loader'] = training_loader
    idd['net'] = net
    idd['loss_fct'] = tc.Loss_pinball(idd['list_quantiles'],idd['dev'])
    idd['val_test_feat'] = [feat_val_pt,feat_test_pt]
    idd['val_test_lab'] = [lab_val_pt,lab_test_pt]

    tf.train_forecaster(idd)




    x=1
