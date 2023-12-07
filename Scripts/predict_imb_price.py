import numpy as np
import pandas as pd
import torch
import datetime as dt
import pytz
import sys
import Data_Elia_API as dea
import time
import os
import pickle
import copy
sys.path.insert(0,"train_SI_forecaster")
sys.path.insert(0,"scaling")
import functions_data_preprocessing as fdp
import scaling
import math



sys.path.append(f"{os.path.dirname(__file__)}/train_SI_forecaster") #Need to do this for loading pytorch models
from train_SI_forecaster import torch_classes


def determine_start_end(steps, timeframe):
    now = dt.datetime.now().astimezone(pytz.timezone('GMT'))
    # now = dt.datetime(year=2023,month=5,day=4,hour=15,minute=59).astimezone(pytz.timezone('GMT')) #If you want to set it manually
    if timeframe == 'fut':
        start = now
        end = start + dt.timedelta(minutes=steps * 15)
    elif timeframe == 'past':
        start = now - dt.timedelta(minutes=steps * 15)
        end = now
    else:
        raise ValueError(f'Unsupported timeframe {timeframe}')
    return start, end


def get_dataframe(list_data,steps,timeframe):
    #TODO: check if outcoming data correct

    start,end = determine_start_end(steps,timeframe)

    df = dea.get_dataframes(list_data,start=start,end=end)

    return df

def convert_df_tensor(df):
    #TODO: check why model to 'cpu' iso data to 'cuda' doesn't work

    try:
        df = df.drop(['datetime'],axis=1)
    except:
        pass
    np_array = df.to_numpy()
    np_array= np_array.astype(float)
    tensor = torch.from_numpy(np_array)
    # tensor = tensor[None,:,:].float().to('cuda')
    tensor = tensor[None, :, :].float().to('cpu')

    return tensor

def load_forecaster(dict,type,dev):

    if type == 'imb':
        loc = dict['loc_SI_FC']
    elif type == 'price':
        loc = dict['loc_SI_FC']

    model = torch.load(loc,map_location='cpu')
    model.set_device(dev)

    slash_index = loc.rfind("/")
    loc_folder = loc[:slash_index+1]
    loc_data = loc_folder+"data_dict.pkl"

    with open(loc_data,'rb') as file:
        model_data = pickle.load(file)

    return model, model_data

def pred_SI(lookahead,lookback,dev='cpu'):

    """
    Function returning the RT forecast of the SI

    Parameters:
        - dev: str, optional
            device to which the forecasting model should be loaded: 'cpu' or 'cuda'

    Returns:
        - SI_FC: 2d numpy array with SI forecast. Lookahead instances are along dim 0, pre-determined quantiles along dim 1
        - last_si_value: single value of latest SI. #TODO: currently, this is the imperfect approximation of the SI at the running qh. Adjust this flow of data(?)
        - quantiles: Pre-determined quantiles of SI forecast
    """

    loc_fc = "20231206_2"
    config = 6
    dir = f'train_SI_forecaster/output/models_production/LA_{lookahead}/{loc_fc}/'
    path_data = f"{dir}data_dict.pkl"
    loc_scaling = "scaling/Scaling_values.xlsx"

    with open(path_data, 'rb') as file:
        dict_datapoints = pickle.load(file)

    dict_pred = {
        'lookahead': lookahead,
        'lookback': lookback,
        'data_past': dict_datapoints['read_cols_past_ctxt'],
        'data_fut': dict_datapoints['read_cols_fut_ctxt'],
        'cols_temp': dict_datapoints['cols_temp'],
        'loc_SI_FC': f'{dir}/config_{config}.pt',
        'loc_price_FC': ''
    }

    si_forecaster, model_data = load_forecaster(dict=dict_pred,type='imb',dev=dev)
    quantiles = model_data['list_quantiles']
    scaler = scaling.Scaler(loc_scaling)

    tic = time.time()
    df_past = get_dataframe(list_data=dict_pred['data_past'], steps=dict_pred['lookback'], timeframe='past')
    latest_index = df_past['datetime'].idxmax()
    last_si_value = df_past.at[latest_index,'SI']
    last_si_time = df_past.at[latest_index, 'datetime']

    df_past_scaled = scaling.scale_data(df_past.drop(['datetime'], axis=1))
    df_fut = get_dataframe(list_data=dict_pred['data_fut'], steps=dict_pred['lookahead'], timeframe='fut')
    df_fut_scaled = scaling.scale_data(df_fut.drop(['datetime'], axis=1))
    df_temporal_past = fdp.get_temporal_information(dict_pred, copy.deepcopy(df_past))
    df_temporal_fut = fdp.get_temporal_information(dict_pred, copy.deepcopy(df_fut))

    data_load_time = time.time() - tic
    print(f"Total time for loading data: {data_load_time}s")

    fut_tensor = convert_df_tensor(df_fut_scaled)
    past_tensor = convert_df_tensor(df_past_scaled)
    past_temp_tensor = convert_df_tensor(df_temporal_past)
    fut_temp_tensor = convert_df_tensor(df_temporal_fut)
    past_tensor_temp = torch.cat((past_tensor, past_temp_tensor), dim=2)
    fut_tensor_temp = torch.cat((fut_tensor, fut_temp_tensor), dim=2)

    SI_FC = si_forecaster([past_tensor_temp, fut_tensor_temp]).detach().numpy()

    return SI_FC, (last_si_value,last_si_time), quantiles


def fetch_MO(lookahead,lookback):

    df_past = get_dataframe(list_data=['MO'], steps=lookback, timeframe='past')
    df_fut = get_dataframe(list_data=['MO'], steps=lookahead, timeframe='fut')

    MO_past = df_past.drop('datetime',axis=1).to_numpy()
    MO_fut = df_fut.drop('datetime',axis=1).to_numpy()


    return MO_past,MO_fut

def price_from_SI(SI_FC,MO):

    price_fc = np.zeros_like(SI_FC)

    lookahead = SI_FC.shape[0]
    n_quant = SI_FC.shape[1]

    for i in range(lookahead):
        for j in range(n_quant):
            loc = get_loc_SI_MO(SI_FC[i,j])
            price_fc[i,j] = MO[i,loc]

    return price_fc

def get_loc_SI_MO(si):
    """
    Function that converts single value of SI to corresponding location in merit order
    Assumes that the merit order consists of price for following volume levels: [-1000MW,-900MW,...,-100MW,100MW,200MW,...,1000MW]
    The SI is first truncated to fall within the range of that merit order
    """

    def truncate_si(x):
        if x>=1000:
            return 999
        elif x <= -1000:
            return -999
        else:
            return x

    si = truncate_si(si)

    if si <=0:
        pos_upward_MO = math.ceil(-si/100)-1
        pos = pos_upward_MO +10
    else:
        pos_downward_MO = math.floor(si/100)+1
        pos = 10-pos_downward_MO

    return pos








if __name__ == '__main__':

    dev = 'cpu'
    la = 12
    lb = 8

    quantiles = [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]

    si_fc = (np.random.rand(la,len(quantiles))-0.5)*1000
    # Get the indices that would sort each row
    sorted_indices = np.argsort(si_fc, axis=1)
    # Create a new array with sorted values
    si_fc = np.take_along_axis(si_fc, sorted_indices, axis=1)

    MO_past,MO_fut = fetch_MO(lookahead=la,lookback=lb)

    price_fc_quantiles = price_from_SI(SI_FC=si_fc,MO=MO_fut)



    fc = pred_SI(lookahead=la,lookback=lb,dev=dev)


    x=1

    #SI_FC = si_forecaster([past_tensor,fut_tensor]).detach().numpy()




