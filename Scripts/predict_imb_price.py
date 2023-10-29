import numpy as np
import pandas as pd
import torch
import datetime as dt
import pytz
import sys
import Data_Elia_API
import time
import os

sys.path.append(f"{os.path.dirname(__file__)}/train_SI_forecaster") #Need to do this for loading pytorch models
from train_SI_forecaster import torch_classes



def get_dataframe(list_data,steps,tense):
    #TODO: check if outcoming data correct

    def determine_start_end(steps,tense):
        now = dt.datetime.now().astimezone(pytz.timezone('GMT'))
        #now = dt.datetime(year=2023,month=5,day=4,hour=15,minute=59).astimezone(pytz.timezone('GMT'))
        if tense == 'fut':
            start = now
            end = start + dt.timedelta(minutes=steps*15)
        elif tense == 'past':
            start=now-dt.timedelta(minutes=steps*15)
            end = now
        else:
            sys.exit('Invalid tense')
        return start,end

    start,end = determine_start_end(steps,tense)

    df = Data_Elia_API.get_dataframes(list_data,start=start,end=end)

    return df

def convert_df_tensor(df):
    #TODO: check why model to 'cpu' iso data to 'cuda' doesn't work

    df = df.drop(['datetime'],axis=1)
    np_array = df.to_numpy()
    tensor = torch.from_numpy(np_array)
    tensor = tensor[None,:,:].float().to('cuda')

    return tensor
def load_forecaster(dict,type):

    if type == 'imb':
        loc = dict['loc_SI_FC']
    elif type == 'price':
        loc = dict['loc_SI_FC']

    model = torch.load(loc)

    return model


if __name__ == '__main__':

    dict_pred = {
        'lookahead': 10,
        'lookback': 4,
        'data_past': ['RT_wind','RT_pv', 'RT_SI'],
        'data_fut': ['DA_F_wind','DA_F_pv','DA_F_nuclear','DA_F_gas'],
        'loc_SI_FC': 'train_SI_forecaster/output/trained_models/LA_10/20230503/config_3.pt',
        'loc_price_FC': ''
    }

    tic = time.time()
    df_past = get_dataframe(list_data=dict_pred['data_past'],steps=dict_pred['lookback'],tense='past')
    df_fut = get_dataframe(list_data=dict_pred['data_fut'],steps=dict_pred['lookahead'],tense='fut')
    data_load_time = time.time()-tic

    fut_tensor = convert_df_tensor(df_fut)
    past_tensor = convert_df_tensor(df_past)

    si_forecaster = load_forecaster(dict=dict_pred,type='imb')

    #TODO: add scaling


    SI_FC = si_forecaster([past_tensor,fut_tensor]).detach().numpy()


    x=1


