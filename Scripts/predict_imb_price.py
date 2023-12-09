import numpy as np
import pandas as pd
import torch
import datetime as dt
from datetime import datetime,timedelta
import pytz
import sys
import Data_Elia_API as dea
import time
import os
import pickle
import copy
import math
sys.path.insert(0,"train_SI_forecaster")
sys.path.insert(0,"scaling")
import scaling
import functions_data_preprocessing as fdp
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

def get_most_recent(lookback=10):

    def get_most_recent(df,target):
        dict_target = {
            'SI': 'systemimbalance',
            'imbPrice': 'positiveimbalanceprice'
        }
        df_sorted = df.sort_values(by='datetime',ascending=False)
        df_filtered = df_sorted[df_sorted['systemimbalance'].apply(lambda x: isinstance(x, float))]

        latestval = df_filtered[dict_target[target]].iloc[0]
        latest_dt = df_filtered['datetime'].iloc[0]

        return (latestval,latest_dt)

    start,end = determine_start_end(lookback,'past')
    df = dea.get_dataframes(list_data=['SI_and_price'],start=start,end=end)

    tuple_SI = get_most_recent(df,'SI')
    tuple_imbPrice = get_most_recent(df,'imbPrice')

    return tuple_SI, tuple_imbPrice

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
        -lookahead: int
            amount of instances of prediction
        -lookback: int
            amount of instances of past observations used in prediction
        - dev: str, optional
            device to which the forecasting model should be loaded: 'cpu' or 'cuda'

    Returns:
        - SI_FC: 2d numpy array with SI forecast. Lookahead instances are along dim 0, pre-determined quantiles along dim 1
        - curr_qh: datetime object of quarter hour that marks last qh considered as 'past'
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
    curr_qh = df_past.at[latest_index, 'datetime']

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

    return SI_FC, curr_qh, quantiles

def fetch_MO(lookahead,lookback):

    """
    Returns past and future ARC MO based on amount of lookahead and lookback instances as numpy arrays
    """

    brussels_timezone = pytz.timezone('Europe/Brussels')

    # Get the current time in the Brussels time zone
    current_time = datetime.now(brussels_timezone)

    time_end_la = current_time + timedelta(minutes=lookahead * 15)

    df_past = get_dataframe(list_data=['MO'], steps=lookback, timeframe='past')
    #TODO: this prolly will give issues too when we use this as features for price forecasting, and cannot be solved the same way as for the 'fut' case --> I think we should store these values to re-use them

    if current_time.date() == time_end_la.date():
        df_fut = get_dataframe(list_data=['MO'], steps=lookahead, timeframe='fut')
    else:
        df_fut = get_dataframe(list_data=['MO_bids'], steps=lookahead, timeframe='fut')
        #TODO: test if this implementation works when the bids are extended to the next day but the volume levels aren't

    MO_past = df_past.drop('datetime',axis=1).to_numpy()
    MO_fut = df_fut.drop('datetime',axis=1).to_numpy()


    return MO_past,MO_fut

def price_from_SI(SI_FC,MO):

    """
    Returns quantile forecasts of the imbalance price by combining quantile forecasts of SI with the ARC merit order

    Parameters:
        -SI_FC: (LAxn_quant) np array type float
            Quantile forecasts of the system imbalance
            LA: lookahead of prediction
            n_quant: number of quantiles included in probabilistic forecast
        -MO: (LA x MO_size) np array type float
            Merit order containing the activation price per volume level
            LA: lookahead of prediction
            MO_size: size of the merit order, including both upward and downward volume levels

    Returns:
        -price_fc: (LA x n_quant) np array type float
            imbalance price forecast corresponding one-on-one with the SI forecasts given as input
    """

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

    Parameters:
        -si: float
            value of system imbalance

    Returns:
        -pos: int
            location along 1d merit order (assuming the above volume levels) corresponding to the specific si value
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

def convert_quantiles_probabilities(quantiles):
    """
    Assign probability to specific quantiles
    Assumption:
        -Probability of one quantile is found by finding the middle between its neighbouring quantiles, and calculating the width of these 'endpoints'
    #TODO: Do this more rigourously?

    Parameters:
        -quantiles: list type float
            The quantiles (values in interval (0,1), ascending order) which are predicted

    Returns:
        -probs: list type float
            Probabilities associated with specified quantiles
    """

    def get_endpoints(quantiles):
        """
        Determines probability interval covered by the different quantiles by considering the "midpoints" between the different quantiles

        Parameters:
            -quantiles: list type float
                The quantiles (values in interval (0,1), ascending order) which are predicted

        Returns:
            -endpoints: list of tuples type float
                Values in interval [0,1] that are assumed to be covered by the specified quantiles
        """
        endpoints = []
        ep_l = 0

        for (i,q) in enumerate(quantiles):
            if i >0:
                ep_l = ep_h
            if i == len(quantiles)-1:
                ep_h = 1
            else:
                ep_h = (quantiles[i+1]+q)/2
            endpoints.append((ep_l,ep_h))

        return endpoints

    probs = []
    endpoints = get_endpoints(quantiles)
    for (ep_l,ep_h) in endpoints:
        probs.append(ep_h-ep_l)

    return probs

def calc_avg_price(price_quantiles,probs):
    """
    Calculates weighted average of price quantile forecasts given probabilities of specified quantiles

    Parameters:
        -price_quantiles: (LA x n_quant) np array type float
            Quantile forecasts of the imbalance price
            LA: lookahead of prediction
            n_quant: number of quantiles included in probabilistic forecast

    Returns:
        -avg_price_fc: (LA) np array type float
            weighted average of price forecast per lookahead instance
    """
    la = price_quantiles.shape[0]
    n_quantiles = price_quantiles.shape[1]

    assert n_quantiles==len(probs), f"Number of probabilities is {len(probs)}, whereas the price forecast contains {n_quantiles} fields."

    avg_price_fc = np.zeros(la)

    for i in range(la):
        avg_price_fc[i] = np.sum(np.multiply(probs,price_quantiles[i,:]))

    return avg_price_fc

def get_price_fc(SI_FC,MO,quantiles):
    """
    Returns average and quantile price forecast given quantile forecasts of the SI, ARC merit order and the quantile values which were forecasted

    Parameters:
        -SI_FC: (LAxn_quant) np array type float
            Quantile forecasts of the system imbalance
            LA: lookahead of prediction
            n_quant: number of quantiles included in probabilistic forecast
        -MO: (LA x MO_size) np array type float
            Merit order containing the activation price per volume level
            LA: lookahead of prediction
            MO_size: size of the merit order, including both upward and downward volume levels
        -quantiles: list type float
            The quantiles (values in interval (0,1), ascending order) which are predicted

    Returns:
        -avg_price_fc: (LA) np array type float
            weighted average of price forecast per lookahead instance
        -price_fc_quantiles: (LA x n_quant) np array type float
            imbalance price quantile forecast corresponding one-on-one with the SI quantile forecasts given as input
    """

    probs = convert_quantiles_probabilities(quantiles)

    price_fc_quantiles = price_from_SI(SI_FC,MO)

    avg_price_fc = calc_avg_price(price_quantiles=price_fc_quantiles,probs=probs)

    return avg_price_fc,price_fc_quantiles

def optimize_schedule(soc_0,avg_price_forecast):

    """
    Returns optimized schedule based on average price forecast
    CURRENT IMPLEMENTATION: RANDOM SCHEDULE GENERATOR

    Parameters:
        -soc_0: float
            State of charge at start of optimization
        -avg_price_forecast: (LA) np array type float
            Average price forecast over lookahead horizon
            LA: lookahead of prediction
    Returns:
        -c: (LA) np array type float, positive values
            vector of charge decisions
        -d: (LA) np array type float, positive values
            vector of discharge decisions
        -soc: (LA+1) np array type float
            vector of optimized state of charge evolution, soc[0] = soc_0
    """

    lookahead = avg_price_forecast.shape[0]
    c = np.zeros_like(avg_price_forecast)
    d = np.zeros_like(avg_price_forecast)
    soc = np.zeros(lookahead+1)
    soc[0] = soc_0

    rands = np.random.uniform(-1,1,size=(lookahead))

    soc_max = 4
    soc_min = 0

    for la in range(lookahead):

        if rands[la] > 0:
            if soc[la] - rands[la] < soc_min:
                d[la] = soc[la]-soc_min
            else:
                d[la] = rands[la]
        else:
            if soc[la] - rands[la] > soc_max:
                c[la] = soc_max - soc[la]
            else:
                c[la] = -rands[la]

        soc[la+1] = soc[la] + c[la] - d[la]

    return c,d,soc








def call_prediction(soc_0):

    """
    Function returning the RT forecast of the SI and imbalance price

    Returns:
        -fc: (LA x n_quant) np array type float
            Quantile forecasts of the system imbalance
            LA: lookahead of prediction
            n_quant: number of quantiles included in probabilistic forecast

        -avg_price_fc: (LA) np array type float
            weighted average of price forecast per lookahead instance
        -quantile_price_fc: (LA x n_quant) np array type float
            imbalance price forecast corresponding one-on-one with the SI forecasts given as input
        -quantiles: list type float
            The quantiles (values in interval (0,1), ascending order) which are predicted
        -curr_qh: datetime
            Lastest quarter hour considered as 'past' information
        -last_si_value: float
            Value of latest known full qh SI.
        -last_si_time: datetime
            quarter hour of latest known SI
        -last_imbPrice_value: float
            Value of latest known full qh imbalance price.
        -last_imbPrice_dt: datetime
            quarter hour of latest known imbalance price
        -c: (LA) np array type float, positive values
            vector of charge decisions
        -d: (LA) np array type float, positive values
            vector of discharge decisions
        -soc: (LA+1) np array type float
            vector of optimized state of charge evolution, soc[0] = soc_0
    """

    dev = 'cpu'
    la = 12
    lb = 8


    (last_si_value,last_si_dt),(last_imbPrice_value,last_imbPrice_dt) = get_most_recent()

    MO_past,MO_fut = fetch_MO(lookahead=la,lookback=lb)

    si_quantile_fc, curr_qh, quantiles = pred_SI(lookahead=la,lookback=lb,dev=dev)

    avg_price_fc,quantile_price_fc = get_price_fc(SI_FC=si_quantile_fc,MO=MO_fut,quantiles=quantiles)

    c,d,soc = optimize_schedule(soc_0=soc_0,avg_price_forecast=avg_price_fc)

    return si_quantile_fc, avg_price_fc, quantile_price_fc, quantiles, curr_qh, (last_si_value,last_si_dt), (last_imbPrice_value,last_imbPrice_dt), (c,d,soc)


if __name__ == '__main__':

    soc_0 = 2

    si_quantile_fc, avg_price_fc, quantile_price_fc, quantiles, curr_qh, (last_si_value,last_si_time), (last_imbPrice_value,last_imbPrice_dt), (c,d,soc) = call_prediction(soc_0)

    x=1




