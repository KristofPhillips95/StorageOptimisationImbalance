import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(current_dir, '..')
sys.path.insert(0,data_dir)

import functions_support as sf
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
#from entsoe import EntsoePandasClient
# Running this script requires an ENTSOE API Key. We did not include ours here, but reviewers interested in executing the script can find info on obtaining one here:
# https://thesmartinsights.com/how-to-query-data-from-the-entso-e-transparency-platform-using-python/.


# To get the last hour of the current day
def rounded_to_the_DA_FC_end():
    tz = pytz.timezone('Europe/Brussels')
    now = datetime.now(tz).replace(tzinfo=None)
    rounded = now - (now - datetime.min) % timedelta(minutes=1440) + timedelta(hours=23)
    return rounded

def get_data_API():

    first = rounded_to_the_DA_FC_end() - timedelta(days=365 * 4)
    last = rounded_to_the_DA_FC_end() - timedelta(hours=48)  # add a 48 hours lag to make sure all data is avaialble
    last_fc = last + timedelta(hours=24)
    start = pd.Timestamp(first, tz='Europe/Brussels')
    end = pd.Timestamp(last, tz='Europe/Brussels')
    end_fc = pd.Timestamp(last_fc, tz='Europe/Brussels')
    #client = EntsoePandasClient(api_key='ENTSOE_API_KEY HERE')
    client = None
    country_code = 'BE'  # Hungary

    df_DA = client.query_day_ahead_prices(country_code, start=start, end=end)  # different than the forecasted load end
    df_FR = client.query_load_forecast('FR', start=start, end=end_fc).resample('60T').mean()
    df_NL = client.query_load_forecast('NL', start=start, end=end_fc).resample('60T').mean()

    # In case of missing values, interpolate
    df_DA.interpolate(method='linear', inplace=True)
    df_FR.interpolate(method='linear', inplace=True)
    df_NL.interpolate(method='linear', inplace=True)

    df_DA.index = df_DA.index.tz_convert(None)
    df_FR.index = df_FR.index.tz_convert(None)
    df_NL.index = df_NL.index.tz_convert(None)

    df_LOAD = client.query_load_forecast(country_code, start=start, end=end_fc).resample('60T').mean()
    df_GEN = client.query_generation_forecast(country_code, start=start, end=end_fc).resample('60T').mean()

    # In case of missing values, interpolate
    df_LOAD.interpolate(method='linear', inplace=True)
    df_GEN.interpolate(method='linear', inplace=True)

    # Creating the dataframe for the model
    X_df = pd.DataFrame()
    X_df['ds'] = df_LOAD.index
    X_df['unique_id'] = 'BE'
    len_price = len(df_DA.values)
    X_df["y"] = 0
    X_df["y"][0:len_price] = df_DA.values  # The remaining yero values will be the forecasted ones
    X_df['LOAD_FC'] = df_LOAD.values
    X_df['GEN_FC'] = df_GEN.values
    X_df['FR'] = df_FR.values
    X_df['NL'] = df_NL.values
    # Adding the temporal features
    X_df['week_day'] = X_df['ds'].dt.day_of_week
    X_df['hour-of-day'] = X_df['ds'].dt.hour

    # For local purpose
    X_df = X_df[:-24]  # Remove the last 24 hours of the dataframe

    return X_df

def get_daily_data(features,labels,scale_mode):

    def drop_first_rows(list_df,rows):

        list_df_cut = []

        for df in list_df:
            df = df.drop(df.index[:rows])
            df.reset_index(drop=True, inplace=True)
            list_df_cut.append(df)

        return list_df_cut

    def get_extended_array_df(df,la=24):


        n_datapoints = df.shape[1]
        array = np.zeros((la,n_datapoints))
        cols = df.columns.tolist()

        for(i,_) in enumerate(df.itertuples(index=False)):
            for (j,c) in enumerate(cols):
                array[i,j] = df.loc[i,c]

        return array

    if scale_mode == 'none':
        end_of_day = 23
    elif scale_mode == 'norm':
        end_of_day = 1
    else:
        ValueError(f'{scale_mode} unsupported scale mode for retrieving daily data')


    list_arrays_feat = []
    list_arrays_lab = []

    while features.shape[0] > 23:
        next_index_day_start = (abs(features['hour-of-day']-end_of_day) < 1e-6).idxmax()

        if next_index_day_start ==23:
            list_arrays_feat.append(get_extended_array_df(features.head(24)))
            list_arrays_lab.append(get_extended_array_df(labels.head(24)))

        features,labels = drop_first_rows([features,labels],round(next_index_day_start+1))

        # Stick lists of arrays
    feat_comb = np.stack(list_arrays_feat)
    lab_comb = np.stack(list_arrays_lab)

    return feat_comb,lab_comb

def split_train_val_test(array,indices):

    list_arrays = []
    for indic in indices:
        list_arrays.append([np.take(array,indic,axis=0)])

    return list_arrays

def aggregate_data_daily(data):
    data_agg = np.zeros((data.shape[0],data.shape[1]*data.shape[2]))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                data_agg[i,j*data.shape[2]+k] = data[i,j,k]

    return data_agg

def get_data_csv(data_dict,aggregate_daily=False,encoder_decoder=False):
    print("started")
    tic = time.time()
    #Extract information from input dict
    days_train = data_dict['days_train']
    last_ex_test = data_dict['last_ex_test']
    train_share = data_dict['train_share']
    cols_feat = data_dict['feat_cols']
    cols_lab = data_dict['lab_cols']
    scale_mode = data_dict['scale_mode']
    cols_no_centering = data_dict['cols_no_centering']
    print(f"Dict info extracted in {time.time()-tic} seconds")
    tic = time.time()

    #Read csv file in dataframe and scale
    try:
        X_df = pd.read_csv(data_dict['loc_data']+"X_df_ds.csv")
    except:
        X_df = pd.read_csv(data_dict['loc_data'])

    X_df['ds'] = X_df['ds'].astype(str)
    X_df['ds'] = pd.to_datetime(X_df['ds'])
    print(f"df read in {time.time()-tic} seconds")
    tic = time.time()
    X_df_scaled,_ = sf.scale_df_new(X_df,scale_mode,cols_no_centering)
    buffer = 10
    X_df = X_df.tail((days_train + last_ex_test+buffer)*24).reset_index(drop=True)
    X_df_scaled = X_df_scaled.tail((days_train + last_ex_test+buffer)*24).reset_index(drop=True)


    #Get unscaled hour of day --> required in next step for grouping the data per day
    #X_df_scaled['hour-of-day'] = X_df['hour-of-day']
    print(f"df scaled in {time.time()-tic} seconds")
    tic = time.time()

    #Select scaled or unscaled features
    if scale_mode == 'none':
        df_feat = X_df[cols_feat]
    else:
        df_feat = X_df_scaled[cols_feat]
    df_lab = X_df[cols_lab]

    #Convert 2d information to 3d array grouped per day
    daily_feat,daily_lab = get_daily_data(df_feat,df_lab,scale_mode)
    print(f"df converted to 3d array in {time.time()-tic} seconds")
    tic = time.time()

    if aggregate_daily:
        daily_feat = aggregate_data_daily(daily_feat)
        daily_lab = aggregate_data_daily(daily_lab)

    if encoder_decoder:

        if scale_mode == 'none':
            daily_lab_feat = daily_lab
        else:
            daily_lab_feat = daily_lab/np.max(daily_lab)*2
            #daily_lab_feat = daily_lab


        daily_feat_enc = np.concatenate((daily_feat,daily_lab_feat),axis=2)
        daily_feat_enc = daily_feat_enc[:-1,...]
        daily_feat_dec = daily_feat[1:,...]
        daily_lab_dec = daily_lab[1:, ...]



    if not encoder_decoder:
        #Select indices from 3 arrays to split in train,val,test sets
        indices = sf.get_indices(last_ex_test,days_train,train_share,daily_feat.shape[0],data_dict['val_split_mode'])
        features = split_train_val_test(daily_feat,indices)
        labels = split_train_val_test(daily_lab,indices)

        print(f"Split in train/val/test set in {time.time()-tic} seconds")
        tic = time.time()

    else:
        indices = sf.get_indices(last_ex_test,days_train,train_share,daily_feat_enc.shape[0],data_dict['val_split_mode'])
        features_encoder = split_train_val_test(daily_feat_enc,indices)
        features_decoder = split_train_val_test(daily_feat_dec,indices)
        labels = split_train_val_test(daily_lab_dec,indices)

        features = []

        for (f_set_enc,f_set_dec) in zip(features_encoder,features_decoder):
            f_set = f_set_enc+f_set_dec
            features.append(f_set)



    return features,labels