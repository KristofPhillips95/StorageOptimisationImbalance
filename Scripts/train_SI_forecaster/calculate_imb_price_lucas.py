import sys
sys.path.insert(0,"../data_preprocessing")
sys.path.insert(0,"../scaling")

import scaling
import functions_train as ft
import functions_data_preprocessing as fdp
import numpy as np
from datetime import datetime
import os
import pandas as pd
import datetime as dt
import torch
import pickle
import math
from datetime import timedelta
import numpy as np
import sys
from dateutil.relativedelta import relativedelta

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

    lookahead = MO.shape[0]
    n_quant = SI_FC.shape[1]

    price_fc = np.zeros((lookahead,n_quant))

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


def string_from_month(month):
    if month < 10:
        return '0' + str(month)
    else:
        return str(month)

def load_ARC_month(year,month):
    folder_ARC_data = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance_price_forecast//Data//meritOrder'
    file_loc_ARC = folder_ARC_data + '//ARC_VolumeLevelPrices_'+str(year)+'_'+string_from_month(month)+'.csv'

    df_ARC = pd.read_csv(file_loc_ARC, delimiter=';', decimal=',')
    df_ARC = df_ARC.rename({'#NAME?': '-Max'}, axis=1)
    columns = list(df_ARC.columns)

    for i in range(2, 12):
        df_ARC[columns[i]].fillna(df_ARC['-Max'], inplace=True)

    for i in range(12, 22):
        df_ARC[columns[i]].fillna(df_ARC['Max'], inplace=True)

    df_ARC['Datetime'] = pd.to_datetime(df_ARC['Quarter'].str[:16], format=datetime_format)
    df_ARC.drop(['Quarter', '-Max', 'Max'], axis=1, inplace=True)

    return df_ARC

def load_SI_ImbPrice_month(year,month):
    folder_SI_imbPrice_data = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance_price_forecast//Data//SI_NRV_ImbPrice'
    file_loc_SI_imbPrice = folder_SI_imbPrice_data + '//ImbalanceNrvPrices_'+str(year) + string_from_month(month) + '.csv'

    df_SI_imbPrice = pd.read_csv(file_loc_SI_imbPrice, decimal=',')[['NRV', 'SI', 'Alpha', 'PPOS', 'EXECDATE', 'QUARTERHOUR', 'MIP', 'MDP']]
    df_SI_imbPrice['Datetime_string'] = df_SI_imbPrice['EXECDATE'] + ' ' + df_SI_imbPrice['QUARTERHOUR'].str[:5]
    df_SI_imbPrice['Datetime'] = pd.to_datetime(df_SI_imbPrice['Datetime_string'], format=datetime_format)
    df_SI_imbPrice.drop(['EXECDATE', 'QUARTERHOUR', 'Datetime_string'], axis=1, inplace=True)

    return df_SI_imbPrice

def load_act_vol_month(year,month):
    folder_activated_volumes = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance_price_forecast//Data//activated_volumes'
    file_loc_act_vol = folder_activated_volumes + '//ActivatedEnergyVolumes_'+str(year) + string_from_month(month) + '.csv'

    df_act_vol = pd.read_csv(file_loc_act_vol, delimiter=',', decimal=',')[
        ['execdate', 'strQuarter', 'GUV', 'Vol_GCC_I', 'GDV', 'Vol_GCC_D']]
    df_act_vol['Datetime_string'] = df_act_vol['execdate'] + ' ' + df_act_vol['strQuarter'].str[:5]
    df_act_vol['Datetime'] = pd.to_datetime(df_act_vol['Datetime_string'], format=datetime_format)
    df_act_vol.drop(['execdate', 'strQuarter', 'Datetime_string'], axis=1, inplace=True)
    df_act_vol.replace(np.nan, 0, inplace=True)

    return df_act_vol

def load_FC_SI_year(year,months,la,n_quantiles):
    # Get forecasted SI
    #test
    folder_SI_FC = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance_price_forecast//Data//forecast_SI'
    file_loc_FC = folder_SI_FC + '//y_hat_probabilistic_test_combi_'+str(year)+'.csv'

    df_FC_SI = pd.read_csv(file_loc_FC, delimiter=';')

    df_FC_SI['Datetime'] = pd.to_datetime(df_FC_SI['Unnamed: 0'], format=datetime_format)
    df_FC_SI = df_FC_SI[df_FC_SI['Datetime'].dt.month.isin(months)]

    columns_to_keep = ['Datetime']

    for col in range(la*n_quantiles):
        columns_to_keep += [str(col+1)]

    filtered_df_FC_SI = df_FC_SI[columns_to_keep]

    return filtered_df_FC_SI

    ###TO DO: what if we also want market clearing based on forecasted data? do we keep this data?
    #filtered_df_FC_SI_with_LA = df_FC_SI.drop(columns_to_drop, axis=1)
    #col = filtered_df_FC_SI_with_LA.pop('Datetime')
    #filtered_df_FC_SI_with_LA.insert(0, 'Datetime', col)



    # Merge data in large dataframes

    df_merged_FC_SI = pd.merge(filtered_df_FC_SI, df_all, on='Datetime')
    df_merged_with_LA = pd.merge(filtered_df_FC_SI_with_LA, df_all, on='Datetime')

def check_correct_imb_price_calc(df):
    df['imb_price_calc'] = np.where(df['SI'] > 0, df['MDP'] - df['Alpha'], df['MIP'] + df['Alpha'])
    df['Check_calculated_correctly'] = abs(df['imb_price_calc'] - df['PPOS']) < 0.02
    # df_check = df[df['Check_calculated_correctly'] == False]

    discrepancy = df.shape[0] - df[df['Check_calculated_correctly'] == True].shape[0]
    df.drop(['Check_calculated_correctly','imb_price_calc'],axis=1,inplace=True)

    return discrepancy

def add_adjusted_imb_price(df_all):
    discr = check_correct_imb_price_calc(df_all)
    print('A discrepancy was detected in recalculating the imbalance price adjusted for alpha in '+str(discr)+'instances')

    df_all['adjusted_imb_price_alpha'] = np.where(df_all['SI'] > 0, df_all['MDP'], df_all['MIP'])

    return df_all

def add_previous_timestep_imb_price(df):
    df['previous_timestep'] = df['Datetime'] - timedelta(hours=0.25)

    simplified_df = df[['Datetime','adjusted_imb_price_alpha']]
    simplified_df.rename(columns={'adjusted_imb_price_alpha':'adjusted_imb_price_alpha_previous','Datetime': 'Datetime_right'},inplace=True)

    merged_df = pd.merge(df,simplified_df,left_on='previous_timestep',right_on='Datetime_right').drop(columns=['Datetime_right','previous_timestep'],axis=1)
    return merged_df

def check_NRV_GUV_GDV(df_all):
    #Check if GUV/GDV correspond to NRV, and that IGCC is <= GUV or GDV
    df_all['NRV_check'] = abs(df_all['GUV'] - df_all['GDV'] - df_all['NRV']) <0.1
    NRV_check_counter = len(df_all) - df_all['NRV_check'].sum()
    df_all['GUV_check'] = df_all['GUV'] - df_all['Vol_GCC_I'] > -0.01
    GUV_check_counter = len(df_all) - df_all['GUV_check'].sum()
    df_all['GDV_check'] = df_all['GDV'] - df_all['Vol_GCC_D'] > -0.01
    GDV_check_counter = len(df_all) - df_all['GDV_check'].sum()

    return [NRV_check_counter,GUV_check_counter,GDV_check_counter] == [0,0,0]

def data_checks(df):

    NRV_GUV_GDV_check = check_NRV_GUV_GDV(df)
    datetime_uniqueness_check = df['Datetime'].is_unique

    if NRV_GUV_GDV_check & datetime_uniqueness_check:
        print('NRV/GUV/GDV check and datetime uniqueness check ok')
        df.drop(['NRV_check','GUV_check','GDV_check'],axis=1,inplace=True)
    elif NRV_GUV_GDV_check:
        print('datetime uniqueness check not ok')
        sys.exit()
    elif datetime_uniqueness_check:
        print('NRV/GUV/GDV check not ok')
        sys.exit()
    else:
        print('Both NRV/GUV/GDV check and datetime uniqueness check not ok')
        sys.exit()

def store_in_csv(df,folder,addon,train_test_split=False,threshold_day_train_test=20):

    if train_test_split:
        df_train_set = df[(df['Datetime'].dt.day < threshold_day_train_test)]
        df_test_set = df[(df['Datetime'].dt.day >= threshold_day_train_test)]
        #df_with_LA_train = df_merged_with_LA_filtered[(df_merged_with_LA_filtered['Datetime'].dt.day < threshold_day_train_test)]
        #df_with_LA_test = df_merged_with_LA_filtered[(df_merged_with_LA_filtered['Datetime'].dt.day >= threshold_day_train_test)]

        df_train_set.to_csv(folder + '//train_set'+addon+'.csv')
        df_test_set.to_csv(folder + '//test_set'+addon+'.csv')
        #df_with_LA_train.to_csv(folder + '//train_set_with_LA.csv')
        #df_with_LA_test.to_csv(folder + '//test_set_with_LA.csv')

    else:
        df.to_csv(folder + '//data'+addon+'.csv')

def load_data(years,months,excluding_year_months=[],la=1,n_quantiles=9):

    for year in years:
        months_year = get_months_with_exlcusion_list(year,months,excluding_year_months)
        for month in months_year:

            df_ARC = load_ARC_month(year, month)
            df_SI_imbPrice = load_SI_ImbPrice_month(year, month)
            #df_act_vol = load_act_vol_month(year, month)

            df_combined_month = pd.merge(df_ARC, df_SI_imbPrice, on='Datetime')
            #df_combined_month = pd.merge(df_combined_month, df_act_vol, on='Datetime')

            if (month == min(months)):
                df_all_year = df_combined_month
            else:
                df_all_year = pd.concat([df_all_year, df_combined_month])

        #df_FC_SI = load_FC_SI_year(year = year,months=months_year,la=la,n_quantiles=n_quantiles)
        #df_combined_year = pd.merge(df_all_year, df_FC_SI, on='Datetime')

        if (year == min(years)):
            df_all = df_all_year
        else:
            df_all = pd.concat([df_all, df_all_year])

    return df_all

def get_months_with_exlcusion_list(year,months,excluding_year_months):
    months_to_keep = months
    for tuple in excluding_year_months:
        if tuple[0] == year:
            months_to_keep.remove(tuple[1])

    return months_to_keep

def find_average_monthly_values(df_all,years,months):
    avg_MDP_month = np.zeros(len(years) * len(months))
    avg_MIP_month = np.zeros(len(years) * len(months))
    avg_IGCCU_month = np.zeros(len(years) * len(months))
    avg_IGCCD_month = np.zeros(len(years) * len(months))
    avg_imb_price_month = np.zeros(len(years) * len(months))

    for year in range(len(years)):
        for month in range(len(months)):
            start_dt = dt.datetime(year=years[year], month=months[month], day=1)
            end_dt = start_dt + relativedelta(months=1)
            filtered_df = df_all[(df_all['Datetime'] >= start_dt) & (df_all['Datetime'] < end_dt)]

            avg_MDP_month[year * len(months) + month] = filtered_df['MDP'].mean()
            avg_MIP_month[year * len(months) + month] = filtered_df['MIP'].mean()
            avg_IGCCD_month[year * len(months) + month] = filtered_df['Vol_GCC_D'].mean()
            avg_IGCCU_month[year * len(months) + month] = filtered_df['Vol_GCC_I'].mean()
            avg_imb_price_month[year * len(months) + month] = filtered_df['adjusted_imb_price_alpha'].mean()

    return avg_imb_price_month

def find_stdev_monthly_values(df_all,years,months):
    stdev_MDP_month = np.zeros(len(years) * len(months))
    stdev_MIP_month = np.zeros(len(years) * len(months))
    stdev_IGCCU_month = np.zeros(len(years) * len(months))
    stdev_IGCCD_month = np.zeros(len(years) * len(months))
    stdev_imb_price_month = np.zeros(len(years) * len(months))

    for year in range(len(years)):
        for month in range(len(months)):
            start_dt = dt.datetime(year=years[year], month=months[month], day=1)
            end_dt = start_dt + relativedelta(months=1)
            filtered_df = df_all[(df_all['Datetime'] >= start_dt) & (df_all['Datetime'] < end_dt)]

            stdev_MDP_month[year * len(months) + month] = filtered_df['MDP'].std()
            stdev_MIP_month[year * len(months) + month] = filtered_df['MIP'].std()
            stdev_IGCCD_month[year * len(months) + month] = filtered_df['Vol_GCC_D'].std()
            stdev_IGCCU_month[year * len(months) + month] = filtered_df['Vol_GCC_I'].std()
            stdev_imb_price_month[year * len(months) + month] = filtered_df['adjusted_imb_price_alpha'].std()

    return stdev_imb_price_month




if __name__ == '__main__':

    ##### FEATURES SI FORECAST #####

    makedir = False

    idd = {
        'data_file_loc': "../data_preprocessing/data_scaled.h5",
        'read_cols_past_ctxt': ['SI','PV_act','PV_fc','wind_act', 'wind_fc','load_act', 'load_fc'],
        'read_cols_fut_ctxt': ['PV_fc','wind_fc','Gas_fc', 'Nuclear_fc','load_fc'],
        'cols_temp': ['working_day','month_cos','month_sin', 'hour_cos', 'hour_sin', 'qh_cos', 'qh_sin'],
        'target_col': 'SI', #Before: "Frame_SI_norm"
        'datetime_from': datetime(2019,9,1,0,0,0),
        'datetime_to': datetime(2023,9,1,0,0,0),
        'batch_size': 32,
        'list_quantiles': [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
        'tvt_split': [4/7,1/7,2/7],
        'lookahead': 12,
        'lookback': 8,
        'dev': 'cuda',
        'n_components_feat':2, #number of input tensors to neural network for forward pass
        'n_components_lab': 1, #number of input tensors for loss function calc
        'split_val_test': 20, #split up forward pass on validation & test set to avoid memory issues
        #'n_configs': 3, #Number of HP configurations
        'store_code': '20240115_test',
        'epochs': 100,
        'patience': 10,
        'loc_scaler': "../scaling/Scaling_values.xlsx",
        "unscale_labels":True,
        'forecaster_type': 'ED_RNN' # 'ED_RNN' or 'ED_RNN_att'
    }

    #scaler = scaling.Scaler(idd['loc_scaler'])

    df_past_ctxt = fdp.read_data_h5(input_dict=idd, mode='past')#.drop(["FROM_DATE"],axis=1)
    df_fut_ctxt = fdp.read_data_h5(input_dict=idd, mode='fut')#.drop(["FROM_DATE"],axis=1)
    df_temporal = fdp.get_temporal_information(idd)


    ##### MERIT ORDER AND IMBALANCE PRICE #####

    # Define years and months for which you want to load data
    years = [2019,2020,2021,2022,2023]
    months = [i + 1 for i in range(12)]
    excluding_year_month_combi = []

    # Define dates for which time changes happened, and specific days to be deleted from the data set
    dates_timechange_march = [dt.date(2019, 3, 31), dt.date(2020, 3, 29), dt.date(2021, 3, 28), dt.date(2022,3,27), dt.date(2023,3,26)]
    dates_timechange_october = [dt.date(2019, 10, 27), dt.date(2020, 10, 25), dt.date(2021, 10, 31), dt.date(2022,10,30), dt.date(2023,10,29)]
    #days_to_delete = [dt.date(2020, 1, 1), dt.date(2020, 12, 30)]
    days_to_delete = []

    # Define folders where data is located, and where the aggregated data should be stored
    folder_export_data = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance_price_forecast//Data//processed_data'
    addon_string_filename = '_data_2years_excl20200910_LA10'  # This will be included in the file name when storing the data

    # Define datetime format necessary to read data
    datetime_format = "%d/%m/%Y %H:%M"

    # Indicate the amount of forecasted SI instances should be included in dataframe
    lookahead_FC = 10

    # Load ARC, actual SI, NRV, alpha data per month and merge in large dataframe
    df_all = load_data(years=years, months=months, excluding_year_months=excluding_year_month_combi, la=lookahead_FC)
    df_all.set_index('Datetime', inplace=True)
    df_all.drop(columns=['NRV', 'SI', 'Alpha', 'MIP', 'MDP'], inplace=True)

    x=1




    ##### MERGE, PROCESS AND SPLIT #####

    df_MO_price = df_all[~df_all.index.duplicated(keep='first')]

    df_fut_ctxt = df_fut_ctxt.merge(df_MO_price,how='left', left_index=True,right_index=True)

    df_fut_ctxt = df_fut_ctxt.fillna(method='ffill')

    array_past_ctxt = df_past_ctxt.to_numpy()
    array_fut_ctxt = df_fut_ctxt.to_numpy()
    array_temp = df_temporal.to_numpy()

    #Extend arrays (for RNN input)
    array_ext_past_ctxt, array_ext_fut_ctxt,array_ext_past_temp,array_ext_fut_temp = fdp.get_3d_arrays(past_ctxt=array_past_ctxt,fut_ctxt=array_fut_ctxt,temp=array_temp,input_dict=idd)
    labels_ext = fdp.get_3d_arrays_labels(labels = df_past_ctxt,input_dict=idd)

    #MO_ext = array_ext_fut_ctxt[:,:,5:25]
    #price_ext = array_ext_fut_ctxt[:,:,-1]
    #array_ext_fut_ctxt = array_ext_fut_ctxt[:,:,0:5]

    array_ext_past = np.concatenate((array_ext_past_ctxt,array_ext_past_temp),axis=2)
    array_ext_fut = np.concatenate((array_ext_fut_ctxt,array_ext_fut_temp),axis=2)

    feat_train,feat_val,feat_test = fdp.get_train_val_test_arrays([array_ext_past,array_ext_fut],idd)
    lab_train,lab_val,lab_test = fdp.get_train_val_test_arrays([labels_ext],idd)


    MO_test = feat_test[1][:,:,5:25]
    Price_test = feat_test[1][:,:,25]
    feat_test[1] = np.delete(feat_test[1],slice(5,26),axis=2)

    list_arrays = [feat_train,lab_train,feat_val,lab_val,feat_test,lab_test]

    [feat_train_pt, lab_train_pt, feat_val_pt, lab_val_pt, feat_test_pt, lab_test_pt] = set_arrays_to_tensors_device(list_arrays, 'cpu')


    ##### SI FORECASTING AND RETRIEVING PRICE #####

    loc_fc = "20231206_2"
    config = 6
    lookahead = 12
    dir = f'../train_SI_forecaster/output/models_production/LA_{lookahead}/{loc_fc}/'

    dict_pred = {
        'lookahead': lookahead,
        'lookback': 8,
        'loc_SI_FC': f'{dir}/config_{config}.pt',
        'loc_price_FC': ''
    }

    si_forecaster, model_data = load_forecaster(dict=dict_pred,type='imb',dev='cpu')

    SI_FC = si_forecaster(feat_test_pt)
    SI_FC_array = SI_FC.detach().numpy()


    price_fc = np.zeros_like(Price_test)

    for i in range(price_fc.shape[0]):
        avg_price_fc,quantile_price_fc = get_price_fc(SI_FC=SI_FC_array[i,:,:],MO=MO_test[i,:,:],quantiles=model_data['list_quantiles'])
        price_fc[i,:] = avg_price_fc

    print('hello world')


    ##### SAVE RESULTS #####

    # Convert the numpy array to a pandas DataFrame
    df_actual = pd.DataFrame(Price_test)

    # Add column headers
    df_actual.columns = [f"LA:{i}" for i in range(1, df_actual.shape[1] + 1)]

    # Add row indices
    df_actual.index = [f"{i}" for i in range(1, df_actual.shape[0] + 1)]

    # Convert the numpy array to a pandas DataFrame
    df_fc = pd.DataFrame(price_fc)

    # Add column headers
    df_fc.columns = [f"LA:{i}" for i in range(1, df_fc.shape[1] + 1)]

    # Add row indices
    df_fc.index = [f"{i}" for i in range(1, df_fc.shape[0] + 1)]

    df_actual.to_csv("price_act.csv", index=True)
    df_fc.to_csv("price_fc.csv", index=True)



