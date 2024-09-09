import sys
import os
import h5py



current_dir = os.path.dirname(os.path.abspath(__file__))
data_preprocessing_dir = os.path.join(current_dir, '..', '..', '..', 'data_preprocessing')
scaling_dir = os.path.join(data_preprocessing_dir, '..', 'scaling')
train_class_dir = os.path.join(current_dir, '..', '..', 'train_classes')
sys.path.insert(0,data_preprocessing_dir)
sys.path.insert(0,scaling_dir)
sys.path.insert(0,train_class_dir)


from torch_classes import LSTM_ED
import scaling

from workalendar.europe import Belgium
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

    if os.path.isfile(loc):
        pass
    else:
        loc = '../'+loc
    model = torch.load(loc,map_location=dev)
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

def read_data_h5(input_dict, mode):
    """
    Function reading an h5py file into a df, selecting the right columns based on the mode (=past or fut)

    :param input_dict: dictionary containing info on:
        -loc of h5py file: ["data_file_loc"]
        -columns to be filtered by loc: [f"read_cols_{mode}_ctxt"]
    :param mode: 'past' or 'fut', required for selecting the right columns

    :return: dataframe with selected columns
    """
    loc = input_dict["data_file_loc"]
    start_dt = input_dict["datetime_from"]
    end_dt = input_dict["datetime_to"]

    try:
        cols = input_dict[f"read_cols_{mode}_ctxt"]
    except:
        sys.exit("Invalid mode for reading h5py file")

    if os.path.isfile(loc):
        df = pd.read_hdf(loc, "data")[cols]
    else:
        df = pd.read_hdf('../'+loc, "data")[cols]

    return df[(df.index>=start_dt)&(df.index<end_dt)]

def get_temporal_information(input_dict,df=None):
    """
    Function returning pre-processed temporal information for a given period, which coincides to the period of a given
    df or specified in the input dictionary. The temporal datapoints are:
        -Boolean on whether it's a working day
        -cyclic information of:
            -month
            -hour of day
            -qh of hour

    :param input_dict: input dict with information on :
        -location of h5 data file: ['data_file_loc']
        -columns of temporal information to be included in forecasting: ['cols_temp']
    :param df: dataframe covering a period for which we want temporal information. This df should have:
        -datetime objects as index, or:
        -datetime objects in a column named 'datetime'

    :return: dataframe with pre-processed columns on temporal information
    """

    cols = input_dict['cols_temp']
    if df is None:
        start_dt = input_dict["datetime_from"]
        end_dt = input_dict['datetime_to']
        loc = input_dict['data_file_loc']
        if os.path.isfile(loc):
            df = pd.read_hdf(loc, "data")
        else:
            df = pd.read_hdf('../' + loc, "data")
        df = df[(df.index>=start_dt)&(df.index<end_dt)]

    if pd.api.types.is_datetime64_any_dtype(df.index):
        df.reset_index(inplace=True)  # Convert datetime index to numeric
        df.rename(columns={'FROM_DATE': 'datetime'},inplace=True)

    # Binary workday stuff  TODO: this does not capture 'bridge' days, could be improved by assigning values of 0.5 if a workday is in between two non-working days?
    cal_BE = Belgium()
    df['Holiday'] = [int(not cal_BE.is_working_day(x.date())) for x in df.datetime]
    df['dayofweek'] = df.datetime.dt.dayofweek  # 0 to 6 max
    df['weekday'] = convert_binary_dayofweek(df['dayofweek'])
    df['working_day'] = df['weekday'] * (1 - df['Holiday'])

    # Cyclic stuff
    df['month'] = df.datetime.dt.month - 1  # 0 to 11 max
    df['month_cos'], df['month_sin'] = convert_cyclic_info(df['month'])
    df['hour'] = df.datetime.dt.hour  ## 0 to 23max
    df['hour_cos'], df['hour_sin'] = convert_cyclic_info(df['hour'])
    df['qh'] = df.datetime.dt.minute / 15  ## 0 to 3 max Multiple of 15 min
    df['qh_cos'], df['qh_sin'] = convert_cyclic_info(df['qh'])

    df = df[cols]

    return df

def convert_cyclic_info(col):
    """
    Function that takes a df column as input and return two columns with cyclical information
    The input is regarded as vectors rotating on the unit circle

    The linear input information is normalized, which then can be regarded as the angle of the vector
    The angle can be translated to x and y coordinates using the cos and sin transformations, uniquely defining the
        vectors and as such explicitly modeling them as cyclic

    :param col: input column of dataframe
    :return: 2 columns with converted cyclic info
    """

    max_val = col.max()
    min_val = col.min()

    # Normalize data to range [0, 1]
    normalized = (col - min_val) / (max_val + 1 - min_val)

    # Convert normalized data to angle [0, 2*pi]
    angle = normalized * 2 * np.pi

    x = np.cos(angle)
    y = np.sin(angle)

    return x, y

def convert_binary_dayofweek(col):
    """
    Function that takes a df column as input and return a column with binaries indicating whether the day is a weekday (1) or weekend (0)

    Assumption: input organizes work week as 0 = Monday, 6 = Sunday and everything in between

    :param col: input column of dataframe
    :return: col of binaries
    """

    col_bin = (col < 5).astype(int)

    return col_bin

def convert_dict_array(data_dict):
    list_data_arrays = []
    for col in data_dict:
        array = data_dict[col].to_numpy()
        list_data_arrays.append(array)

    concatenated_data = np.concatenate(list_data_arrays, axis=1)

    return concatenated_data

def get_training_indices(list_dataframes, lookahead, lookback):
    list_of_indices = []
    df_0 = list_dataframes[0]

    for index, row in df_0.iterrows():
        if (index <= df_0.shape[0] - lookahead) & (index >= lookback):
            print(f"index: {index}")
            include_index = True
            for df in list_dataframes:
                if row['FROM_DATE'] + dt.timedelta(minutes=(lookahead - 1) * 15) != df['FROM_DATE'][
                    index + lookahead - 1]:
                    include_index = False
                if row['FROM_DATE'] - dt.timedelta(minutes=(lookback) * 15) != df['FROM_DATE'][index - lookback]:
                    include_index = False
            if include_index:
                list_of_indices.append(index)

    return list_of_indices

def get_3d_arrays(past_ctxt, fut_ctxt, temp, input_dict):
    lookahead = input_dict['lookahead']
    lookback = input_dict['lookback']
    n_ex = past_ctxt.shape[0] - lookahead - lookback
    n_cols_past_ctxt = past_ctxt.shape[1]
    n_cols_fut_ctxt = fut_ctxt.shape[1]
    n_cols_temp = temp.shape[1]

    past_ctxt_ext = np.zeros((n_ex, lookback, n_cols_past_ctxt))
    fut_ctxt_ext = np.zeros((n_ex, lookahead, n_cols_fut_ctxt))
    past_temp_ext = np.zeros((n_ex, lookback, n_cols_temp))
    fut_temp_ext = np.zeros((n_ex, lookahead, n_cols_temp))
    past_ctxt_ext_2 = np.zeros((n_ex, lookback, n_cols_past_ctxt))
    past_temp_ext_2 = np.zeros((n_ex, lookback, n_cols_temp))


    for ex in range(n_ex):
        index = ex + lookback

        for lb in range(lookback):
            past_ctxt_ext[ex,lb,:] = past_ctxt[index - lookback + lb, :]
            past_temp_ext[ex, lb, :] = temp[index - lookback +lb, :]



        for la in range(lookahead):
            fut_ctxt_ext[ex, la, :] = fut_ctxt[index + la, :]
            fut_temp_ext[ex, la, :] = temp[index + la, :]

    return past_ctxt_ext, fut_ctxt_ext, past_temp_ext, fut_temp_ext

def get_3d_arrays_labels(labels,input_dict):

    if input_dict['unscale_labels']:
        scaler = scaling.Scaler(input_dict['loc_scaler'])
        labels = scaler.unscale_col(labels,input_dict['target_col'])
        if input_dict['adjust_alpha']:
            labels = scaler.unscale_col(labels,'alpha')

    if input_dict['adjust_alpha']:
        labels['Imb_price'] = labels.apply(lambda row: row['Imb_price'] - row['alpha'] if row['SI'] < 0.5 else row['Imb_price'] + row['alpha'], axis=1)

    labels=labels[input_dict['target_col']].to_numpy()

    lookahead = input_dict['lookahead']
    lookback = input_dict['lookback']
    n_quantiles = len(input_dict['list_quantiles'])
    n_ex = labels.shape[0] - lookahead - lookback
    labels_ext = np.zeros((n_ex, lookahead, n_quantiles))

    for ex in range(n_ex):
        for quant in range(n_quantiles):
            index = ex + lookback
            labels_ext[ex, :, quant] = labels[index:index + lookahead]

    return labels_ext

def get_train_val_test_arrays(list_data, idd):
    """
    Function converting a list of arrays to three lists dividing all arrays into train/validation/test sets
        using specified numerical train/val/test split implementing them chronologically
    Includes a function generating the 'start' and 'stop' indices of the different sets

    :param list_data: a list containing np arrays
    :param idd: dictionary containing info on:
        -train/val/test split: ['tvt_split']
        -lookahead:['lookahead']
        -lookback:['lookback']

    :return: lists of train, val and test arrays
    """
    def get_indices_tvt(data, idd):
        stop_train = int(idd['tvt_split'][0] * data.shape[0])
        start_val = stop_train + max(idd['lookahead'], idd['lookback'])
        stop_val = start_val + int(idd['tvt_split'][1] * data.shape[0])
        start_test = stop_val + max(idd['lookahead'], idd['lookback'])

        return stop_train, start_val, stop_val, start_test

    stop_train, start_val, stop_val, start_test = get_indices_tvt(list_data[0], idd)

    list_train = []
    list_val = []
    list_test = []

    for d in list_data:
        list_train.append(d[0:stop_train])
        list_val.append(d[start_val:stop_val])
        list_test.append(d[start_test:])

    return list_train, list_val, list_test

def scale_si_fc(fc,input_dict):
    scaler = scaling.Scaler(input_dict['loc_scaler'])

    scale_row = 'ACE'
    min = scaler.df_scaling.loc[scaler.df_scaling["Unnamed: 0"] == scale_row, "Min"].values[0]
    max = scaler.df_scaling.loc[scaler.df_scaling["Unnamed: 0"] == scale_row, "Max"].values[0]

    rescaled_fc = (fc-min)/(max-min)
    return rescaled_fc

def get_data_IMB(data_dict):
    # idd = {
    #     'data_file_loc': "../../data_preprocessing/data_qh_SI_imbPrice_scaled.h5",
    #     'read_cols_past_ctxt': ['Imb_price','SI','PV_act','PV_fc','wind_act', 'wind_fc','load_act', 'load_fc'] +[f"-{int((i+1)*100)}MW" for i in range(3)] + [f"{int((i+1)*100)}MW" for i in range(3)],
    #     'read_cols_fut_ctxt': ['PV_fc','wind_fc','Gas_fc', 'Nuclear_fc','load_fc'] + [f"-{int((i+1)*100)}MW" for i in range(10)] + [f"{int((i+1)*100)}MW" for i in range(10)],
    #     'cols_temp': ['working_day','month_cos','month_sin', 'hour_cos', 'hour_sin', 'qh_cos', 'qh_sin'],
    #     'target_col': 'Imb_price', #Before: "Frame_SI_norm"
    #     'datetime_from': datetime(2022,3,1,0,0,0),
    #     'datetime_to': datetime(2024,3,1,0,0,0),
    #     'batch_size': 32,
    #     'list_quantiles': [0.5],
    #     'list_quantiles_SI': [0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99],
    #     'tvt_split': [3/4,1/8,1/8],
    #     'lookahead': 12,
    #     'lookback': 8,
    #     'dev': 'cuda',
    #     #'n_components_feat':2, #number of input tensors to neural network for forward pass
    #     #'n_components_lab': 1, #number of input tensors for loss function calc
    #     #'split_val_test': 20, #split up forward pass on validation & test set to avoid memory issues
    #     #'n_configs': 3, #Number of HP configurations
    #     #'store_code': '20240115_test',
    #     #'epochs': 100,
    #     #'patience': 10,
    #     'loc_scaler': "../../data_preprocessing/scaling/Scaling_values.xlsx",
    #     "unscale_labels":True,
    #     #'forecaster_type': 'ED_RNN' # 'ED_RNN' or 'ED_RNN_att'
    # }

    idd = data_dict

    if idd['adjust_alpha']:
        idd['read_cols_past_ctxt'].append('alpha')


    df_past_ctxt = read_data_h5(input_dict=idd, mode='past')#.drop(["FROM_DATE"],axis=1)
    df_fut_ctxt = read_data_h5(input_dict=idd, mode='fut')#.drop(["FROM_DATE"],axis=1)
    df_temporal = get_temporal_information(idd)

    labels_ext = get_3d_arrays_labels(labels = df_past_ctxt,input_dict=idd)

    if idd['adjust_alpha']:
        df_past_ctxt.drop('alpha', axis=1,inplace=True)





    array_past_ctxt = df_past_ctxt.to_numpy()
    array_fut_ctxt = df_fut_ctxt.to_numpy()
    array_temp = df_temporal.to_numpy()

    #Extend arrays (for RNN input)
    array_ext_past_ctxt, array_ext_fut_ctxt,array_ext_past_temp,array_ext_fut_temp = get_3d_arrays(past_ctxt=array_past_ctxt,fut_ctxt=array_fut_ctxt,temp=array_temp,input_dict=idd)

    #MO_ext = array_ext_fut_ctxt[:,:,5:25]
    #price_ext = array_ext_fut_ctxt[:,:,-1]
    #array_ext_fut_ctxt = array_ext_fut_ctxt[:,:,0:5]

    array_ext_past = np.concatenate((array_ext_past_ctxt,array_ext_past_temp),axis=2)
    array_ext_fut = np.concatenate((array_ext_fut_ctxt,array_ext_fut_temp),axis=2)

    ##### Split features for SI forecasting and imbalance price forecasting #####

    array_ext_past_price = np.concatenate((array_ext_past[:,:,:2],array_ext_past[:,:,8:14]),axis=2) #Select imbalance price (scaled), SI and merit order
    array_ext_fut_price = array_ext_fut[:,:,5:25] #Select merit order

    array_ext_past_SI = np.concatenate((array_ext_past[:,:,1:8],array_ext_past[:,:,14:]),axis=2) #Remove imbalance price and merit order
    array_ext_fut_SI = np.concatenate((array_ext_fut[:,:,:5],array_ext_fut[:,:,25:]),axis=2) #Remove merit order


    ##### SI FORECASTING #####
    feat_train_SI,feat_val_SI,feat_test_SI = get_train_val_test_arrays([array_ext_past_SI,array_ext_fut_SI],idd)
    list_arrays = [feat_train_SI,feat_train_SI,feat_val_SI,feat_val_SI,feat_test_SI,feat_test_SI]
    [feat_train_SI_pt, _, feat_val_SI_pt, _, feat_test_SI_pt, _] = set_arrays_to_tensors_device(list_arrays, 'cpu')

    loc_fc = "20231206_2"
    config = 6
    lookahead = 12
    dir = f'../SI_forecast/trained_models/LA_{lookahead}/{loc_fc}/'

    dict_pred = {
        'lookahead': lookahead,
        'lookback': 8,
        'loc_SI_FC': f'{dir}/config_{config}.pt',
        'loc_price_FC': ''
    }

    si_forecaster, model_data = load_forecaster(dict=dict_pred,type='imb',dev='cpu')

    SI_FC_train = scale_si_fc(si_forecaster(feat_train_SI_pt).detach().numpy(),idd)
    SI_FC_val = scale_si_fc(si_forecaster(feat_val_SI_pt).detach().numpy(),idd)
    SI_FC_test = scale_si_fc(si_forecaster(feat_test_SI_pt).detach().numpy(),idd)


    feat_train_price,feat_val_price,feat_test_price = get_train_val_test_arrays([array_ext_past_price,array_ext_fut_price],idd)
    lab_train_price, lab_val_price, lab_test_price = get_train_val_test_arrays([labels_ext],idd)

    feat_train_price = [feat_train_price[0],np.concatenate((SI_FC_train,feat_train_price[1]),axis=2)]
    feat_val_price = [feat_val_price[0],np.concatenate((SI_FC_val,feat_val_price[1]),axis=2)]
    feat_test_price = [feat_test_price[0],np.concatenate((SI_FC_test,feat_test_price[1]),axis=2)]

    features = [feat_train_price,feat_val_price,feat_test_price]
    labels = [lab_train_price,lab_val_price,lab_test_price]

    return features,labels














