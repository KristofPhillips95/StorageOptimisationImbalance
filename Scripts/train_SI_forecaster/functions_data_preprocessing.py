import numpy as np
import pandas as pd
import datetime as dt
import sys
from workalendar.europe import Belgium
sys.path.insert(0,"../data_preprocessing")
sys.path.insert(0,"../scaling")
import scaling

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

    df = pd.read_hdf(loc, "data")[cols]

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
        df = pd.read_hdf(loc, 'data')
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

