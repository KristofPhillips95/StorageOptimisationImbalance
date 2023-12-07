import datetime as dt

import pandas as pd

import elia
import time
import numbers
import sys
import pytz
import numpy as np
###################
#Helper Methods
###################


def get_dataframes(list_data,start,end):

    list_df = []

    for i,datapoint in enumerate(list_data):

        print(f"Start loading {datapoint}")
        df = get_specific_df(datapoint,start,end)
        list_df.append(df)
        print(f"loaded {datapoint}")
        if i == 0:
            df_all = df
        else:
            df_all = df_all.merge(df,on='datetime')
            #df_all = pd.concat([df_all, df],keys="datetime", axis=1)

    return df_all

def get_specific_df(datapoint,start,end):
    #TODO: check correctness of data

    def filter_df_time(df,start,end):
        df = df[(df['datetime'] <= end) & (df['datetime'] >= start)]
        return df

    if datapoint == 'wind_act':

        df_wind = connection.get_historical_wind_power_estimation_and_forecast_own(start=start, end=end)
        df = aggregate_renewables(df_raw=df_wind, res_type='wind',datapoint='rt')
        df.rename(columns={'realtime': datapoint},inplace=True)

    elif datapoint == 'PV_act':
        df_solar = connection.get_historical_solar_power_estimation_and_forecast_own(start=start, end=end)
        df = aggregate_renewables(df_raw=df_solar, res_type='solar',datapoint='rt')
        df.rename(columns={'realtime': datapoint},inplace=True)

    elif datapoint == 'SI':
        """
        This only yields the appropriate result if the published SI data per minute shows data in the quarter hour that
        you are requesting. Since there is a lag of around 2/3 minutes for SI data per minute, this will yield a dataframe
        with missing data of the last past quarter hour if you request this data in the beginning of said qh
        """
        #TODO: fix such that this works regardless of the time within the qh this is requested
        #TODO: average of SI minutely does not exactly correspond to quarter hourly SI; figure out where this comes from
        df_SI_quarter = connection.get_imbalance_prices_per_quarter_hour_own(start=start, end=end)
        df_SI_min = connection.get_current_system_imbalance()
        df = complement_SI(df_qh=df_SI_quarter, df_min=df_SI_min)

        df.rename(columns={'systemimbalance': datapoint},inplace=True)

    elif datapoint == 'load_act':
        df_load = connection.get_load_on_elia_grid(start=start, end=end)
        df = process_load(df_load,datapoint='rt')
        df.rename(columns={'measured': datapoint},inplace=True)

    elif datapoint == 'wind_fc':
        df_wind = connection.get_historical_wind_power_estimation_and_forecast_own(start=start, end=end)
        df = aggregate_renewables(df_raw=df_wind, res_type='wind',datapoint='da_f')
        df.rename(columns={'dayahead11hforecast': datapoint},inplace=True)

    elif datapoint == 'PV_fc':
        df_solar = connection.get_historical_solar_power_estimation_and_forecast_own(start=start, end=end)
        df = aggregate_renewables(df_raw=df_solar, res_type='solar',datapoint='da_f')
        df.rename(columns={'dayahead11hforecast': datapoint},inplace=True)

    elif datapoint == 'load_fc':
        df_load = connection.get_load_on_elia_grid(start=start, end=end)
        df = process_load(df_load, datapoint='da_f')
        df.rename(columns={'dayaheadforecast': datapoint}, inplace=True)

    elif datapoint == 'Nuclear_fc':
        df_DA_FT = connection.get_DA_schedule_by_fuel(start=start, end=end)
        df = aggregate_conventional(df_raw=df_DA_FT, list_gen_types=['NU'])
        df.rename(columns={'NU': datapoint},inplace=True)

    elif datapoint == 'Gas_fc':
        df_DA_FT = connection.get_DA_schedule_by_fuel(start=start, end=end)
        df = aggregate_conventional(df_raw=df_DA_FT, list_gen_types=['NG'])
        df.rename(columns={'NG': datapoint},inplace=True)

    elif datapoint == 'MO':
        df_ARC_MO_raw = connection.get_ARC_merit_order(start=start, end=end)
        df = convert_raw_ARC(df_raw=df_ARC_MO_raw)

    elif datapoint == 'SI_and_price':
        df = connection.get_imbalance_prices_per_quarter_hour_own(start=start, end=end)
        df.reset_index(inplace=True)


    else:
        raise ValueError(f'Unsupported datapoint {datapoint}')



    df = filter_df_time(df,start,end)

    return df

def get_start_and_end_time_data(attr):
    if attr in ["SI","CBF"]:
        delta = dt.timedelta(hours=5)
        end = dt.datetime.now()
        start = end -delta
    elif attr in ["DA_FT","load"]:
        delta = dt.timedelta(hours=24)
        end = dt.datetime.now()
        start = end - delta
    elif attr in ["RES"]:
        delta = dt.timedelta(hours=72)
        end = dt.datetime.now() + dt.timedelta(hours=24)
        start = end - delta
    elif attr in ["ARC","net_pos_DA"]:
        delta = dt.timedelta(hours=72)
        end = dt.datetime.now() + dt.timedelta(hours=48)
        start = end - delta
    else:
        raise ValueError(f"Start and end time not configured for {attr}")
    return start, end

def convert_raw_ARC(df_raw):
    """Returns dataframe of ARC merit order where volume levels as columns."""
    #TODO: check if merit order is monotonically increasing

    def fill_gaps(val_list):
        """Changes the 'NA' values of the input list to numerical values based on its surrounding values"""
        #TODO: remove dependence of datetime being the first element of the list

        list_floats = []
        for val in val_list:
            if isinstance(val,numbers.Number):
                list_floats.append(val)

        for i, val in enumerate(val_list):
            if val == 'NA':
                if isinstance(val_list[i-1],numbers.Number):
                    if isinstance(val_list[min(i+1,len(val_list)-1)],numbers.Number):
                        val_list[i] = (val_list[i-1] + val_list[i+1])/2
                    else:
                        val_list[i] = val_list[i-1]
                else:
                    if isinstance(val_list[min(i+1,len(val_list)-1)],numbers.Number):
                        val_list[i] = val_list[i+1]
                    else:
                        val_list[i] = min(list_floats)

        return val_list


    #Define columns of return dataframe as datetime, and 1 column per volume level
    columns = ['datetime'] + [str(-1000+i*100) for i in range(21)]
    columns.remove('0')
    df = pd.DataFrame(columns=columns)

    #Retrieve list of timestamps and set them as a separate column in raw df
    timestamps = df_raw.index.unique()
    df_raw.reset_index(inplace=True)

    #For every unique timestamp, add row with price per volumelevel to return df
    for i,ts in enumerate(timestamps):
        df_filt = df_raw[df_raw['datetime']==ts]
        val_list=[]
        for col in columns:
            if col == 'datetime':
                val_list.append(ts)
            else:
                try:
                    val = df_filt[df_filt['volumelevel']==col]['energybidmarginalprice'].values[0]
                except:
                    val = 'NA'
                val_list.append(val)

        if 'NA' in val_list:
            val_list = fill_gaps(val_list)

        df.loc[i] = val_list

    return df

def aggregate_renewables(df_raw,res_type,datapoint):
    """
    Converts raw elia dataframes with observed and forecasted renewable production to usable dataframes
    Includes data enhancement of unknown values if datapoint is measured real-time production

    :param df_raw: pandas dataframe retrieved from Elia data platform
    :param datapoint: indicates which column we want to retrieve, str 'da_f' (day-ahead forecast) or 'rt' (measures real-time values)

    :return: dataframe with column 'datetime' and datapoint
    """
    #TODO: for past context, use difference of actual vs. forecast RES output?

    def  get_value_column(df_filt,col,res_type):
        """
        Gets the aggregate value of renewables based on the standard form published by Elia
        """

        if res_type == 'solar':
            val = df_filt[df_filt['region']=='Belgium'][col].values[0]
        elif res_type == 'wind':
            try:
                val = sum(df_filt[col].values)
            except:
                val = None
        else:
            sys.exit('Invalid type')

        return val

    def estimate_missing_values(df,res_type):
        """
        Fill up values of realtime renewable production that are not published
        Implement continuation of absolute (to avoid issues with 0 forecast) difference latest known measured production vs last forecast
        """

        if res_type=='wind':
            """
            Dataframe on wind shows values NA for the timestamps that have not been recorded/published yet.
            Logic: on the time stamps where 'realtime' == NA, fill data with most recent absolute difference
            """
            #Filter out nan
            df_limited = df[df['realtime'].notnull()]
            last_row = df_limited.loc[df_limited['datetime'].idxmax()]
            rt_vs_mrf = last_row['realtime']-last_row['mostrecentforecast']

            #fill nan values
            mask = df['realtime'].isna()
            df.loc[mask, 'realtime'] = np.maximum(df.loc[mask, 'mostrecentforecast'] + rt_vs_mrf, 0)

            return df[['datetime', 'realtime']]

        elif res_type == 'solar':
            """
            Dataframe on PV shows values 0 for the timestamps that have not been recorded/published yet.
            Logic: find out of last nonzero RT value happened before or after last zero most recent forecast
                if before: it seems like we are in the night. --> fill with most recent forecast
                if after: fill zero values by applying the last known absolute different with the most recent forecast
                          only changing the zero values after the last known nonzero RT value
                TODO: test if this works
            """

            df_known_values = df[df['realtime'] != 0]
            df_expected_zero = df[df['mostrecentforecast'] == 0]

            dt_last_nonzero_rt = df_known_values['datetime'].max()
            dt_last_zero_fc = df_expected_zero['datetime'].max()

            #Replace the last known datetimes with a reference in the past if it doesn't exist
            ref = dt.datetime(2000,1,1,0,0,0).astimezone(pytz.timezone('GMT'))
            if pd.isna(dt_last_nonzero_rt):
                dt_last_nonzero_rt = ref
            if pd.isna(dt_last_zero_fc):
                dt_last_zero_fc = ref

            if dt_last_nonzero_rt > dt_last_zero_fc:
                last_row = df_known_values.loc[df_known_values['datetime'].idxmax()]
                rt_vs_mrf = last_row['realtime'] - last_row['mostrecentforecast']
                mask = (df['realtime']==0) & (df['datetime'] > dt_last_nonzero_rt)
                df.loc[mask, 'realtime'] = np.maximum(df.loc[mask, 'mostrecentforecast'] + rt_vs_mrf, 0)

            else:
                mask = (df['realtime']==0) & (df['datetime'] > dt_last_nonzero_rt)
                df.loc[mask, 'realtime'] = df.loc[mask, 'mostrecentforecast']

            return df[['datetime','realtime']]

    columns = ['datetime','realtime', 'mostrecentforecast','dayahead11hforecast']
    df = pd.DataFrame(columns=columns)

    timestamps = df_raw.index.unique()
    df_raw.reset_index(inplace=True)

    for i,ts in enumerate(timestamps):
        df_filt = df_raw[df_raw['datetime']==ts]
        val_list=[]
        for col in columns:
            if col == 'datetime':
                val_list.append(ts)
            else:
                val = get_value_column(df_filt,col,res_type)
                val_list.append(val)
        df.loc[i] = val_list


    if datapoint == 'da_f':
        df_out =  df[['datetime','dayahead11hforecast']]
    elif datapoint == 'rt':
        df_out =  df[['datetime','realtime','mostrecentforecast']]
        df_out = estimate_missing_values(df_out,res_type) #This can be used to enhance data if we want to
    else:
        sys.exit("Invalid datapoint")

    return df_out

def process_load(df_raw,datapoint):
    """
    Converts raw elia dataframes with observed and forecasted load to usable dataframes
    Includes data enhancement of unknown values if datapoint is measured real-time load

    :param df_raw: pandas dataframe retrieved from Elia data platform
    :param datapoint: indicates which column we want to retrieve, str 'da_f' (day-ahead forecast) or 'rt' (measures real-time values)

    :return: dataframe with column 'datetime' and datapoint
    """

    def estimate_missing_values(df):
        """"
        Implement continuation of relative difference latest known measured load vs last forecast
        """

        #Filter out nan
        df_limited = df[df['measured'].notnull()]
        try:
            last_row = df_limited.loc[df_limited['datetime'].idxmax()]
            rt_vs_mrf_rel = last_row['measured'] / last_row['mostrecentforecast']
        except:
            print("Warning: no latest measured load available, reverting to most recent forecast")
            rt_vs_mrf_rel = 1

        #fill nan values
        mask = df['measured'].isna()
        df.loc[mask, 'measured'] = df.loc[mask, 'mostrecentforecast'] * rt_vs_mrf_rel

        return df[['datetime', 'measured']]

    columns = ['datetime', 'measured', 'mostrecentforecast','dayaheadforecast']
    df_raw.index.name='datetime'
    df = df_raw.reset_index()
    df = df[columns]

    if datapoint == 'da_f':
        df_out =  df[['datetime','dayaheadforecast']]
    elif datapoint == 'rt':
        df_out =  df[['datetime','measured','mostrecentforecast']]
        df_out = estimate_missing_values(df_out) #This can be used to enhance data if we want to
    else:
        sys.exit("Invalid datapoint")

    return df_out

def complement_SI(df_qh,df_min):
    df_qh.reset_index(inplace=True)
    df_min.reset_index(inplace=True)

    now = dt.datetime.now().astimezone(pytz.timezone('GMT'))
    last_known_qh = df_qh.loc[df_qh['datetime'].idxmax()]['datetime']
    current_qh = last_known_qh + dt.timedelta(minutes=15)

    min_filt=df_min.loc[df_min['datetime']>=current_qh]['systemimbalance'].tolist()

    n_elements = len(min_filt)
    trend = (min_filt[-1]-min_filt[0])/(n_elements-1)

    for i in range(15-n_elements):
        min_filt.append(min_filt[-1]+trend)

    est_SI_current = sum(min_filt)/len(min_filt)

    #Add row to quarter hourly dataframe with estimated SI value for current qh
    latest_qh = last_known_qh + dt.timedelta(minutes=15)
    return_df = df_qh[['datetime','systemimbalance']].copy()
    return_df.loc[len(df_qh)] = [latest_qh,est_SI_current]

    return return_df

def aggregate_conventional(df_raw,list_gen_types):

    columns = ['datetime'] + list_gen_types
    df = pd.DataFrame(columns=columns)

    timestamps = df_raw.index.unique()
    df_raw.reset_index(inplace=True)


    for i,ts in enumerate(timestamps):
        list_val = [ts]
        df_filt = df_raw[df_raw['datetime']==ts]
        for gen_type in list_gen_types:
            list_val.append(df_filt[df_filt['fuelcode']==gen_type]['dayaheadgenerationschedule'].values[0])
        df.loc[i] = list_val

    return df

# def filter_df_timeframe(df,tf,ref=None):
#     """
#     Selects the rows of a dataframe in past or future relative to a reference
#
#     :param df: dataframe to be filtered, requires a column named 'datetime'
#     :param tf: timeframe, should be string 'past' or 'future'
#     :param ref: reference compared to which the df is filtered. Should be datetime object with timezone information
#                 if no reference provided, 'now' will be used
#
#     :return: filtered df
#     """
#
#     if ref is None:
#         ref = dt.datetime.now().astimezone(pytz.timezone('GMT'))
#
#     if tf == 'past':
#         df_filtered = df[df['datetime']<=ref]
#     elif tf == 'fut':
#         df_filtered = df[df['datetime']>ref]
#     else:
#         sys.exit('Invalid timeframe')
#
#     return df_filtered

######################
#Create connection object
######################
connection = elia.EliaPandasClient()



if __name__ == '__main__':

    ########################################
    # Cross border flows
    ########################################
    print("Starting timer for border_flows")
    timer = time.time()
    start,end = get_start_and_end_time_data("CBF")
    df_CB = connection.get_cross_border_flows_per_quarter_hour(start = start,end=end)
    print(f"Time spent for CB equals {time.time()-timer}")

    start,end = get_start_and_end_time_data("net_pos_DA")
    df_net_pos = connection.get_DA_net_pos(start=start,end=end)

    ####################################################
    #Conventional generation forecast by fuel type
    #####################################################
    start,end = get_start_and_end_time_data("DA_FT")
    df_DA_FT = connection.get_DA_schedule_by_fuel()



    df_filtered = aggregate_conventional(df_raw=df_DA_FT,list_gen_types=['NU','NG'])



    ####################################################
    #System imbalance information
    #####################################################
    start,end = get_start_and_end_time_data("SI")
    # df_SI_quarter = connection.get_imbalance_prices_per_quarter_hour(start=start,end=end)
    df_SI_quarter = connection.get_imbalance_prices_per_quarter_hour_own(start=start,end=end)

    df_SI_min = connection.get_imbalance_prices_per_min()

    df_SI_quarter = complement_SI(df_qh=df_SI_quarter,df_min=df_SI_min)

    ####################################################
    #System load
    #####################################################
    start,end = get_start_and_end_time_data("load")

    df_load = connection.get_load_on_elia_grid(start=start, end=end)
    df_load = connection.get_load_on_elia_grid(start=start, end=end)

    ####################################################
    #RES Forecast
    #####################################################
    start,end = get_start_and_end_time_data("RES")

    #Note: these NRT datasets only go back about half a day I think, but it should be possible to use another dataset with historical values to go back further
    df_wind = connection.get_historical_wind_power_estimation_and_forecast_own(start=start,end=end)
    df_solar = connection.get_historical_solar_power_estimation_and_forecast_own(start=start,end=end)

    df_wind_past,df_wind_fut = aggregate_renewables(df_raw=df_wind,res_type='wind')
    df_solar_past,df_solar_fut = aggregate_renewables(df_raw=df_solar,res_type='solar')

    ####################################################
    #DA Prices
    #####################################################


    ####################################################
    #Merit order
    #####################################################
    start,end = get_start_and_end_time_data("ARC")
    #df_merit_decr = connection.get_merit_order_decremental(start=start,end=end)
    #df_merit_incr = connection.get_merit_order_decremental(start=start,end=end)
    df_ARC_MO_raw = connection.get_ARC_merit_order(start=start,end=end)
    df_ARC_MO = convert_raw_ARC(df_raw=df_ARC_MO_raw)

    df_solar = connection.get_historical_solar_power_estimation_and_forecast_own(start=start,end=end)

