import datetime as dt

import pandas as pd

import elia
import time
import numbers
import sys
import pytz

###################
#Helper Methods
###################

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
    elif attr in ["ARC"]:
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

def aggregate_renewables(df_raw,res_type):
    """
    Converts raw elia dataframes with renewable production and forecast to usable dataframes
    output: separated dataframes for past context (= actual realizations of renewable production) and future context (=11am DA RES forecast)
    """

    def  get_value_column(df_filt,col,res_type):
        """
        Fills up missing values of 'realtime' column of a dataframe
        For estimating the values, the latest available ratio of realtime production and most recent forecast is used
        """

        if res_type == 'solar':
            val = df_filt[df_filt['region']=='Belgium'][col].values[0]
        elif res_type == 'wind':
            val = sum(df_filt[col].values)
        else:
            sys.exit('Invalid type')

        return val

    def estimate_missing_values(df):
        #Filter out nan
        df_limited = df[df['realtime'].notnull()]
        last_row = df_limited.loc[df_limited['datetime'].idxmax()]
        rt_vs_mrf = last_row['realtime']/last_row['mostrecentforecast']

        #fill nan values
        df['realtime'] = df['realtime'].mask(df['realtime'].isna(),df['mostrecentforecast']*rt_vs_mrf)

        return df[['datetime', 'realtime']]


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

    now = dt.datetime.now().astimezone(pytz.timezone('GMT'))
    df_fut = df[df['datetime']>=now][['datetime','dayahead11hforecast']]
    df_past = df[df['datetime']<=now][['datetime','realtime','mostrecentforecast']]
    df_past = estimate_missing_values(df_past)

    return df_past,df_fut

######################
#Create connection object
######################
connection = elia.EliaPandasClient()


########################################
# Cross border flows
########################################
print("Starting timer for border_flows")
timer = time.time()
start,end = get_start_and_end_time_data("CBF")
df_CB = connection.get_cross_border_flows_per_quarter_hour(start = start,end=end)
print(f"Time spent for CB equals {time.time()-timer}")

####################################################
#Conventional generation forecast by fuel type
#####################################################
start,end = get_start_and_end_time_data("DA_FT")
df_DA_FT = connection.get_DA_schedule_by_fuel()

####################################################
#System imbalance information
#####################################################
start,end = get_start_and_end_time_data("SI")
# df_SI_quarter = connection.get_imbalance_prices_per_quarter_hour(start=start,end=end)
df_SI_quarter = connection.get_imbalance_prices_per_quarter_hour_own(start=start,end=end)

df_SI_min = connection.get_imbalance_prices_per_min()

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




