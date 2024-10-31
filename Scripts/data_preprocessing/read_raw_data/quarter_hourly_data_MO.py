import pandas as pd
from datetime import date, time, datetime, timedelta
from dateutil.rrule import rrule, MONTHLY, YEARLY
import os
import math
# Retrieve current working directory (`cwd`)
cwd = os.getcwd()
# Change directory
os.chdir(cwd)

def load_ARC_month(year,month):

    folder_ARC_data = 'C://Users//u0137781//OneDrive - KU Leuven//Imbalance_price_forecast//Data//meritOrder'
    file_loc_ARC = folder_ARC_data + '//ARC_VolumeLevelPrices_'+str(year)+'_'+string_from_month(month)+'.csv'

    df_ARC = pd.read_csv(file_loc_ARC, delimiter=';', decimal=',')
    df_ARC = df_ARC.rename({'#NAME?': '-Max'}, axis=1)
    columns = list(df_ARC.columns)

    datetime_format = '%d/%m/%Y %H:%M'

    for i in range(2, 12):
        df_ARC[columns[i]].fillna(df_ARC['-Max'], inplace=True)

    for i in range(12, 22):
        df_ARC[columns[i]].fillna(df_ARC['Max'], inplace=True)

    df_ARC['Datetime'] = pd.to_datetime(df_ARC['Quarter'].str[:16], format=datetime_format)
    df_ARC.drop(['Quarter', '-Max', 'Max'], axis=1, inplace=True)

    return df_ARC


def import_quarter_hourly_MO(config_parameters):
    file_xlsx =r'''file csv\DATA Elia\RES\pv\Global_data.xlsx'''
    file_xlsx = f"{config_parameters['loc_data'][0]}/Elia/RES/pv/Global_data.xlsx"
    file_h5 = config_parameters['file_h5'][0]

    datetime_format = '%d/%m/%Y %H:%M'



    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H

    dates_loop = [dt for dt in rrule(MONTHLY, dtstart=start_trainingData, until=end_trainingData)]
    for i in range(len(dates_loop)):
        print(dates_loop[i])
        n_year = dates_loop[i].year
        n_month = '{:02d}'.format(dates_loop[i].month)
        file = f"{config_parameters['loc_data'][0]}/Elia/balancing_MO/ARC_VolumeLevelPrices_{n_year}_{n_month}.csv"

        if os.path.isfile(file) == True:
            # Load excel file
            df_ARC = pd.read_csv(file, delimiter=';', decimal=',')
            df_ARC = df_ARC.rename({'#NAME?': '-Max'}, axis=1)

            columns = list(df_ARC.columns)


            for i in range(2, 12):
                df_ARC[columns[i]].fillna(df_ARC['-Max'], inplace=True)

            for i in range(12, 22):
                df_ARC[columns[i]].fillna(df_ARC['Max'], inplace=True)


            df_ARC['DateTime'] = pd.to_datetime(df_ARC['Quarter'].str[:16], format=datetime_format)
            df_ARC.set_index('DateTime',inplace=True) # column Date/Time as index
            # file_unprocessed = file_unprocessed[file_unprocessed.index.month == dates_loop[i].month]  # select only the beginning of each day
            # if 'Week-Ahead forecast [MW]' in file_unprocessed.columns:
            #     file_unprocessed.drop(['Week-Ahead forecast [MW]'], axis=1, inplace=True)
            # if 'Most recent forecast [MW]' in file_unprocessed.columns:
            #     file_unprocessed.rename(columns={'Most recent forecast [MW]': 'Intraday forecast [MW]'}, inplace=True)
            # percent_missing = file_unprocessed.isnull().sum() * 100 / len(file_unprocessed) # percentage of missing data per column
            # print( 'Missing data (%) : DA '+str(round(percent_missing['Day-Ahead forecast [MW]'],2))+', RT '+str(round(percent_missing['Real-time Upscaled Measurement [MW]'],2)) )
            #
            df_ARC.drop(['Quarter'], axis=1, inplace=True)
            df_ARC = df_ARC.resample('15min').mean().interpolate(limit_direction='both') # Deals with spring hour and fall hour.
            for date_dst_fall in DST_in_fall:
                if date_dst_fall in df_ARC.index.get_level_values('DateTime'):
                    print('OK ') # resample seems to do the job for removing the added hour in fall
                    #print(file_unprocessed.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
            for date_dst_spring in DST_in_spring:
                if date_dst_spring in df_ARC.index.get_level_values('DateTime'):
                    print('OK ') # resample seems to do the job for interpolating the missing hour in spring
                    #print(file_unprocessed.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification

            if 'df' in locals() or 'df' in globals():
                df = df._append(df_ARC)
            else:
                df = df_ARC.copy(deep=True)
        else:
            print('This file doesn t exist',file)

    df['FROM_DATE'] = df.index

    df.to_hdf(file_h5, 'DATA_ELIA_MO')

if __name__ == '__main__':
    config_parameters = pd.DataFrame({'start_trainingData': [datetime(2016, 1, 1)],
                                      'end_trainingData': [datetime(2024, 2, 29)],
                                      'file_h5': 'data_qh_SI_imbPrice.h5',
                                      })
    import_quarter_hourly_MO(config_parameters)