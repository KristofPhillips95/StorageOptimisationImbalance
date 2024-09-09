import pandas as pd
from datetime import date, time, datetime, timedelta
from dateutil.rrule import rrule, MONTHLY, YEARLY
import os
import numpy as np
# Retrieve current working directory (`cwd`)
cwd = os.getcwd()
# Change directory
os.chdir(cwd)

#################################E.g. for dates
#PV
####################################
def import_quarter_hourly_data_load(config_parameters):
    # Ecritrue fichier CSV
    #file_xlsx = r'''file csv\DATA Elia\Load\Global_data_quarter_hourly.xlsx'''
    file_xlsx = f"C:/Users/u0137781/OneDrive - KU Leuven/data/SI_forecasting/Elia/Load/Global_data_quarter_hourly.xlsx"

    file_h5 = config_parameters['file_h5'][0]

    start_trainingData = config_parameters['start_trainingData'][0]
    end_trainingData = config_parameters['end_trainingData'][0]
    df_quarter_hourly = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE', 'Total_Load_real_values','Total_Load_forecasted_values'], dtype=float)
    df_hourly = pd.DataFrame(columns=['FROM_DATE', 'TO_DATE', 'Total_Load_real_values','Total_Load_forecasted_values'], dtype=float)
    #Time change in spring
    DST_in_spring=[datetime(2014, 3, 30,1,0,0),datetime(2015, 3, 29,1,0,0),datetime(2016, 3, 27,1,0,0),datetime(2017, 3, 26,1,0,0),datetime(2018, 3, 25,1,0,0),datetime(2019, 3, 31,1,0,0),datetime(2020, 3, 29,1,0,0),datetime(2021, 3, 28,1,0,0),datetime(2022, 3, 27,1,0,0),datetime(2023, 3, 26,1,0,0)] # - 1 H
    #Time change in fall
    DST_in_fall=[datetime(2014, 10, 26,1,0,0),datetime(2015, 10, 25,1,0,0),datetime(2016, 10, 30,1,0,0),datetime(2017, 10, 29,1,0,0),datetime(2018, 10, 28,1,0,0),datetime(2019, 10, 27,1,0,0),datetime(2020, 10, 25,1,0,0),datetime(2021, 10, 31,1,0,0),datetime(2022, 10, 30,1,0,0),datetime(2023, 10, 29,1,0,0)] # + 1 H

    dates_loop = [dt for dt in rrule(YEARLY, dtstart=start_trainingData, until=end_trainingData)]

    for i in range(len(dates_loop)):
        print(dates_loop[i])
        df_1_quarter_hourly = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
        df_1_hourly = pd.DataFrame()  # Instantiating a dataframe to save data over one excel file
        n_year = dates_loop[i].year
        #file_real_value = r'''file csv\DATA Elia\Load\Total_load_'''+str(n_year)+'''.xlsx'''
        file_real_value = f"C:/Users/u0137781/OneDrive - KU Leuven/data/SI_forecasting/Elia/Load/Total_load_{n_year}.xlsx"


        if os.path.isfile(file_real_value)==True:
            # Load excel file_real_value
            file_real_value_unprocessed=pd.read_excel(file_real_value, header=0, skiprows=1,engine = 'openpyxl')
            file_real_value_unprocessed.rename(columns={"Date/QH": "Date"},inplace=True) #Renaming the column Date/QH
            if '02:00*' in file_real_value_unprocessed.columns:
                file_real_value_unprocessed.drop(['02:00*', '02:15*', '02:30*', '02:45*'], axis=1, inplace=True) # Removing the added hour in fall
            file_real_value_unprocessed['Date'] = file_real_value_unprocessed['Date'].dt.strftime('%d/%m/%Y')
            file_real_value_unprocessed.set_index([file_real_value_unprocessed['Date']], inplace=True) # Index of date (%Y/%M/%D)
            if 'Unnamed: 1' in file_real_value_unprocessed.columns and  'Unnamed: 10' in file_real_value_unprocessed.columns:
                file_real_value_unprocessed.drop(['Unnamed: 1', 'Unnamed: 10','Date'], axis=1, inplace=True) # removing useless columns
            if pd.isnull(file_real_value_unprocessed.columns).any() and 'None.1' in file_real_value_unprocessed.columns and  'Unnamed: 103' in file_real_value_unprocessed.columns:
                file_real_value_unprocessed.drop([file_real_value_unprocessed.columns[pd.isnull(file_real_value_unprocessed.columns) == True][0]], axis=1, inplace=True) # removing useless columns
                file_real_value_unprocessed.drop(['None.1', 'Unnamed: 103', 'Date'], axis=1,
                                             inplace=True)  # removing useless columns
            file_real_value_unprocessed = file_real_value_unprocessed.stack(dropna=False) # Multiple index of date (%Y/%M/%D) and hour (%H/%M) as index
            file_real_value_unprocessed = file_real_value_unprocessed.reset_index(level=[0, 1])
            file_real_value_unprocessed = file_real_value_unprocessed[file_real_value_unprocessed.level_1 != 'Unnamed: 103']
            file_real_value_unprocessed["period"] = file_real_value_unprocessed["Date"]+" "+file_real_value_unprocessed["level_1"]  # create a timestamp in a correct format
            file_real_value_unprocessed.drop(['Date','level_1'], axis=1, inplace=True) # remove useless columns
            file_real_value_unprocessed['period']=pd.to_datetime(file_real_value_unprocessed['period'], format='%d/%m/%Y %H:%M') # column period in datetime format
            file_real_value_unprocessed.set_index('period',inplace=True) # index of date (%Y/%M/%D %H:%M)
            file_real_value_unprocessed.rename(columns={0: "Total_Load"},inplace=True) #Renaming the column to Total_Load
            percent_missing = file_real_value_unprocessed.isnull().sum() * 100 / len(file_real_value_unprocessed) # percentage of missing data per column
            print( 'Missing data (%) : Total_load_ '+str(round(percent_missing['Total_Load'],2)) )
            file_real_value_unprocessed = file_real_value_unprocessed.resample('15T').mean().interpolate(limit_direction='both') # Deals with spring hour and fall hour.
            for date_dst_fall in DST_in_fall:
                if date_dst_fall in file_real_value_unprocessed.index.get_level_values('period'):
                    print('OK ') # resample seems to do the job for removing the added hour in fall
                    #print(file_real_value_unprocessed.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
            for date_dst_spring in DST_in_spring:
                if date_dst_spring in file_real_value_unprocessed.index.get_level_values('period'):
                    print('OK ') # resample seems to do the job for interpolating the missing hour in spring
                    #print(file_real_value_unprocessed.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
            file_real_value_unprocessed_quarter_hourly = file_real_value_unprocessed.copy()
            df_1_quarter_hourly = df_1_quarter_hourly.assign(FROM_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 * mul_min, seconds=0) for mul_min in range(file_real_value_unprocessed_quarter_hourly.shape[0])]))
            df_1_quarter_hourly = df_1_quarter_hourly.assign(TO_DATE=pd.Series([dates_loop[i] + timedelta(days=0, hours=0, minutes=15 + 15 * mul_min, seconds=0) for mul_min in range(file_real_value_unprocessed_quarter_hourly.shape[0])]))
            df_1_quarter_hourly = df_1_quarter_hourly.assign(Total_Load_real_values=file_real_value_unprocessed_quarter_hourly['Total_Load'].values)


        else:
            print('This file doesn t exist', file_real_value)




        file_forecasted_value = r'''file csv\DATA Elia\Load\Total_load_forecast_'''+str(n_year)+'''.xls''' # Begins at 09-08-2014
        file_forecasted_value = f"C:/Users/u0137781/OneDrive - KU Leuven/data/SI_forecasting/Elia/Load/Total_load_forecast_{n_year}.xls"

        if os.path.isfile(file_forecasted_value)==True:
            # Load excel file_forecasted_value
            file_forecasted_value_unprocessed=pd.read_excel(file_forecasted_value, header=0, skiprows=1)
            file_forecasted_value_unprocessed.rename(columns={"Date/QH": "Date"},inplace=True) #Renaming the column Date/QH
            if '02:00*' in file_forecasted_value_unprocessed.columns:
                file_forecasted_value_unprocessed.drop(['02:00*', '02:15*', '02:30*', '02:45*'], axis=1, inplace=True) # Removing the added hour in fall
            file_forecasted_value_unprocessed['Date'] = file_forecasted_value_unprocessed['Date'].dt.strftime('%d/%m/%Y')
            file_forecasted_value_unprocessed.set_index([file_forecasted_value_unprocessed['Date']], inplace=True) # Index of date (%Y/%M/%D)
            file_forecasted_value_unprocessed.drop(['Unnamed: 1', 'Unnamed: 10','Date'], axis=1, inplace=True) # removing useless columns
            file_forecasted_value_unprocessed = file_forecasted_value_unprocessed.stack(dropna=False) # Multiple index of date (%Y/%M/%D) and hour (%H/%M) as index
            file_forecasted_value_unprocessed = file_forecasted_value_unprocessed.reset_index(level=[0, 1])
            file_forecasted_value_unprocessed["period"] = file_forecasted_value_unprocessed["Date"]+" "+file_forecasted_value_unprocessed["level_1"]  # create a timestamp in a correct format
            file_forecasted_value_unprocessed.drop(['Date','level_1'], axis=1, inplace=True) # remove useless columns
            file_forecasted_value_unprocessed['period']=pd.to_datetime(file_forecasted_value_unprocessed['period'], format='%d/%m/%Y %H:%M') # column period in datetime format
            file_forecasted_value_unprocessed.set_index('period',inplace=True) # index of date (%Y/%M/%D %H:%M)
            file_forecasted_value_unprocessed.rename(columns={0: "Total_Load"},inplace=True) #Renaming the column to Total_Load
            percent_missing = file_forecasted_value_unprocessed.isnull().sum() * 100 / len(file_forecasted_value_unprocessed) # percentage of missing data per column
            print( 'Missing data (%) : Total_load_forecast_ '+str(round(percent_missing['Total_Load'],2)) )
            if percent_missing['Total_Load'] > 30: #If missing values over 30%
                file_forecasted_value_unprocessed.fillna(file_real_value_unprocessed, inplace=True)
            file_forecasted_value_unprocessed = file_forecasted_value_unprocessed.resample('15T').mean().interpolate(limit_direction='both') # Deals with spring hour and fall hour.
            for date_dst_fall in DST_in_fall:
                if date_dst_fall in file_forecasted_value_unprocessed.index.get_level_values('period'):
                    print('OK ') # resample seems to do the job for removing the added hour in fall
                    #print(file_forecasted_value_unprocessed.loc[date_dst_fall:date_dst_fall+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
            for date_dst_spring in DST_in_spring:
                if date_dst_spring in file_forecasted_value_unprocessed.index.get_level_values('period'):
                    print('OK ') # resample seems to do the job forinterpolating the missing hour in spring
                    #print(file_forecasted_value_unprocessed.loc[date_dst_spring:date_dst_spring+timedelta(days=0, hours=0, minutes=15 * 8, seconds=0)]) # Verification
            file_forecasted_value_unprocessed_quarter_hourly = file_forecasted_value_unprocessed.copy()

            df_1_quarter_hourly = df_1_quarter_hourly.assign(Total_Load_forecasted_values=file_forecasted_value_unprocessed_quarter_hourly['Total_Load'].values)
            df_quarter_hourly = df_quarter_hourly.append(df_1_quarter_hourly, ignore_index=True)

        else:
            print('This file doesn t exist', file_forecasted_value)



    df_quarter_hourly = df_quarter_hourly.set_index('FROM_DATE')
    # Ecritrue fichier CSV
    open(file_xlsx, 'w+')
    df_quarter_hourly.to_excel(file_xlsx)
    # break
    df_quarter_hourly.to_hdf(file_h5, 'Total_Load')



















if __name__ == '__main__':
    config_parameters = pd.DataFrame({'start_trainingData': [datetime(2014, 1, 1)],
                                      'end_trainingData': [datetime(2024, 2, 29)],
                                      'file_h5': 'data_qh_SI_imbPrice.h5',
                                      })
    import_quarter_hourly_data_load(config_parameters)
